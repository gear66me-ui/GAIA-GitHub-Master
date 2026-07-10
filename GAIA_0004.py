# GAIA_0004
# Equal-area, sector-balanced, Sun-centered Gaia DR3 stellar bubble.

from __future__ import annotations

import importlib
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "GAIA_0004"
OUTPUT_DIR = Path(os.environ.get("GAIA_OUTPUT_DIR", "/content/GAIA_OUTPUT"))
CACHE_DIR = OUTPUT_DIR / "GAIA_0004_SECTOR_CACHE"
ROWS_PER_SECTOR = int(os.environ.get("GAIA_ROWS_PER_SECTOR", "80"))
MAX_DISTANCE_KPC = float(os.environ.get("GAIA_MAX_DISTANCE_KPC", "60.0"))
MAX_RUWE = float(os.environ.get("GAIA_MAX_RUWE", "1.4"))
QUERY_RETRIES = int(os.environ.get("GAIA_QUERY_RETRIES", "4"))
LONGITUDE_SECTORS = 8
LATITUDE_SECTORS = 4
LUMINOSITY_MIN = 0.001
LUMINOSITY_MAX = 8.0
LUMINOSITY_LEVELS = 8
GALACTIC_CENTER_DISTANCE_KPC = 8.2

LEVEL_COLORS = {
    1: "#2447FF",
    2: "#00A8FF",
    3: "#00E5FF",
    4: "#00FF9D",
    5: "#7CFF00",
    6: "#FFF000",
    7: "#FF8A00",
    8: "#FF245E",
}


def colombia_timestamp() -> str:
    return datetime.now(ZoneInfo("America/Bogota")).strftime(
        "%Y-%m-%d %H:%M:%S America/Bogota"
    )


def compact_error(error: Exception) -> str:
    return " ".join(str(error).split())[:260]


def ensure_dependencies() -> None:
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "astroquery": "astroquery",
    }
    missing = []
    for module_name, package_name in required.items():
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(package_name)
    if missing:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *missing],
            check=True,
        )


def longitude_edges(np):
    return np.linspace(0.0, 360.0, LONGITUDE_SECTORS + 1)


def latitude_edges(np):
    sin_edges = np.linspace(-1.0, 1.0, LATITUDE_SECTORS + 1)
    return np.degrees(np.arcsin(sin_edges))


def sector_cache_path(l_index: int, b_index: int) -> Path:
    return CACHE_DIR / f"sector_l{l_index:02d}_b{b_index:02d}.csv"


def build_sector_query(
    l_min: float,
    l_max: float,
    b_min: float,
    b_max: float,
    last_l: bool,
    last_b: bool,
) -> str:
    l_upper_operator = "<=" if last_l else "<"
    b_upper_operator = "<=" if last_b else "<"
    maximum_distance_pc = MAX_DISTANCE_KPC * 1000.0

    return f"""
    SELECT TOP {ROWS_PER_SECTOR}
        gs.source_id,
        gs.l,
        gs.b,
        gs.phot_g_mean_mag,
        gs.bp_rp,
        gs.ruwe,
        gs.random_index,
        ap.distance_gspphot,
        ap.distance_gspphot_lower,
        ap.distance_gspphot_upper,
        ap.lum_flame,
        ap.lum_flame_lower,
        ap.lum_flame_upper,
        ap.ag_gspphot
    FROM gaiadr3.gaia_source AS gs
    JOIN gaiadr3.astrophysical_parameters AS ap
      ON gs.source_id = ap.source_id
    WHERE gs.l >= {l_min:.12f}
      AND gs.l {l_upper_operator} {l_max:.12f}
      AND gs.b >= {b_min:.12f}
      AND gs.b {b_upper_operator} {b_max:.12f}
      AND gs.ruwe < {MAX_RUWE:.6f}
      AND gs.phot_g_mean_mag IS NOT NULL
      AND gs.bp_rp IS NOT NULL
      AND ap.distance_gspphot IS NOT NULL
      AND ap.distance_gspphot > 0.0
      AND ap.distance_gspphot <= {maximum_distance_pc:.6f}
      AND ap.lum_flame IS NOT NULL
      AND ap.lum_flame > 0.0
    ORDER BY gs.random_index
    """


def normalize_sector(frame, pd):
    numeric_columns = [
        "l",
        "b",
        "phot_g_mean_mag",
        "bp_rp",
        "ruwe",
        "random_index",
        "distance_gspphot",
        "distance_gspphot_lower",
        "distance_gspphot_upper",
        "lum_flame",
        "lum_flame_lower",
        "lum_flame_upper",
        "ag_gspphot",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    required = ["l", "b", "distance_gspphot", "lum_flame"]
    return frame.dropna(subset=required).reset_index(drop=True)


def query_one_sector(
    Gaia,
    pd,
    l_index: int,
    b_index: int,
    l_min: float,
    l_max: float,
    b_min: float,
    b_max: float,
) -> tuple:
    cache_path = sector_cache_path(l_index, b_index)
    if cache_path.exists() and cache_path.stat().st_size > 200:
        cached = pd.read_csv(cache_path)
        cached = normalize_sector(cached, pd)
        if not cached.empty:
            return cached, "CACHE", 0

    query = build_sector_query(
        l_min,
        l_max,
        b_min,
        b_max,
        l_index == LONGITUDE_SECTORS - 1,
        b_index == LATITUDE_SECTORS - 1,
    )

    last_error = None
    for attempt in range(1, QUERY_RETRIES + 1):
        try:
            job = Gaia.launch_job(
                query,
                dump_to_file=False,
                output_format="csv",
                verbose=False,
            )
            frame = normalize_sector(job.get_results().to_pandas(), pd)
            if frame.empty:
                raise RuntimeError("Sector query returned zero valid rows.")
            frame.to_csv(cache_path, index=False)
            return frame, "TAP", attempt
        except Exception as error:
            last_error = error
            if attempt < QUERY_RETRIES:
                time.sleep(8 * attempt)

    raise RuntimeError(
        f"Sector L{l_index} B{b_index} failed: {compact_error(last_error)}"
    )


def collect_balanced_catalog(Gaia, np, pd):
    Gaia.ROW_LIMIT = -1
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    l_edges = longitude_edges(np)
    b_edges = latitude_edges(np)
    frames = []
    audit_rows = []
    sector_number = 0
    total_sectors = LONGITUDE_SECTORS * LATITUDE_SECTORS

    for b_index in range(LATITUDE_SECTORS):
        for l_index in range(LONGITUDE_SECTORS):
            sector_number += 1
            l_min = float(l_edges[l_index])
            l_max = float(l_edges[l_index + 1])
            b_min = float(b_edges[b_index])
            b_max = float(b_edges[b_index + 1])

            frame, mode, attempt = query_one_sector(
                Gaia,
                pd,
                l_index,
                b_index,
                l_min,
                l_max,
                b_min,
                b_max,
            )
            frame["longitude_sector"] = l_index + 1
            frame["latitude_sector"] = b_index + 1
            frame["sector_id"] = f"L{l_index + 1:02d}_B{b_index + 1:02d}"
            frames.append(frame)
            audit_rows.append(
                {
                    "sector_id": frame["sector_id"].iloc[0],
                    "l_min_deg": l_min,
                    "l_max_deg": l_max,
                    "b_min_deg": b_min,
                    "b_max_deg": b_max,
                    "star_count": len(frame),
                    "query_mode": mode,
                    "query_attempt": attempt,
                }
            )
            print(
                f"SECTOR {sector_number:02d}/{total_sectors:02d} | "
                f"{frame['sector_id'].iloc[0]} | {len(frame):03d} stars | {mode}"
            )

    catalog = pd.concat(frames, ignore_index=True)
    catalog = catalog.drop_duplicates(subset=["source_id"]).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows)
    return catalog, audit


def transform_catalog(frame, np):
    distance_kpc = frame["distance_gspphot"].to_numpy(dtype=float) / 1000.0
    l_rad = np.radians(frame["l"].to_numpy(dtype=float))
    b_rad = np.radians(frame["b"].to_numpy(dtype=float))

    cos_b = np.cos(b_rad)
    frame = frame.copy()
    frame["distance_kpc"] = distance_kpc
    frame["x_sun_kpc"] = distance_kpc * cos_b * np.cos(l_rad)
    frame["y_sun_kpc"] = distance_kpc * cos_b * np.sin(l_rad)
    frame["z_sun_kpc"] = distance_kpc * np.sin(b_rad)

    luminosity = frame["lum_flame"].to_numpy(dtype=float)
    edges = np.geomspace(
        LUMINOSITY_MIN,
        LUMINOSITY_MAX,
        LUMINOSITY_LEVELS + 1,
    )
    clipped = np.clip(
        luminosity,
        edges[0],
        np.nextafter(edges[-1], edges[0]),
    )
    frame["luminosity_level"] = (
        np.digitize(clipped, edges[1:-1], right=False) + 1
    ).astype(int)
    frame["luminosity_color"] = frame["luminosity_level"].map(LEVEL_COLORS)

    log_l = np.log10(np.clip(luminosity, LUMINOSITY_MIN, 1.0e5))
    low = float(np.nanpercentile(log_l, 2.0))
    high = float(np.nanpercentile(log_l, 98.0))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        frame["marker_size"] = 8.0
    else:
        scaled = np.clip((log_l - low) / (high - low), 0.0, 1.0)
        frame["marker_size"] = 3.0 + 28.0 * scaled

    return frame, edges


def style_axis(axis, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_facecolor("black")
    axis.set_title(title, color="white", fontsize=12)
    axis.set_xlabel(xlabel, color="white")
    axis.set_ylabel(ylabel, color="white")
    axis.tick_params(colors="#D8D8D8", width=0.5)
    for spine in axis.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.5)
    axis.grid(True, color="#444444", linewidth=0.3, alpha=0.45)
    axis.axhline(0.0, color="#777777", linewidth=0.45, alpha=0.75)
    axis.axvline(0.0, color="#777777", linewidth=0.45, alpha=0.75)


def scatter_levels(axis, frame, x_column: str, y_column: str) -> None:
    for level in range(1, LUMINOSITY_LEVELS + 1):
        subset = frame[frame["luminosity_level"] == level]
        axis.scatter(
            subset[x_column],
            subset[y_column],
            s=subset["marker_size"],
            c=LEVEL_COLORS[level],
            alpha=0.72,
            linewidths=0.0,
            rasterized=True,
        )
    axis.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=130,
        c="#FFFFFF",
        edgecolors="#FFD84A",
        linewidths=0.8,
        zorder=10,
    )


def draw_sphere_shell(axis, radius: float, np) -> None:
    u_values = np.linspace(0.0, 2.0 * np.pi, 28)
    v_values = np.linspace(0.0, np.pi, 15)
    x_values = radius * np.outer(np.cos(u_values), np.sin(v_values))
    y_values = radius * np.outer(np.sin(u_values), np.sin(v_values))
    z_values = radius * np.outer(np.ones_like(u_values), np.cos(v_values))
    axis.plot_wireframe(
        x_values,
        y_values,
        z_values,
        rstride=3,
        cstride=3,
        color="#666666",
        linewidth=0.25,
        alpha=0.18,
    )


def luminosity_labels(edges) -> list:
    labels = []
    for index in range(LUMINOSITY_LEVELS):
        if index == 0:
            labels.append(f"L1: <= {edges[1]:.4g} Lsun")
        elif index == LUMINOSITY_LEVELS - 1:
            labels.append(f"L8: >= {edges[-2]:.4g} Lsun")
        else:
            labels.append(
                f"L{index + 1}: {edges[index]:.4g}-{edges[index + 1]:.4g} Lsun"
            )
    return labels


def plot_balanced_bubble(frame, edges, np, plt, Line2D) -> Path:
    output_path = OUTPUT_DIR / f"{VERSION}_SECTOR_BALANCED_SCALAR_BUBBLE.png"
    figure = plt.figure(figsize=(15, 12), facecolor="black")
    grid = figure.add_gridspec(2, 2, hspace=0.22, wspace=0.20)

    axis_xy = figure.add_subplot(grid[0, 0])
    axis_xz = figure.add_subplot(grid[0, 1])
    axis_yz = figure.add_subplot(grid[1, 0])
    axis_3d = figure.add_subplot(grid[1, 1], projection="3d")

    scatter_levels(axis_xy, frame, "x_sun_kpc", "y_sun_kpc")
    style_axis(
        axis_xy,
        "Equal-area Sun-centered X-Y",
        "X toward Galactic center (kpc)",
        "Y toward Galactic rotation (kpc)",
    )
    axis_xy.set_aspect("equal", adjustable="box")
    axis_xy.scatter(
        [GALACTIC_CENTER_DISTANCE_KPC],
        [0.0],
        marker="+",
        s=90,
        c="#FFFFFF",
        linewidths=1.0,
        zorder=10,
    )

    scatter_levels(axis_xz, frame, "x_sun_kpc", "z_sun_kpc")
    style_axis(
        axis_xz,
        "Equal-area Sun-centered X-Z",
        "X toward Galactic center (kpc)",
        "Z toward Galactic north (kpc)",
    )
    axis_xz.scatter(
        [GALACTIC_CENTER_DISTANCE_KPC],
        [0.0],
        marker="+",
        s=90,
        c="#FFFFFF",
        linewidths=1.0,
        zorder=10,
    )

    scatter_levels(axis_yz, frame, "y_sun_kpc", "z_sun_kpc")
    style_axis(
        axis_yz,
        "Equal-area Sun-centered Y-Z",
        "Y toward Galactic rotation (kpc)",
        "Z toward Galactic north (kpc)",
    )

    axis_3d.set_facecolor("black")
    for level in range(1, LUMINOSITY_LEVELS + 1):
        subset = frame[frame["luminosity_level"] == level]
        axis_3d.scatter(
            subset["x_sun_kpc"],
            subset["y_sun_kpc"],
            subset["z_sun_kpc"],
            s=subset["marker_size"] * 0.7,
            c=LEVEL_COLORS[level],
            alpha=0.68,
            linewidths=0.0,
            rasterized=True,
        )

    observed_radius = float(np.nanpercentile(frame["distance_kpc"], 99.0))
    plot_radius = max(5.0, min(MAX_DISTANCE_KPC, math.ceil(observed_radius / 5.0) * 5.0))
    for shell_radius in np.arange(10.0, plot_radius + 0.1, 10.0):
        draw_sphere_shell(axis_3d, float(shell_radius), np)

    axis_3d.scatter(
        [0.0],
        [0.0],
        [0.0],
        marker="*",
        s=135,
        c="#FFFFFF",
        edgecolors="#FFD84A",
        linewidths=0.8,
        zorder=10,
    )
    axis_3d.scatter(
        [GALACTIC_CENTER_DISTANCE_KPC],
        [0.0],
        [0.0],
        marker="+",
        s=90,
        c="#FFFFFF",
        linewidths=1.0,
        zorder=10,
    )
    axis_3d.set_xlim(-plot_radius, plot_radius)
    axis_3d.set_ylim(-plot_radius, plot_radius)
    axis_3d.set_zlim(-plot_radius, plot_radius)
    axis_3d.set_box_aspect((1.0, 1.0, 1.0))
    axis_3d.set_title("Sun-centered scalar luminosity bubble", color="white")
    axis_3d.set_xlabel("X (kpc)", color="white")
    axis_3d.set_ylabel("Y (kpc)", color="white")
    axis_3d.set_zlabel("Z (kpc)", color="white")
    axis_3d.tick_params(colors="#D8D8D8", width=0.5)
    axis_3d.xaxis.pane.set_facecolor((0.0, 0.0, 0.0, 1.0))
    axis_3d.yaxis.pane.set_facecolor((0.0, 0.0, 0.0, 1.0))
    axis_3d.zaxis.pane.set_facecolor((0.0, 0.0, 0.0, 1.0))
    axis_3d.view_init(elev=24.0, azim=42.0)

    counts = frame.groupby("luminosity_level").size().to_dict()
    handles = []
    for level, label in enumerate(luminosity_labels(edges), start=1):
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=6,
                markerfacecolor=LEVEL_COLORS[level],
                markeredgewidth=0.0,
                label=f"{label} ({counts.get(level, 0)})",
            )
        )
    handles.extend(
        [
            Line2D(
                [0], [0], marker="*", linestyle="", markersize=9,
                markerfacecolor="#FFFFFF", markeredgecolor="#FFD84A",
                label="Sun",
            ),
            Line2D(
                [0], [0], marker="+", linestyle="", markersize=9,
                color="#FFFFFF", label="Galactic center direction",
            ),
        ]
    )
    legend = figure.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=8,
    )
    for text in legend.get_texts():
        text.set_color("white")

    figure.suptitle(
        (
            f"Gaia DR3 equal-area stellar bubble | {len(frame):,} unique stars | "
            f"{LONGITUDE_SECTORS * LATITUDE_SECTORS} sectors | "
            "marker size = Gaia FLAME luminosity"
        ),
        color="white",
        fontsize=14,
        y=0.98,
    )
    figure.subplots_adjust(bottom=0.15, top=0.94)
    figure.savefig(
        output_path,
        dpi=240,
        bbox_inches="tight",
        facecolor="black",
    )
    plt.show()
    plt.close(figure)
    return output_path


def save_outputs(frame, audit, edges, pd):
    catalog_path = OUTPUT_DIR / f"{VERSION}_BALANCED_CATALOG.csv"
    sector_path = OUTPUT_DIR / f"{VERSION}_SECTOR_AUDIT.csv"
    level_path = OUTPUT_DIR / f"{VERSION}_LUMINOSITY_AUDIT.csv"

    frame.to_csv(catalog_path, index=False)
    audit.to_csv(sector_path, index=False)

    level_summary = (
        frame.groupby("luminosity_level", as_index=False)
        .agg(
            star_count=("source_id", "count"),
            minimum_luminosity_lsun=("lum_flame", "min"),
            median_luminosity_lsun=("lum_flame", "median"),
            maximum_luminosity_lsun=("lum_flame", "max"),
            minimum_distance_kpc=("distance_kpc", "min"),
            median_distance_kpc=("distance_kpc", "median"),
            maximum_distance_kpc=("distance_kpc", "max"),
        )
    )
    labels = luminosity_labels(edges)
    level_summary["level_label"] = level_summary["luminosity_level"].map(
        {index + 1: labels[index] for index in range(LUMINOSITY_LEVELS)}
    )
    level_summary["color_hex"] = level_summary["luminosity_level"].map(
        LEVEL_COLORS
    )
    level_summary.to_csv(level_path, index=False)
    return catalog_path, sector_path, level_path, level_summary


def print_summary(
    frame,
    audit,
    catalog_path: Path,
    sector_path: Path,
    level_path: Path,
    plot_path: Path,
    level_summary,
) -> None:
    print()
    print(f"{'METRIC':<34} {'VALUE':>20}")
    print(f"{'-' * 34} {'-' * 20}")
    print(f"{'Coordinate origin':<34} {'SUN':>20}")
    print(f"{'Inner distance cutoff':<34} {'NONE':>20}")
    print(f"{'Maximum GSP-Phot distance':<34} {MAX_DISTANCE_KPC:>17.3f} kpc")
    print(f"{'Equal-area sectors':<34} {len(audit):>20d}")
    print(f"{'Target stars per sector':<34} {ROWS_PER_SECTOR:>20d}")
    print(f"{'Unique stars plotted':<34} {len(frame):>20,}")
    print(f"{'Observed maximum distance':<34} {frame['distance_kpc'].max():>17.3f} kpc")
    print(f"{'Distance source':<34} {'GSP-PHOT':>20}")
    print(f"{'Luminosity source':<34} {'FLAME':>20}")
    print()
    print(level_summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print(f"{'FILE':<34} PATH")
    print(f"{'-' * 34} {'-' * 72}")
    print(f"{'Balanced catalog CSV':<34} {catalog_path}")
    print(f"{'Sector audit CSV':<34} {sector_path}")
    print(f"{'Luminosity audit CSV':<34} {level_path}")
    print(f"{'Scalar bubble PNG':<34} {plot_path}")


def execute() -> None:
    print(f"CODE OUTPUT: {VERSION}")
    try:
        ensure_dependencies()

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from astroquery.gaia import Gaia

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        catalog, sector_audit = collect_balanced_catalog(Gaia, np, pd)
        transformed, edges = transform_catalog(catalog, np)
        catalog_path, sector_path, level_path, level_summary = save_outputs(
            transformed,
            sector_audit,
            edges,
            pd,
        )
        plot_path = plot_balanced_bubble(
            transformed,
            edges,
            np,
            plt,
            Line2D,
        )
        print_summary(
            transformed,
            sector_audit,
            catalog_path,
            sector_path,
            level_path,
            plot_path,
            level_summary,
        )
    except Exception as error:
        print()
        print("STATUS: FAILED")
        print(f"ERROR: {compact_error(error)}")
    finally:
        print()
        print(colombia_timestamp())
        print(f"END OF CODE OUTPUT: {VERSION}")


if __name__ == "__main__":
    execute()
