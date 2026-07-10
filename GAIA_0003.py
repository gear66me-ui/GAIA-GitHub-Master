# GAIA_0003
# Sun-centered Gaia DR3 far-star composite with black background and 8 luminosity levels.

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "GAIA_0003"
OUTPUT_DIR = Path(os.environ.get("GAIA_OUTPUT_DIR", "/content/GAIA_OUTPUT"))
CACHE_FILE = OUTPUT_DIR / "GAIA_0002_GAIA_DR3_SAMPLE.csv"
MAX_ROWS = int(os.environ.get("GAIA_MAX_ROWS", "2000"))
QUERY_RETRIES = int(os.environ.get("GAIA_QUERY_RETRIES", "4"))
MIN_DISTANCE_KPC = float(os.environ.get("GAIA_MIN_DISTANCE_KPC", "5.0"))
MAX_DISTANCE_KPC = float(os.environ.get("GAIA_MAX_DISTANCE_KPC", "50.0"))
MIN_PARALLAX_OVER_ERROR = float(os.environ.get("GAIA_MIN_POE", "5.0"))
MAX_RUWE = float(os.environ.get("GAIA_MAX_RUWE", "1.4"))
SOLAR_ABSOLUTE_G_MAG = 4.67
LUMINOSITY_LEVELS = 8

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
    return " ".join(str(error).split())[:240]


def ensure_dependencies() -> None:
    required = {
        "numpy": "numpy",
        "pandas": "pandas",
        "matplotlib": "matplotlib",
        "astropy": "astropy",
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


def build_query() -> str:
    minimum_parallax_mas = 1.0 / MAX_DISTANCE_KPC
    maximum_parallax_mas = 1.0 / MIN_DISTANCE_KPC
    return f"""
    SELECT TOP {MAX_ROWS}
        source_id,
        ra,
        dec,
        parallax,
        parallax_error,
        parallax_over_error,
        phot_g_mean_mag,
        bp_rp,
        ruwe
    FROM gaiadr3.gaia_source
    WHERE parallax BETWEEN {minimum_parallax_mas:.12f}
                       AND {maximum_parallax_mas:.12f}
      AND parallax_over_error >= {MIN_PARALLAX_OVER_ERROR:.6f}
      AND ruwe < {MAX_RUWE:.6f}
      AND phot_g_mean_mag IS NOT NULL
      AND bp_rp IS NOT NULL
    ORDER BY parallax ASC
    """


def normalize_catalog(frame, pd):
    numeric_columns = [
        "ra",
        "dec",
        "parallax",
        "parallax_error",
        "parallax_over_error",
        "phot_g_mean_mag",
        "bp_rp",
        "ruwe",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=numeric_columns).reset_index(drop=True)
    return frame


def load_or_query_catalog(pd, Gaia):
    if CACHE_FILE.exists() and CACHE_FILE.stat().st_size > 1000:
        cached = pd.read_csv(CACHE_FILE)
        required = {
            "source_id",
            "ra",
            "dec",
            "parallax",
            "parallax_error",
            "parallax_over_error",
            "phot_g_mean_mag",
            "bp_rp",
            "ruwe",
        }
        if required.issubset(cached.columns):
            return normalize_catalog(cached, pd), "GAIA_0002_CACHE", 0

    Gaia.ROW_LIMIT = -1
    last_error = None
    for attempt in range(1, QUERY_RETRIES + 1):
        try:
            job = Gaia.launch_job(
                build_query(),
                dump_to_file=False,
                output_format="csv",
                verbose=False,
            )
            frame = normalize_catalog(job.get_results().to_pandas(), pd)
            return frame, "SYNCHRONOUS_TAP", attempt
        except Exception as error:
            last_error = error
            print(
                f"GAIA TAP RETRY: {attempt}/{QUERY_RETRIES} | "
                f"{compact_error(error)}"
            )
            if attempt < QUERY_RETRIES:
                time.sleep(10 * attempt)

    raise RuntimeError(
        "Gaia synchronous TAP failed after all retries. "
        f"Last error: {compact_error(last_error)}"
    )


def transform_sun_centered(frame, np, pd, u, SkyCoord):
    distance_pc = 1000.0 / frame["parallax"].to_numpy(dtype=float)
    distance_kpc = distance_pc / 1000.0

    coordinates = SkyCoord(
        ra=frame["ra"].to_numpy(dtype=float) * u.deg,
        dec=frame["dec"].to_numpy(dtype=float) * u.deg,
        distance=distance_kpc * u.kpc,
        frame="icrs",
    )
    galactic = coordinates.galactic
    cartesian = galactic.cartesian

    absolute_g = (
        frame["phot_g_mean_mag"].to_numpy(dtype=float)
        - 5.0 * np.log10(distance_pc / 10.0)
    )
    luminosity_proxy = 10.0 ** (
        -0.4 * (absolute_g - SOLAR_ABSOLUTE_G_MAG)
    )

    transformed = frame.copy()
    transformed["distance_pc"] = distance_pc
    transformed["distance_kpc"] = distance_kpc
    transformed["absolute_g_mag"] = absolute_g
    transformed["luminosity_proxy_lsun"] = luminosity_proxy
    transformed["x_sun_kpc"] = cartesian.x.to_value(u.kpc)
    transformed["y_sun_kpc"] = cartesian.y.to_value(u.kpc)
    transformed["z_sun_kpc"] = cartesian.z.to_value(u.kpc)

    luminosity_rank = transformed["luminosity_proxy_lsun"].rank(
        method="first", pct=True
    )
    transformed["luminosity_level"] = (
        np.ceil(luminosity_rank * LUMINOSITY_LEVELS)
        .clip(1, LUMINOSITY_LEVELS)
        .astype(int)
    )
    transformed["luminosity_color"] = transformed[
        "luminosity_level"
    ].map(LEVEL_COLORS)

    return transformed


def style_axis(axis, title: str, xlabel: str, ylabel: str) -> None:
    axis.set_facecolor("black")
    axis.set_title(title, color="white", fontsize=12)
    axis.set_xlabel(xlabel, color="white")
    axis.set_ylabel(ylabel, color="white")
    axis.tick_params(colors="#D8D8D8", width=0.5)
    for spine in axis.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.5)
    axis.grid(True, color="#444444", linewidth=0.35, alpha=0.55)
    axis.axhline(0.0, color="#777777", linewidth=0.45, alpha=0.75)
    axis.axvline(0.0, color="#777777", linewidth=0.45, alpha=0.75)


def scatter_by_level(axis, frame, x_column: str, y_column: str) -> None:
    for level in range(1, LUMINOSITY_LEVELS + 1):
        subset = frame[frame["luminosity_level"] == level]
        axis.scatter(
            subset[x_column],
            subset[y_column],
            s=5.0,
            c=LEVEL_COLORS[level],
            alpha=0.78,
            linewidths=0.0,
            label=f"L{level}",
            rasterized=True,
        )
    axis.scatter(
        [0.0],
        [0.0],
        marker="*",
        s=120,
        c="#FFFFFF",
        edgecolors="#FFD84A",
        linewidths=0.8,
        zorder=10,
        label="Sun",
    )


def plot_composite(frame, plt, Line2D) -> Path:
    output_path = OUTPUT_DIR / f"{VERSION}_SUN_CENTERED_L_LEVELS.png"
    figure = plt.figure(figsize=(15, 12), facecolor="black")
    grid = figure.add_gridspec(2, 2, hspace=0.22, wspace=0.20)

    axis_xy = figure.add_subplot(grid[0, 0])
    axis_xz = figure.add_subplot(grid[0, 1])
    axis_yz = figure.add_subplot(grid[1, 0])
    axis_3d = figure.add_subplot(grid[1, 1], projection="3d")

    scatter_by_level(axis_xy, frame, "x_sun_kpc", "y_sun_kpc")
    style_axis(
        axis_xy,
        "Sun-centered Galactic X-Y bird's-eye view",
        "X toward Galactic center (kpc)",
        "Y toward Galactic rotation (kpc)",
    )
    axis_xy.set_aspect("equal", adjustable="box")

    scatter_by_level(axis_xz, frame, "x_sun_kpc", "z_sun_kpc")
    style_axis(
        axis_xz,
        "Sun-centered Galactic X-Z",
        "X toward Galactic center (kpc)",
        "Z toward Galactic north (kpc)",
    )

    scatter_by_level(axis_yz, frame, "y_sun_kpc", "z_sun_kpc")
    style_axis(
        axis_yz,
        "Sun-centered Galactic Y-Z",
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
            s=3.0,
            c=LEVEL_COLORS[level],
            alpha=0.72,
            linewidths=0.0,
            rasterized=True,
        )
    axis_3d.scatter(
        [0.0],
        [0.0],
        [0.0],
        marker="*",
        s=120,
        c="#FFFFFF",
        edgecolors="#FFD84A",
        linewidths=0.8,
        zorder=10,
    )
    axis_3d.set_title("Sun-centered 3D Galactic view", color="white")
    axis_3d.set_xlabel("X (kpc)", color="white")
    axis_3d.set_ylabel("Y (kpc)", color="white")
    axis_3d.set_zlabel("Z (kpc)", color="white")
    axis_3d.tick_params(colors="#D8D8D8", width=0.5)
    axis_3d.xaxis.pane.set_facecolor((0.0, 0.0, 0.0, 1.0))
    axis_3d.yaxis.pane.set_facecolor((0.0, 0.0, 0.0, 1.0))
    axis_3d.zaxis.pane.set_facecolor((0.0, 0.0, 0.0, 1.0))
    axis_3d.grid(True)
    axis_3d.view_init(elev=22.0, azim=42.0)

    level_summary = (
        frame.groupby("luminosity_level")["luminosity_proxy_lsun"]
        .agg(["min", "max", "count"])
    )
    legend_handles = []
    for level in range(1, LUMINOSITY_LEVELS + 1):
        row = level_summary.loc[level]
        legend_handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=6,
                markerfacecolor=LEVEL_COLORS[level],
                markeredgewidth=0.0,
                label=(
                    f"L{level}: {row['min']:.3g} to "
                    f"{row['max']:.3g} Lsun ({int(row['count'])})"
                ),
            )
        )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="",
            markersize=9,
            markerfacecolor="#FFFFFF",
            markeredgecolor="#FFD84A",
            label="Sun at origin",
        )
    )

    legend = figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
        fontsize=8,
    )
    for text in legend.get_texts():
        text.set_color("white")

    figure.suptitle(
        (
            f"Gaia DR3 Sun-centered far-star map | {len(frame):,} stars | "
            "L1 faintest to L8 brightest luminosity octiles"
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


def save_outputs(frame, pd):
    catalog_path = OUTPUT_DIR / f"{VERSION}_SUN_CENTERED_CATALOG.csv"
    summary_path = OUTPUT_DIR / f"{VERSION}_L_LEVEL_SUMMARY.csv"
    frame.to_csv(catalog_path, index=False)

    summary = (
        frame.groupby("luminosity_level", as_index=False)
        .agg(
            star_count=("source_id", "count"),
            minimum_luminosity_lsun=("luminosity_proxy_lsun", "min"),
            median_luminosity_lsun=("luminosity_proxy_lsun", "median"),
            maximum_luminosity_lsun=("luminosity_proxy_lsun", "max"),
            minimum_distance_kpc=("distance_kpc", "min"),
            median_distance_kpc=("distance_kpc", "median"),
            maximum_distance_kpc=("distance_kpc", "max"),
        )
    )
    summary["color_hex"] = summary["luminosity_level"].map(LEVEL_COLORS)
    summary.to_csv(summary_path, index=False)
    return catalog_path, summary_path, summary


def print_summary(
    frame,
    query_mode: str,
    query_attempt: int,
    catalog_path: Path,
    summary_path: Path,
    plot_path: Path,
    summary,
) -> None:
    print()
    print(f"{'METRIC':<32} {'VALUE':>18}")
    print(f"{'-' * 32} {'-' * 18}")
    print(f"{'Coordinate origin':<32} {'SUN':>18}")
    print(f"{'Background':<32} {'BLACK':>18}")
    print(f"{'Query mode':<32} {query_mode:>18}")
    print(f"{'Successful attempt':<32} {query_attempt:>18d}")
    print(f"{'Stars plotted':<32} {len(frame):>18,}")
    print(f"{'Luminosity levels':<32} {LUMINOSITY_LEVELS:>18d}")
    print()
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print()
    print(f"{'FILE':<32} PATH")
    print(f"{'-' * 32} {'-' * 72}")
    print(f"{'Sun-centered catalog CSV':<32} {catalog_path}")
    print(f"{'L-level summary CSV':<32} {summary_path}")
    print(f"{'Composite PNG':<32} {plot_path}")


def execute() -> None:
    print(f"CODE OUTPUT: {VERSION}")
    try:
        ensure_dependencies()

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astroquery.gaia import Gaia

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        catalog, query_mode, query_attempt = load_or_query_catalog(pd, Gaia)
        if catalog.empty:
            raise RuntimeError("Gaia query returned zero valid rows.")

        transformed = transform_sun_centered(
            catalog, np, pd, u, SkyCoord
        )
        catalog_path, summary_path, summary = save_outputs(
            transformed, pd
        )
        plot_path = plot_composite(transformed, plt, Line2D)
        print_summary(
            transformed,
            query_mode,
            query_attempt,
            catalog_path,
            summary_path,
            plot_path,
            summary,
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
