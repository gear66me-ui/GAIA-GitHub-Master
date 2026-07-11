# GAIA_0007
# Preserve the 32-sector catalogue and add a global Gaia random-index sample.

from __future__ import annotations

import importlib
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "GAIA_0007"
OUTPUT_DIR = Path(os.environ.get("GAIA_OUTPUT_DIR", "/content/GAIA_OUTPUT"))
BALANCED_CANDIDATES = [
    OUTPUT_DIR / "GAIA_0006_BALANCED_CATALOG.csv",
    OUTPUT_DIR / "GAIA_0005_BALANCED_CATALOG.csv",
    OUTPUT_DIR / "GAIA_0004_BALANCED_CATALOG.csv",
]
GLOBAL_CACHE = OUTPUT_DIR / "GAIA_0007_GLOBAL_RANDOM_CACHE.csv"
WINDOW_AUDIT = OUTPUT_DIR / "GAIA_0007_GLOBAL_RANDOM_WINDOWS.json"
GLOBAL_RANDOM_TARGET = int(os.environ.get("GAIA_GLOBAL_RANDOM_TARGET", "32000"))
CHUNK_ROWS = int(os.environ.get("GAIA_GLOBAL_CHUNK_ROWS", "500"))
MAX_DISTANCE_KPC = float(os.environ.get("GAIA_MAX_DISTANCE_KPC", "60.0"))
MAX_RUWE = float(os.environ.get("GAIA_MAX_RUWE", "1.4"))
RANDOM_WINDOWS = int(os.environ.get("GAIA_RANDOM_WINDOWS", "4096"))
RANDOM_INDEX_MAX = int(os.environ.get("GAIA_RANDOM_INDEX_MAX", "2000000000"))
QUERY_RETRIES = int(os.environ.get("GAIA_QUERY_RETRIES", "3"))
RANDOM_SEED = 6607007
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

GLOBAL_SIZES = {1: 0.7, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.2, 6: 1.5, 7: 2.0, 8: 2.8}
GLOBAL_ALPHA = {1: 0.78, 2: 0.74, 3: 0.70, 4: 0.66, 5: 0.58, 6: 0.48, 7: 0.36, 8: 0.24}


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


def find_balanced_catalog() -> Path:
    for path in BALANCED_CANDIDATES:
        if path.exists() and path.stat().st_size > 1000:
            return path
    raise FileNotFoundError(
        "No completed GAIA_0004/0005/0006 balanced catalogue was found."
    )


def normalize(frame, pd):
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
    required = ["source_id", "l", "b", "distance_gspphot", "lum_flame"]
    return frame.dropna(subset=required).drop_duplicates("source_id").reset_index(drop=True)


def load_window_audit() -> set[int]:
    if not WINDOW_AUDIT.exists():
        return set()
    try:
        payload = json.loads(WINDOW_AUDIT.read_text(encoding="utf-8"))
        return {int(value) for value in payload.get("completed_windows", [])}
    except Exception:
        return set()


def save_window_audit(completed: set[int]) -> None:
    WINDOW_AUDIT.write_text(
        json.dumps({"completed_windows": sorted(completed)}, indent=2),
        encoding="utf-8",
    )


def build_global_query(window_index: int) -> str:
    width = RANDOM_INDEX_MAX / RANDOM_WINDOWS
    random_min = int(window_index * width)
    random_max = int((window_index + 1) * width)
    maximum_distance_pc = MAX_DISTANCE_KPC * 1000.0
    return f"""
    SELECT TOP {CHUNK_ROWS}
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
    WHERE gs.random_index >= {random_min}
      AND gs.random_index < {random_max}
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


def collect_global_random(Gaia, pd, balanced_ids: set) -> tuple:
    Gaia.ROW_LIMIT = -1
    completed = load_window_audit()
    if GLOBAL_CACHE.exists() and GLOBAL_CACHE.stat().st_size > 500:
        global_frame = normalize(pd.read_csv(GLOBAL_CACHE), pd)
    else:
        global_frame = pd.DataFrame()

    rng = random.Random(RANDOM_SEED)
    window_order = list(range(RANDOM_WINDOWS))
    rng.shuffle(window_order)
    successful_queries = 0

    for window_index in window_order:
        if len(global_frame) >= GLOBAL_RANDOM_TARGET:
            break
        if window_index in completed:
            continue

        chunk = None
        last_error = None
        for attempt in range(1, QUERY_RETRIES + 1):
            try:
                job = Gaia.launch_job(
                    build_global_query(window_index),
                    dump_to_file=False,
                    output_format="csv",
                    verbose=False,
                )
                chunk = normalize(job.get_results().to_pandas(), pd)
                break
            except Exception as error:
                last_error = error
                print(
                    f"GLOBAL WINDOW {window_index:04d} | "
                    f"retry {attempt}/{QUERY_RETRIES} | {compact_error(error)}"
                )
                if attempt < QUERY_RETRIES:
                    time.sleep(5 * attempt)

        if chunk is None:
            print(
                f"GLOBAL WINDOW {window_index:04d} | SKIPPED | "
                f"{compact_error(last_error)}"
            )
            continue

        completed.add(window_index)
        save_window_audit(completed)
        if not chunk.empty:
            chunk = chunk[~chunk["source_id"].isin(balanced_ids)]
            global_frame = pd.concat([global_frame, chunk], ignore_index=True)
            global_frame = normalize(global_frame, pd)
            global_frame = global_frame.iloc[:GLOBAL_RANDOM_TARGET].copy()
            global_frame.to_csv(GLOBAL_CACHE, index=False)
            successful_queries += 1

        print(
            f"GLOBAL WINDOW {window_index:04d} | "
            f"+{0 if chunk is None else len(chunk):03d} | "
            f"total {len(global_frame):05d}/{GLOBAL_RANDOM_TARGET:05d}"
        )

    if global_frame.empty:
        raise RuntimeError("The global random Gaia sample returned zero rows.")
    return global_frame.iloc[:GLOBAL_RANDOM_TARGET].copy(), successful_queries


def transform(frame, np):
    frame = frame.copy()
    distance_kpc = frame["distance_gspphot"].to_numpy(float) / 1000.0
    l_rad = np.radians(frame["l"].to_numpy(float))
    b_rad = np.radians(frame["b"].to_numpy(float))
    cos_b = np.cos(b_rad)
    frame["distance_kpc"] = distance_kpc
    frame["x_sun_kpc"] = distance_kpc * cos_b * np.cos(l_rad)
    frame["y_sun_kpc"] = distance_kpc * cos_b * np.sin(l_rad)
    frame["z_sun_kpc"] = distance_kpc * np.sin(b_rad)

    edges = np.geomspace(
        LUMINOSITY_MIN,
        LUMINOSITY_MAX,
        LUMINOSITY_LEVELS + 1,
    )
    luminosity = frame["lum_flame"].to_numpy(float)
    clipped = np.clip(
        luminosity,
        edges[0],
        np.nextafter(edges[-1], edges[0]),
    )
    frame["luminosity_level"] = (
        np.digitize(clipped, edges[1:-1], right=False) + 1
    ).astype(int)
    frame["luminosity_color"] = frame["luminosity_level"].map(LEVEL_COLORS)
    return frame, edges


def style_axis(axis, title: str, xlabel: str, ylabel: str, radius: float) -> None:
    axis.set_facecolor("black")
    axis.set_title(title, color="white", fontsize=12)
    axis.set_xlabel(xlabel, color="white")
    axis.set_ylabel(ylabel, color="white")
    axis.set_xlim(-radius, radius)
    axis.set_ylim(-radius, radius)
    axis.tick_params(colors="#D8D8D8", width=0.5)
    axis.grid(True, color="#444444", linewidth=0.3, alpha=0.42)
    axis.axhline(0.0, color="#777777", linewidth=0.4, alpha=0.7)
    axis.axvline(0.0, color="#777777", linewidth=0.4, alpha=0.7)
    for spine in axis.spines.values():
        spine.set_color("#777777")
        spine.set_linewidth(0.5)


def scatter_balanced(axis, frame, x_column: str, y_column: str) -> None:
    for level in range(8, 0, -1):
        subset = frame[frame["luminosity_level"] == level]
        axis.scatter(
            subset[x_column],
            subset[y_column],
            s=0.45,
            c=LEVEL_COLORS[level],
            alpha=0.055,
            linewidths=0.0,
            rasterized=True,
        )


def scatter_global(axis, frame, x_column: str, y_column: str) -> None:
    for level in range(8, 0, -1):
        subset = frame[frame["luminosity_level"] == level]
        axis.scatter(
            subset[x_column],
            subset[y_column],
            s=GLOBAL_SIZES[level],
            c=LEVEL_COLORS[level],
            alpha=GLOBAL_ALPHA[level],
            linewidths=0.0,
            rasterized=True,
        )


def plot_map(balanced, global_frame, edges, np, plt, Line2D) -> tuple[Path, float, float]:
    output_path = OUTPUT_DIR / f"{VERSION}_HYBRID_RANDOM_DENSITY_MAP.png"
    global_p995 = float(np.nanpercentile(global_frame["distance_kpc"], 99.5))
    all_p997 = float(
        np.nanpercentile(
            np.concatenate(
                [balanced["distance_kpc"].to_numpy(), global_frame["distance_kpc"].to_numpy()]
            ),
            99.7,
        )
    )
    main_radius = min(35.0, max(15.0, math.ceil(global_p995 / 5.0) * 5.0))
    full_radius = min(60.0, max(20.0, math.ceil(all_p997 / 10.0) * 10.0))

    figure = plt.figure(figsize=(15, 12), facecolor="black")
    grid = figure.add_gridspec(2, 2, hspace=0.22, wspace=0.20)
    axis_xy = figure.add_subplot(grid[0, 0])
    axis_xz = figure.add_subplot(grid[0, 1])
    axis_yz = figure.add_subplot(grid[1, 0])
    axis_3d = figure.add_subplot(grid[1, 1], projection="3d")

    panels = [
        (axis_xy, "x_sun_kpc", "y_sun_kpc", "Sun-centered X-Y density", "X to Galactic center (kpc)", "Y Galactic rotation (kpc)"),
        (axis_xz, "x_sun_kpc", "z_sun_kpc", "Sun-centered X-Z density", "X to Galactic center (kpc)", "Z Galactic north (kpc)"),
        (axis_yz, "y_sun_kpc", "z_sun_kpc", "Sun-centered Y-Z density", "Y Galactic rotation (kpc)", "Z Galactic north (kpc)"),
    ]
    for axis, x_column, y_column, title, xlabel, ylabel in panels:
        scatter_balanced(axis, balanced, x_column, y_column)
        scatter_global(axis, global_frame, x_column, y_column)
        style_axis(axis, title, xlabel, ylabel, main_radius)
        axis.scatter([0.0], [0.0], marker="*", s=80, c="#FFFFFF", edgecolors="#FFD84A", linewidths=0.7, zorder=20)
    axis_xy.set_aspect("equal", adjustable="box")
    axis_xy.scatter([GALACTIC_CENTER_DISTANCE_KPC], [0.0], marker="+", s=70, c="#FFFFFF", linewidths=0.9, zorder=20)
    axis_xz.scatter([GALACTIC_CENTER_DISTANCE_KPC], [0.0], marker="+", s=70, c="#FFFFFF", linewidths=0.9, zorder=20)

    axis_3d.set_facecolor("black")
    for level in range(8, 0, -1):
        base_subset = balanced[balanced["luminosity_level"] == level]
        axis_3d.scatter(
            base_subset["x_sun_kpc"], base_subset["y_sun_kpc"], base_subset["z_sun_kpc"],
            s=0.28, c=LEVEL_COLORS[level], alpha=0.035, linewidths=0.0, rasterized=True,
        )
        subset = global_frame[global_frame["luminosity_level"] == level]
        axis_3d.scatter(
            subset["x_sun_kpc"], subset["y_sun_kpc"], subset["z_sun_kpc"],
            s=GLOBAL_SIZES[level] * 0.65, c=LEVEL_COLORS[level],
            alpha=GLOBAL_ALPHA[level] * 0.9, linewidths=0.0, rasterized=True,
        )

    axis_3d.scatter([0.0], [0.0], [0.0], marker="*", s=90, c="#FFFFFF", edgecolors="#FFD84A", linewidths=0.7, zorder=20)
    axis_3d.scatter([GALACTIC_CENTER_DISTANCE_KPC], [0.0], [0.0], marker="+", s=70, c="#FFFFFF", linewidths=0.9, zorder=20)
    axis_3d.set_xlim(-full_radius, full_radius)
    axis_3d.set_ylim(-full_radius, full_radius)
    axis_3d.set_zlim(-full_radius, full_radius)
    axis_3d.set_box_aspect((1.0, 1.0, 1.0))
    axis_3d.set_title("Full hybrid 3D stellar distribution", color="white")
    axis_3d.set_xlabel("X (kpc)", color="white")
    axis_3d.set_ylabel("Y (kpc)", color="white")
    axis_3d.set_zlabel("Z (kpc)", color="white")
    axis_3d.tick_params(colors="#D8D8D8", width=0.5)
    axis_3d.xaxis.pane.set_facecolor((0.0, 0.0, 0.0, 1.0))
    axis_3d.yaxis.pane.set_facecolor((0.0, 0.0, 0.0, 1.0))
    axis_3d.zaxis.pane.set_facecolor((0.0, 0.0, 0.0, 1.0))
    axis_3d.view_init(elev=24.0, azim=42.0)

    handles = []
    for level in range(1, 9):
        handles.append(
            Line2D([0], [0], marker="o", linestyle="", markersize=5,
                   markerfacecolor=LEVEL_COLORS[level], markeredgewidth=0.0,
                   label=f"L{level}")
        )
    handles.extend(
        [
            Line2D([0], [0], marker="o", linestyle="", markersize=4,
                   markerfacecolor="#FFFFFF", alpha=0.20, markeredgewidth=0.0,
                   label="32-sector coverage layer"),
            Line2D([0], [0], marker="o", linestyle="", markersize=4,
                   markerfacecolor="#FFFFFF", markeredgewidth=0.0,
                   label="Global random density layer"),
        ]
    )
    legend = figure.legend(handles=handles, loc="lower center", ncol=5,
                           frameon=False, bbox_to_anchor=(0.5, 0.012), fontsize=8)
    for text in legend.get_texts():
        text.set_color("white")

    figure.suptitle(
        (f"Gaia DR3 hybrid density map | {len(balanced):,} preserved sector stars + "
         f"{len(global_frame):,} global random stars | L8 drawn first"),
        color="white", fontsize=14, y=0.98,
    )
    figure.subplots_adjust(bottom=0.12, top=0.94)
    figure.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="black")
    plt.show()
    plt.close(figure)
    return output_path, main_radius, full_radius


def save_outputs(balanced, global_frame, pd):
    balanced = balanced.copy()
    global_frame = global_frame.copy()
    balanced["sample_origin"] = "SECTOR_BALANCED"
    global_frame["sample_origin"] = "GLOBAL_RANDOM_INDEX"
    combined = pd.concat([balanced, global_frame], ignore_index=True)
    combined = combined.drop_duplicates("source_id").reset_index(drop=True)
    combined_path = OUTPUT_DIR / f"{VERSION}_HYBRID_CATALOG.csv"
    combined.to_csv(combined_path, index=False)
    return combined_path, combined


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
        balanced_path = find_balanced_catalog()
        balanced = normalize(pd.read_csv(balanced_path), pd)
        balanced_ids = set(balanced["source_id"].astype(str))
        balanced["source_id"] = balanced["source_id"].astype(str)

        global_frame, successful_queries = collect_global_random(
            Gaia, pd, balanced_ids
        )
        global_frame["source_id"] = global_frame["source_id"].astype(str)

        balanced, edges = transform(balanced, np)
        global_frame, _ = transform(global_frame, np)
        combined_path, combined = save_outputs(balanced, global_frame, pd)
        plot_path, main_radius, full_radius = plot_map(
            balanced, global_frame, edges, np, plt, Line2D
        )

        print()
        print(f"{'METRIC':<34} {'VALUE':>20}")
        print(f"{'-' * 34} {'-' * 20}")
        print(f"{'Balanced catalogue':<34} {len(balanced):>20,}")
        print(f"{'Global random catalogue':<34} {len(global_frame):>20,}")
        print(f"{'Combined unique stars':<34} {len(combined):>20,}")
        print(f"{'Successful new TAP windows':<34} {successful_queries:>20d}")
        print(f"{'2D display radius':<34} {main_radius:>17.3f} kpc")
        print(f"{'3D display radius':<34} {full_radius:>17.3f} kpc")
        print(f"{'Plot order':<34} {'L8 TO L1':>20}")
        print(f"{'Visible density driver':<34} {'GLOBAL RANDOM':>20}")
        print()
        print(f"{'FILE':<34} PATH")
        print(f"{'-' * 34} {'-' * 72}")
        print(f"{'Balanced input CSV':<34} {balanced_path}")
        print(f"{'Global random cache CSV':<34} {GLOBAL_CACHE}")
        print(f"{'Hybrid catalogue CSV':<34} {combined_path}")
        print(f"{'Hybrid density PNG':<34} {plot_path}")
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
