# GAIA_0001
# Gaia DR3 far-star composite mapping: XY, XZ, YZ, and 3D views with 8 luminosity levels.

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


VERSION = "GAIA_0001"
OUTPUT_DIR = Path(os.environ.get("GAIA_OUTPUT_DIR", "/content/GAIA_OUTPUT"))
MAX_ROWS = int(os.environ.get("GAIA_MAX_ROWS", "50000"))

MIN_DISTANCE_KPC = float(os.environ.get("GAIA_MIN_DISTANCE_KPC", "5.0"))
MAX_DISTANCE_KPC = float(os.environ.get("GAIA_MAX_DISTANCE_KPC", "50.0"))
MIN_PARALLAX_OVER_ERROR = float(os.environ.get("GAIA_MIN_POE", "5.0"))
MAX_RUWE = float(os.environ.get("GAIA_MAX_RUWE", "1.4"))

LUMINOSITY_MIN = float(os.environ.get("GAIA_L_MIN", "0.001"))
LUMINOSITY_MAX = float(os.environ.get("GAIA_L_MAX", "8.0"))
LUMINOSITY_LEVELS = 8

SOLAR_ABSOLUTE_G_MAG = 4.67
GALACTOCENTRIC_DISTANCE_KPC = 8.122
SUN_HEIGHT_PC = 20.8


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
            __import__(module_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", *missing],
            check=True,
        )


ensure_dependencies()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from astropy import units as u
from astropy.coordinates import SkyCoord, Galactocentric
from astroquery.gaia import Gaia


def build_query() -> str:
    min_parallax_mas = 1.0 / MAX_DISTANCE_KPC
    max_parallax_mas = 1.0 / MIN_DISTANCE_KPC

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
    WHERE parallax BETWEEN {min_parallax_mas:.12f} AND {max_parallax_mas:.12f}
      AND parallax_over_error >= {MIN_PARALLAX_OVER_ERROR:.6f}
      AND ruwe < {MAX_RUWE:.6f}
      AND phot_g_mean_mag IS NOT NULL
      AND bp_rp IS NOT NULL
    ORDER BY parallax ASC
    """


def query_gaia() -> pd.DataFrame:
    Gaia.ROW_LIMIT = -1
    job = Gaia.launch_job_async(
        build_query(),
        dump_to_file=False,
        verbose=False,
    )
    table = job.get_results()
    frame = table.to_pandas()

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


def transform_catalog(frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[float, float, float]]:
    distance_pc = 1000.0 / frame["parallax"].to_numpy(dtype=float)
    distance_kpc = distance_pc / 1000.0

    coordinates = SkyCoord(
        ra=frame["ra"].to_numpy(dtype=float) * u.deg,
        dec=frame["dec"].to_numpy(dtype=float) * u.deg,
        distance=distance_kpc * u.kpc,
        frame="icrs",
    )

    galactocentric_frame = Galactocentric(
        galcen_distance=GALACTOCENTRIC_DISTANCE_KPC * u.kpc,
        z_sun=SUN_HEIGHT_PC * u.pc,
        roll=0.0 * u.deg,
    )
    galactocentric = coordinates.transform_to(galactocentric_frame)

    absolute_g = (
        frame["phot_g_mean_mag"].to_numpy(dtype=float)
        - 5.0 * np.log10(distance_pc / 10.0)
    )
    luminosity_proxy = 10.0 ** (
        -0.4 * (absolute_g - SOLAR_ABSOLUTE_G_MAG)
    )

    level_edges = np.geomspace(
        LUMINOSITY_MIN,
        LUMINOSITY_MAX,
        LUMINOSITY_LEVELS + 1,
    )
    clipped_luminosity = np.clip(
        luminosity_proxy,
        level_edges[0],
        np.nextafter(level_edges[-1], level_edges[0]),
    )
    level_index = np.digitize(
        clipped_luminosity,
        level_edges[1:-1],
        right=False,
    ) + 1

    transformed = frame.copy()
    transformed["distance_pc"] = distance_pc
    transformed["distance_kpc"] = distance_kpc
    transformed["absolute_g_mag"] = absolute_g
    transformed["luminosity_proxy_lsun"] = luminosity_proxy
    transformed["luminosity_level"] = level_index.astype(int)
    transformed["x_gc_kpc"] = galactocentric.x.to_value(u.kpc)
    transformed["y_gc_kpc"] = galactocentric.y.to_value(u.kpc)
    transformed["z_gc_kpc"] = galactocentric.z.to_value(u.kpc)

    sun_origin = SkyCoord(
        ra=0.0 * u.deg,
        dec=0.0 * u.deg,
        distance=0.0 * u.pc,
        frame="icrs",
    ).transform_to(galactocentric_frame)
    sun_xyz = (
        float(sun_origin.x.to_value(u.kpc)),
        float(sun_origin.y.to_value(u.kpc)),
        float(sun_origin.z.to_value(u.kpc)),
    )

    return transformed, sun_xyz


def level_labels() -> tuple[np.ndarray, list[str]]:
    edges = np.geomspace(
        LUMINOSITY_MIN,
        LUMINOSITY_MAX,
        LUMINOSITY_LEVELS + 1,
    )
    labels = []
    for index in range(LUMINOSITY_LEVELS):
        lower = edges[index]
        upper = edges[index + 1]
        if index == 0:
            labels.append(f"L1: ≤ {upper:.4g} L☉")
        elif index == LUMINOSITY_LEVELS - 1:
            labels.append(f"L8: ≥ {lower:.4g} L☉")
        else:
            labels.append(f"L{index + 1}: {lower:.4g}–{upper:.4g} L☉")
    return edges, labels


def add_reference_markers(ax, sun_xyz: tuple[float, float, float], projection: str) -> None:
    sx, sy, sz = sun_xyz

    if projection == "xy":
        ax.scatter([0.0], [0.0], marker="+", s=80, linewidths=1.0, color="black", zorder=5)
        ax.scatter([sx], [sy], marker="*", s=90, linewidths=0.5, color="black", zorder=5)
        ax.add_patch(Circle((0.0, 0.0), 15.0, fill=False, linewidth=0.5, linestyle="--", alpha=0.45))
        ax.add_patch(Circle((0.0, 0.0), 25.0, fill=False, linewidth=0.5, linestyle=":", alpha=0.35))
    elif projection == "xz":
        ax.scatter([0.0], [0.0], marker="+", s=80, linewidths=1.0, color="black", zorder=5)
        ax.scatter([sx], [sz], marker="*", s=90, linewidths=0.5, color="black", zorder=5)
    elif projection == "yz":
        ax.scatter([0.0], [0.0], marker="+", s=80, linewidths=1.0, color="black", zorder=5)
        ax.scatter([sy], [sz], marker="*", s=90, linewidths=0.5, color="black", zorder=5)


def plot_composite(frame: pd.DataFrame, sun_xyz: tuple[float, float, float]) -> Path:
    output_path = OUTPUT_DIR / f"{VERSION}_COMPOSITE.png"

    cmap = plt.get_cmap("plasma", LUMINOSITY_LEVELS)
    colors = cmap(frame["luminosity_level"].to_numpy(dtype=int) - 1)
    marker_size = 2.0

    figure = plt.figure(figsize=(15, 12))
    grid = figure.add_gridspec(2, 2, hspace=0.22, wspace=0.18)

    ax_xy = figure.add_subplot(grid[0, 0])
    ax_xz = figure.add_subplot(grid[0, 1])
    ax_yz = figure.add_subplot(grid[1, 0])
    ax_3d = figure.add_subplot(grid[1, 1], projection="3d")

    ax_xy.scatter(
        frame["x_gc_kpc"],
        frame["y_gc_kpc"],
        c=colors,
        s=marker_size,
        alpha=0.58,
        linewidths=0.0,
        rasterized=True,
    )
    ax_xy.set_title("Milky Way bird's-eye view — Galactic X–Y")
    ax_xy.set_xlabel("X from Galactic center (kpc)")
    ax_xy.set_ylabel("Y from Galactic center (kpc)")
    ax_xy.set_aspect("equal", adjustable="box")
    add_reference_markers(ax_xy, sun_xyz, "xy")
    ax_xy.grid(True, linewidth=0.3, alpha=0.35)

    ax_xz.scatter(
        frame["x_gc_kpc"],
        frame["z_gc_kpc"],
        c=colors,
        s=marker_size,
        alpha=0.58,
        linewidths=0.0,
        rasterized=True,
    )
    ax_xz.set_title("Galactic X–Z")
    ax_xz.set_xlabel("X from Galactic center (kpc)")
    ax_xz.set_ylabel("Z from Galactic plane (kpc)")
    add_reference_markers(ax_xz, sun_xyz, "xz")
    ax_xz.grid(True, linewidth=0.3, alpha=0.35)

    ax_yz.scatter(
        frame["y_gc_kpc"],
        frame["z_gc_kpc"],
        c=colors,
        s=marker_size,
        alpha=0.58,
        linewidths=0.0,
        rasterized=True,
    )
    ax_yz.set_title("Galactic Y–Z")
    ax_yz.set_xlabel("Y from Galactic center (kpc)")
    ax_yz.set_ylabel("Z from Galactic plane (kpc)")
    add_reference_markers(ax_yz, sun_xyz, "yz")
    ax_yz.grid(True, linewidth=0.3, alpha=0.35)

    ax_3d.scatter(
        frame["x_gc_kpc"],
        frame["y_gc_kpc"],
        frame["z_gc_kpc"],
        c=colors,
        s=1.0,
        alpha=0.42,
        linewidths=0.0,
        rasterized=True,
    )
    ax_3d.scatter([0.0], [0.0], [0.0], marker="+", s=70, color="black", linewidths=1.0)
    ax_3d.scatter(
        [sun_xyz[0]],
        [sun_xyz[1]],
        [sun_xyz[2]],
        marker="*",
        s=75,
        color="black",
        linewidths=0.5,
    )
    ax_3d.set_title("Galactocentric 3D view")
    ax_3d.set_xlabel("X (kpc)")
    ax_3d.set_ylabel("Y (kpc)")
    ax_3d.set_zlabel("Z (kpc)")
    ax_3d.view_init(elev=22.0, azim=42.0)

    _, labels = level_labels()
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=5,
            markerfacecolor=cmap(index),
            markeredgewidth=0.0,
            label=label,
        )
        for index, label in enumerate(labels)
    ]
    handles.extend(
        [
            Line2D([0], [0], marker="+", color="black", linestyle="", markersize=8, label="Galactic center"),
            Line2D([0], [0], marker="*", color="black", linestyle="", markersize=8, label="Sun"),
        ]
    )

    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=8,
    )
    figure.suptitle(
        (
            f"Gaia DR3 far-star sample | {len(frame):,} stars | "
            f"{MIN_DISTANCE_KPC:.1f}–{MAX_DISTANCE_KPC:.1f} kpc | "
            f"parallax_over_error ≥ {MIN_PARALLAX_OVER_ERROR:.1f}"
        ),
        fontsize=14,
        y=0.98,
    )
    figure.subplots_adjust(bottom=0.12, top=0.94)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_outputs(frame: pd.DataFrame) -> tuple[Path, Path]:
    raw_path = OUTPUT_DIR / f"{VERSION}_GAIA_DR3_SAMPLE.csv"
    summary_path = OUTPUT_DIR / f"{VERSION}_LUMINOSITY_LEVEL_SUMMARY.csv"

    frame.to_csv(raw_path, index=False)

    summary = (
        frame.groupby("luminosity_level", as_index=False)
        .agg(
            star_count=("source_id", "count"),
            minimum_distance_kpc=("distance_kpc", "min"),
            median_distance_kpc=("distance_kpc", "median"),
            maximum_distance_kpc=("distance_kpc", "max"),
            minimum_luminosity_lsun=("luminosity_proxy_lsun", "min"),
            median_luminosity_lsun=("luminosity_proxy_lsun", "median"),
            maximum_luminosity_lsun=("luminosity_proxy_lsun", "max"),
        )
    )

    _, labels = level_labels()
    label_map = {index + 1: labels[index] for index in range(LUMINOSITY_LEVELS)}
    summary["level_label"] = summary["luminosity_level"].map(label_map)
    summary.to_csv(summary_path, index=False)

    return raw_path, summary_path


def print_summary(
    frame: pd.DataFrame,
    raw_path: Path,
    summary_path: Path,
    plot_path: Path,
) -> None:
    print(f"CODE OUTPUT: {VERSION}")
    print()
    print(f"{'METRIC':<30} {'VALUE':>18}")
    print(f"{'-' * 30} {'-' * 18}")
    print(f"{'Stars plotted':<30} {len(frame):>18,}")
    print(f"{'Minimum distance (kpc)':<30} {frame['distance_kpc'].min():>18.6f}")
    print(f"{'Median distance (kpc)':<30} {frame['distance_kpc'].median():>18.6f}")
    print(f"{'Maximum distance (kpc)':<30} {frame['distance_kpc'].max():>18.6f}")
    print(f"{'Luminosity levels':<30} {LUMINOSITY_LEVELS:>18d}")
    print()
    print(f"{'FILE':<30} PATH")
    print(f"{'-' * 30} {'-' * 72}")
    print(f"{'Catalog CSV':<30} {raw_path}")
    print(f"{'Level summary CSV':<30} {summary_path}")
    print(f"{'Composite PNG':<30} {plot_path}")
    print()
    timestamp = datetime.now(ZoneInfo("America/Bogota")).strftime("%Y-%m-%d %H:%M:%S %Z")
    print(timestamp)
    print(f"# {VERSION}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    catalog = query_gaia()
    if catalog.empty:
        raise RuntimeError("Gaia query returned zero rows for the configured filters.")

    transformed, sun_xyz = transform_catalog(catalog)
    raw_path, summary_path = save_outputs(transformed)
    plot_path = plot_composite(transformed, sun_xyz)
    print_summary(transformed, raw_path, summary_path, plot_path)


if __name__ == "__main__":
    main()
