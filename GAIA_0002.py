# GAIA_0002
# Gaia DR3 far-star composite mapping with TAP retry and synchronous fallback.

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

VERSION = "GAIA_0002"
OUTPUT_DIR = Path(os.environ.get("GAIA_OUTPUT_DIR", "/content/GAIA_OUTPUT"))

MAX_ROWS = int(os.environ.get("GAIA_MAX_ROWS", "50000"))
SYNC_FALLBACK_ROWS = int(os.environ.get("GAIA_SYNC_FALLBACK_ROWS", "2000"))
ASYNC_RETRIES = int(os.environ.get("GAIA_ASYNC_RETRIES", "4"))
RETRY_DELAY_SECONDS = int(os.environ.get("GAIA_RETRY_DELAY_SECONDS", "15"))

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

np = None
pd = None
plt = None
Line2D = None
Circle = None
u = None
SkyCoord = None
Galactocentric = None
Gaia = None


def colombia_timestamp() -> str:
    return datetime.now(ZoneInfo("America/Bogota")).strftime(
        "%Y-%m-%d %H:%M:%S America/Bogota"
    )


def compact_error(error: Exception) -> str:
    text = " ".join(str(error).split())
    return text[:220]


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


def load_dependencies() -> None:
    global np, pd, plt, Line2D, Circle
    global u, SkyCoord, Galactocentric, Gaia

    import numpy as numpy_module
    import pandas as pandas_module
    import matplotlib.pyplot as pyplot_module
    from matplotlib.lines import Line2D as line2d_class
    from matplotlib.patches import Circle as circle_class
    from astropy import units as units_module
    from astropy.coordinates import (
        Galactocentric as galactocentric_class,
        SkyCoord as skycoord_class,
    )
    from astroquery.gaia import Gaia as gaia_class

    np = numpy_module
    pd = pandas_module
    plt = pyplot_module
    Line2D = line2d_class
    Circle = circle_class
    u = units_module
    SkyCoord = skycoord_class
    Galactocentric = galactocentric_class
    Gaia = gaia_class


def build_query(row_limit: int) -> str:
    minimum_parallax_mas = 1.0 / MAX_DISTANCE_KPC
    maximum_parallax_mas = 1.0 / MIN_DISTANCE_KPC

    return f"""
    SELECT TOP {row_limit}
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


def table_to_frame(table):
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

    return frame.dropna(subset=numeric_columns).reset_index(drop=True)


def query_gaia():
    Gaia.ROW_LIMIT = -1
    query = build_query(MAX_ROWS)
    failures = []

    for attempt in range(1, ASYNC_RETRIES + 1):
        try:
            job = Gaia.launch_job_async(
                query,
                dump_to_file=False,
                output_format="csv",
                verbose=False,
            )
            frame = table_to_frame(job.get_results())
            return frame, "ASYNC", attempt
        except Exception as error:
            failures.append(compact_error(error))
            print(
                f"GAIA TAP RETRY: {attempt}/{ASYNC_RETRIES} | "
                f"{failures[-1]}"
            )
            if attempt < ASYNC_RETRIES:
                time.sleep(RETRY_DELAY_SECONDS * attempt)

    print(
        "GAIA TAP FALLBACK: synchronous query | "
        f"TOP {SYNC_FALLBACK_ROWS}"
    )
    try:
        fallback_job = Gaia.launch_job(
            build_query(SYNC_FALLBACK_ROWS),
            dump_to_file=False,
            output_format="csv",
            verbose=False,
        )
        frame = table_to_frame(fallback_job.get_results())
        return frame, "SYNC_FALLBACK", ASYNC_RETRIES + 1
    except Exception as error:
        failures.append(compact_error(error))
        raise RuntimeError(
            "Gaia TAP failed after asynchronous retries and synchronous "
            f"fallback. Last error: {failures[-1]}"
        ) from error


def transform_catalog(frame):
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
    clipped = np.clip(
        luminosity_proxy,
        level_edges[0],
        np.nextafter(level_edges[-1], level_edges[0]),
    )
    level_index = (
        np.digitize(clipped, level_edges[1:-1], right=False) + 1
    )

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


def luminosity_labels():
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
            labels.append(
                f"L{index + 1}: {lower:.4g}–{upper:.4g} L☉"
            )
    return labels


def add_reference_markers(axis, sun_xyz, projection: str) -> None:
    sun_x, sun_y, sun_z = sun_xyz

    if projection == "xy":
        axis.scatter(
            [0.0], [0.0], marker="+", s=80,
            linewidths=1.0, color="black", zorder=5,
        )
        axis.scatter(
            [sun_x], [sun_y], marker="*", s=90,
            linewidths=0.5, color="black", zorder=5,
        )
        axis.add_patch(
            Circle(
                (0.0, 0.0), 15.0, fill=False,
                linewidth=0.5, linestyle="--", alpha=0.45,
            )
        )
        axis.add_patch(
            Circle(
                (0.0, 0.0), 25.0, fill=False,
                linewidth=0.5, linestyle=":", alpha=0.35,
            )
        )
    elif projection == "xz":
        axis.scatter(
            [0.0], [0.0], marker="+", s=80,
            linewidths=1.0, color="black", zorder=5,
        )
        axis.scatter(
            [sun_x], [sun_z], marker="*", s=90,
            linewidths=0.5, color="black", zorder=5,
        )
    elif projection == "yz":
        axis.scatter(
            [0.0], [0.0], marker="+", s=80,
            linewidths=1.0, color="black", zorder=5,
        )
        axis.scatter(
            [sun_y], [sun_z], marker="*", s=90,
            linewidths=0.5, color="black", zorder=5,
        )


def plot_composite(frame, sun_xyz) -> Path:
    output_path = OUTPUT_DIR / f"{VERSION}_COMPOSITE.png"
    color_map = plt.get_cmap("plasma", LUMINOSITY_LEVELS)
    colors = color_map(
        frame["luminosity_level"].to_numpy(dtype=int) - 1
    )

    figure = plt.figure(figsize=(15, 12))
    grid = figure.add_gridspec(
        2, 2, hspace=0.22, wspace=0.18
    )

    axis_xy = figure.add_subplot(grid[0, 0])
    axis_xz = figure.add_subplot(grid[0, 1])
    axis_yz = figure.add_subplot(grid[1, 0])
    axis_3d = figure.add_subplot(grid[1, 1], projection="3d")

    axis_xy.scatter(
        frame["x_gc_kpc"], frame["y_gc_kpc"],
        c=colors, s=2.0, alpha=0.58,
        linewidths=0.0, rasterized=True,
    )
    axis_xy.set_title("Milky Way bird's-eye view — Galactic X–Y")
    axis_xy.set_xlabel("X from Galactic center (kpc)")
    axis_xy.set_ylabel("Y from Galactic center (kpc)")
    axis_xy.set_aspect("equal", adjustable="box")
    add_reference_markers(axis_xy, sun_xyz, "xy")
    axis_xy.grid(True, linewidth=0.3, alpha=0.35)

    axis_xz.scatter(
        frame["x_gc_kpc"], frame["z_gc_kpc"],
        c=colors, s=2.0, alpha=0.58,
        linewidths=0.0, rasterized=True,
    )
    axis_xz.set_title("Galactic X–Z")
    axis_xz.set_xlabel("X from Galactic center (kpc)")
    axis_xz.set_ylabel("Z from Galactic plane (kpc)")
    add_reference_markers(axis_xz, sun_xyz, "xz")
    axis_xz.grid(True, linewidth=0.3, alpha=0.35)

    axis_yz.scatter(
        frame["y_gc_kpc"], frame["z_gc_kpc"],
        c=colors, s=2.0, alpha=0.58,
        linewidths=0.0, rasterized=True,
    )
    axis_yz.set_title("Galactic Y–Z")
    axis_yz.set_xlabel("Y from Galactic center (kpc)")
    axis_yz.set_ylabel("Z from Galactic plane (kpc)")
    add_reference_markers(axis_yz, sun_xyz, "yz")
    axis_yz.grid(True, linewidth=0.3, alpha=0.35)

    axis_3d.scatter(
        frame["x_gc_kpc"],
        frame["y_gc_kpc"],
        frame["z_gc_kpc"],
        c=colors,
        s=1.0,
        alpha=0.42,
        linewidths=0.0,
        rasterized=True,
    )
    axis_3d.scatter(
        [0.0], [0.0], [0.0],
        marker="+", s=70, color="black", linewidths=1.0,
    )
    axis_3d.scatter(
        [sun_xyz[0]], [sun_xyz[1]], [sun_xyz[2]],
        marker="*", s=75, color="black", linewidths=0.5,
    )
    axis_3d.set_title("Galactocentric 3D view")
    axis_3d.set_xlabel("X (kpc)")
    axis_3d.set_ylabel("Y (kpc)")
    axis_3d.set_zlabel("Z (kpc)")
    axis_3d.view_init(elev=22.0, azim=42.0)

    handles = [
        Line2D(
            [0], [0], marker="o", linestyle="",
            markersize=5, markerfacecolor=color_map(index),
            markeredgewidth=0.0, label=label,
        )
        for index, label in enumerate(luminosity_labels())
    ]
    handles.extend(
        [
            Line2D(
                [0], [0], marker="+", color="black",
                linestyle="", markersize=8, label="Galactic center",
            ),
            Line2D(
                [0], [0], marker="*", color="black",
                linestyle="", markersize=8, label="Sun",
            ),
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
    plt.show()
    plt.close(figure)
    return output_path


def save_outputs(frame):
    catalog_path = OUTPUT_DIR / f"{VERSION}_GAIA_DR3_SAMPLE.csv"
    summary_path = (
        OUTPUT_DIR / f"{VERSION}_LUMINOSITY_LEVEL_SUMMARY.csv"
    )
    frame.to_csv(catalog_path, index=False)

    summary = (
        frame.groupby("luminosity_level", as_index=False)
        .agg(
            star_count=("source_id", "count"),
            minimum_distance_kpc=("distance_kpc", "min"),
            median_distance_kpc=("distance_kpc", "median"),
            maximum_distance_kpc=("distance_kpc", "max"),
            minimum_luminosity_lsun=(
                "luminosity_proxy_lsun", "min"
            ),
            median_luminosity_lsun=(
                "luminosity_proxy_lsun", "median"
            ),
            maximum_luminosity_lsun=(
                "luminosity_proxy_lsun", "max"
            ),
        )
    )
    labels = luminosity_labels()
    summary["level_label"] = summary["luminosity_level"].map(
        {index + 1: labels[index] for index in range(8)}
    )
    summary.to_csv(summary_path, index=False)
    return catalog_path, summary_path


def print_summary(
    frame,
    query_mode: str,
    query_attempt: int,
    catalog_path: Path,
    summary_path: Path,
    plot_path: Path,
) -> None:
    print()
    print(f"{'METRIC':<30} {'VALUE':>18}")
    print(f"{'-' * 30} {'-' * 18}")
    print(f"{'Query mode':<30} {query_mode:>18}")
    print(f"{'Successful attempt':<30} {query_attempt:>18d}")
    print(f"{'Stars plotted':<30} {len(frame):>18,}")
    print(
        f"{'Minimum distance (kpc)':<30} "
        f"{frame['distance_kpc'].min():>18.6f}"
    )
    print(
        f"{'Median distance (kpc)':<30} "
        f"{frame['distance_kpc'].median():>18.6f}"
    )
    print(
        f"{'Maximum distance (kpc)':<30} "
        f"{frame['distance_kpc'].max():>18.6f}"
    )
    print(f"{'Luminosity levels':<30} {LUMINOSITY_LEVELS:>18d}")
    print()
    print(f"{'FILE':<30} PATH")
    print(f"{'-' * 30} {'-' * 72}")
    print(f"{'Catalog CSV':<30} {catalog_path}")
    print(f"{'Level summary CSV':<30} {summary_path}")
    print(f"{'Composite PNG':<30} {plot_path}")


def execute() -> None:
    print(f"CODE OUTPUT: {VERSION}")
    try:
        ensure_dependencies()
        load_dependencies()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        catalog, query_mode, query_attempt = query_gaia()
        if catalog.empty:
            raise RuntimeError(
                "Gaia query returned zero valid rows."
            )

        transformed, sun_xyz = transform_catalog(catalog)
        catalog_path, summary_path = save_outputs(transformed)
        plot_path = plot_composite(transformed, sun_xyz)
        print_summary(
            transformed,
            query_mode,
            query_attempt,
            catalog_path,
            summary_path,
            plot_path,
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
