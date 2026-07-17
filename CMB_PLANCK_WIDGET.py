"""Interactive Planck PR3 CMB map viewer for Google Colab/Jupyter.

Run directly with:
    %run CMB_PLANCK_WIDGET.py

The script downloads official Planck Public Release 3 HEALPix FITS maps on
first use, caches them locally, and provides an ipywidgets-based viewer.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple


def _ensure_package(import_name: str, pip_name: Optional[str] = None) -> None:
    """Install a missing runtime dependency, then import it."""
    try:
        importlib.import_module(import_name)
    except ImportError:
        package = pip_name or import_name
        print(f"Installing required package: {package} ...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", package]
        )
        importlib.invalidate_caches()


for _import_name, _pip_name in (
    ("numpy", None),
    ("matplotlib", None),
    ("astropy", None),
    ("healpy", None),
    ("ipywidgets", None),
):
    _ensure_package(_import_name, _pip_name)

import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import ipywidgets as widgets
from IPython.display import clear_output, display


BASE_URL = (
    "https://irsa.ipac.caltech.edu/data/Planck/release_3/"
    "all-sky-maps/maps/component-maps/cmb"
)

PRODUCTS: Dict[str, str] = {
    "SMICA-NOSZ": "COM_CMB_IQU-smica-nosz_2048_R3.00_full.fits",
    "SMICA": "COM_CMB_IQU-smica_2048_R3.00_full.fits",
    "NILC": "COM_CMB_IQU-nilc_2048_R3.00_full.fits",
    "Commander": "COM_CMB_IQU-commander_2048_R3.00_full.fits",
    "SEVEM": "COM_CMB_IQU-sevem_2048_R3.01_full.fits",
}

CACHE_DIR = Path(
    os.environ.get("PLANCK_CMB_CACHE", str(Path.home() / ".cache" / "planck_pr3_cmb"))
).expanduser()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_MAP_CACHE: Dict[Tuple[str, int, float], np.ndarray] = {}
_CACHE_LOCK = threading.Lock()


def _human_bytes(value: float) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(value)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _fits_sanity_check(path: Path) -> bool:
    try:
        if not path.is_file() or path.stat().st_size < 2880:
            return False
        with path.open("rb") as handle:
            header = handle.read(80)
        return header.startswith(b"SIMPLE")
    except OSError:
        return False


def _download_product(product: str, status: widgets.HTML) -> Path:
    filename = PRODUCTS[product]
    destination = CACHE_DIR / filename

    if _fits_sanity_check(destination):
        status.value = (
            f"<span style='color:#2e7d32'><b>Cached:</b> {filename} "
            f"({_human_bytes(destination.stat().st_size)})</span>"
        )
        return destination

    if destination.exists():
        try:
            destination.unlink()
        except OSError:
            pass

    url = f"{BASE_URL}/{filename}"
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        try:
            temporary.unlink()
        except OSError:
            pass

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 Planck-CMB-Widget/1.0",
            "Accept": "application/fits,application/octet-stream,*/*",
        },
    )

    try:
        status.value = (
            f"<span style='color:#1565c0'><b>Downloading:</b> {filename}</span>"
        )
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as out:
            total = int(response.headers.get("Content-Length", "0") or 0)
            downloaded = 0
            last_update = 0.0
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                out.write(block)
                downloaded += len(block)
                now = time.monotonic()
                if now - last_update >= 0.35:
                    if total:
                        pct = 100.0 * downloaded / total
                        status.value = (
                            f"<span style='color:#1565c0'><b>Downloading:</b> "
                            f"{filename} — {_human_bytes(downloaded)} / "
                            f"{_human_bytes(total)} ({pct:.1f}%)</span>"
                        )
                    else:
                        status.value = (
                            f"<span style='color:#1565c0'><b>Downloading:</b> "
                            f"{filename} — {_human_bytes(downloaded)}</span>"
                        )
                    last_update = now

        temporary.replace(destination)
        if not _fits_sanity_check(destination):
            raise RuntimeError("Downloaded file did not pass the FITS integrity check.")
        status.value = (
            f"<span style='color:#2e7d32'><b>Download complete:</b> {filename} "
            f"({_human_bytes(destination.stat().st_size)})</span>"
        )
        return destination

    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, RuntimeError) as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(
            f"Could not download {filename}. Check the internet connection or try again. "
            f"Source: {url}. Details: {exc}"
        ) from exc


def _read_temperature_map(path: Path) -> np.ndarray:
    try:
        cmb_k = hp.read_map(str(path), field=0, dtype=np.float32, verbose=False)
    except TypeError:
        cmb_k = hp.read_map(str(path), field=0, dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Unable to read the FITS map: {exc}") from exc

    cmb_uk = np.asarray(cmb_k, dtype=np.float32) * np.float32(1.0e6)
    cmb_uk[~np.isfinite(cmb_uk)] = hp.UNSEEN
    return cmb_uk


def _prepare_map(
    product: str,
    target_nside: int,
    smoothing_fwhm_deg: float,
    path: Path,
    status: widgets.HTML,
) -> np.ndarray:
    key = (product, int(target_nside), round(float(smoothing_fwhm_deg), 4))
    with _CACHE_LOCK:
        cached = _MAP_CACHE.get(key)
    if cached is not None:
        status.value = (
            f"<span style='color:#2e7d32'><b>Ready:</b> using processed map cache "
            f"({product}, Nside={target_nside}).</span>"
        )
        return cached

    status.value = "<span style='color:#1565c0'><b>Reading FITS map...</b></span>"
    sky = _read_temperature_map(path)
    original_nside = hp.get_nside(sky)

    if target_nside != original_nside:
        status.value = (
            f"<span style='color:#1565c0'><b>Resampling:</b> Nside "
            f"{original_nside} → {target_nside}...</span>"
        )
        sky = hp.ud_grade(
            sky,
            nside_out=target_nside,
            order_in="RING",
            order_out="RING",
            power=0,
            dtype=np.float32,
        )

    if smoothing_fwhm_deg > 0.0:
        status.value = (
            f"<span style='color:#1565c0'><b>Smoothing:</b> Gaussian FWHM "
            f"{smoothing_fwhm_deg:g}°...</span>"
        )
        sky = hp.smoothing(
            sky,
            fwhm=np.radians(smoothing_fwhm_deg),
            verbose=False,
        ).astype(np.float32, copy=False)

    with _CACHE_LOCK:
        if len(_MAP_CACHE) >= 8:
            _MAP_CACHE.pop(next(iter(_MAP_CACHE)))
        _MAP_CACHE[key] = sky
    return sky


def _coordinate_code(label: str) -> str:
    return {"Galactic": "G", "Equatorial": "C", "Ecliptic": "E"}[label]


def _overlay_galactic_plane() -> None:
    longitude = np.linspace(-180.0, 180.0, 721)
    latitude = np.zeros_like(longitude)
    hp.projplot(
        longitude,
        latitude,
        lonlat=True,
        coord="G",
        linewidth=1.35,
        linestyle="--",
        alpha=0.9,
    )


def _render_map(
    sky: np.ndarray,
    product: str,
    projection: str,
    coordinates: str,
    color_limit: float,
    smoothing: float,
    nside: int,
    show_grid: bool,
    show_plane: bool,
) -> None:
    coord_code = _coordinate_code(coordinates)
    coord = "G" if coord_code == "G" else ["G", coord_code]
    title = (
        f"Planck PR3 {product} CMB Temperature | {coordinates} | "
        f"Nside={nside} | FWHM={smoothing:g}°"
    )

    plt.close("all")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 12,
        }
    )

    common = dict(
        coord=coord,
        min=-float(color_limit),
        max=float(color_limit),
        unit=r"$\mu$K$_{\mathrm{CMB}}$",
        title=title,
        cmap="RdBu_r",
        badcolor="0.65",
        bgcolor="white",
        notext=False,
        cbar=True,
        hold=False,
    )

    if projection == "Mollweide":
        plt.figure(figsize=(14, 7.8))
        hp.mollview(sky, xsize=1800, **common)
    else:
        plt.figure(figsize=(15, 7.5))
        hp.cartview(
            sky,
            xsize=1800,
            lonra=[-180, 180],
            latra=[-90, 90],
            **common,
        )

    if show_grid:
        hp.graticule(dpar=30, dmer=30, verbose=False)
    if show_plane:
        _overlay_galactic_plane()

    fig = plt.gcf()
    fig.patch.set_facecolor("white")
    plt.show()


product_widget = widgets.Dropdown(
    options=list(PRODUCTS.keys()),
    value="SMICA-NOSZ",
    description="Product:",
    layout=widgets.Layout(width="300px"),
)
projection_widget = widgets.ToggleButtons(
    options=("Mollweide", "Cartesian"),
    value="Mollweide",
    description="Projection:",
)
coordinate_widget = widgets.Dropdown(
    options=("Galactic", "Equatorial", "Ecliptic"),
    value="Galactic",
    description="Coordinates:",
    layout=widgets.Layout(width="300px"),
)
color_widget = widgets.FloatSlider(
    value=500.0,
    min=25.0,
    max=1000.0,
    step=25.0,
    description="Scale ±µK:",
    continuous_update=False,
    readout_format=".0f",
    layout=widgets.Layout(width="500px"),
)
smoothing_widget = widgets.FloatSlider(
    value=0.0,
    min=0.0,
    max=5.0,
    step=0.1,
    description="FWHM (°):",
    continuous_update=False,
    readout_format=".1f",
    layout=widgets.Layout(width="500px"),
)
resolution_widget = widgets.Dropdown(
    options=[
        ("Nside 128  (0.46° pixels, fast)", 128),
        ("Nside 256  (0.23° pixels)", 256),
        ("Nside 512  (0.11° pixels)", 512),
        ("Nside 1024 (0.057° pixels)", 1024),
        ("Nside 2048 (native, memory intensive)", 2048),
    ],
    value=512,
    description="Resolution:",
    layout=widgets.Layout(width="430px"),
)
grid_widget = widgets.Checkbox(value=True, description="Coordinate grid")
plane_widget = widgets.Checkbox(value=False, description="Galactic plane overlay")
render_button = widgets.Button(
    description="Load / Render Map",
    button_style="primary",
    icon="globe",
    layout=widgets.Layout(width="190px", height="38px"),
)
clear_cache_button = widgets.Button(
    description="Clear RAM cache",
    icon="trash",
    layout=widgets.Layout(width="160px", height="38px"),
)
status_widget = widgets.HTML(
    value=(
        "<span style='color:#455a64'><b>Ready.</b> Select settings, then click "
        "<i>Load / Render Map</i>. The first download can be several hundred MB.</span>"
    )
)
output_widget = widgets.Output()


def _set_controls_disabled(disabled: bool) -> None:
    for control in (
        product_widget,
        projection_widget,
        coordinate_widget,
        color_widget,
        smoothing_widget,
        resolution_widget,
        grid_widget,
        plane_widget,
        render_button,
        clear_cache_button,
    ):
        control.disabled = disabled


def _on_render(_button: widgets.Button) -> None:
    _set_controls_disabled(True)
    try:
        product = product_widget.value
        path = _download_product(product, status_widget)
        sky = _prepare_map(
            product=product,
            target_nside=int(resolution_widget.value),
            smoothing_fwhm_deg=float(smoothing_widget.value),
            path=path,
            status=status_widget,
        )
        status_widget.value = (
            "<span style='color:#1565c0'><b>Rendering publication-quality map...</b></span>"
        )
        with output_widget:
            clear_output(wait=True)
            _render_map(
                sky=sky,
                product=product,
                projection=projection_widget.value,
                coordinates=coordinate_widget.value,
                color_limit=float(color_widget.value),
                smoothing=float(smoothing_widget.value),
                nside=int(resolution_widget.value),
                show_grid=bool(grid_widget.value),
                show_plane=bool(plane_widget.value),
            )
        status_widget.value = (
            f"<span style='color:#2e7d32'><b>Rendered:</b> {product}, "
            f"{projection_widget.value}, {coordinate_widget.value}, "
            f"Nside={resolution_widget.value}.</span>"
        )
    except Exception as exc:
        status_widget.value = (
            "<div style='color:#b71c1c'><b>Unable to render map.</b><br>"
            f"{str(exc)}</div>"
        )
        with output_widget:
            clear_output(wait=True)
            print("Planck CMB widget error:")
            print(str(exc))
    finally:
        _set_controls_disabled(False)


def _on_clear_cache(_button: widgets.Button) -> None:
    with _CACHE_LOCK:
        count = len(_MAP_CACHE)
        _MAP_CACHE.clear()
    status_widget.value = (
        f"<span style='color:#2e7d32'><b>RAM cache cleared:</b> "
        f"{count} processed map(s). Downloaded FITS files were retained.</span>"
    )


render_button.on_click(_on_render)
clear_cache_button.on_click(_on_clear_cache)

header = widgets.HTML(
    value="""
    <div style="padding:12px 16px;border-radius:10px;background:#0b172a;color:white;">
      <div style="font-size:22px;font-weight:700;">CMB PLANCK INTERACTIVE WIDGET V01</div>
      <div style="font-size:13px;opacity:0.88;margin-top:4px;">
        Official Planck PR3 component-separated temperature maps · HEALPix · µK<sub>CMB</sub>
      </div>
    </div>
    """
)

controls = widgets.VBox(
    [
        header,
        widgets.HBox([product_widget, coordinate_widget]),
        projection_widget,
        resolution_widget,
        color_widget,
        smoothing_widget,
        widgets.HBox([grid_widget, plane_widget]),
        widgets.HBox([render_button, clear_cache_button]),
        status_widget,
    ],
    layout=widgets.Layout(
        border="1px solid #cfd8dc",
        padding="12px",
        width="100%",
        max_width="1000px",
    ),
)

display(controls, output_widget)
