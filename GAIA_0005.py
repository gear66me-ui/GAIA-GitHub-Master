# GAIA_0005
# Run the verified GAIA_0004 sector-balanced engine at 1,000 stars per sector.

from __future__ import annotations

import os
from pathlib import Path
from urllib.request import Request, urlopen

VERSION = "GAIA_0005"
BASE_VERSION = "GAIA_0004"
ROWS_PER_SECTOR = 1000
ENGINE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "GAIA-GitHub-Master/main/GAIA_0004.py"
)
ENGINE_PATH = Path("/content/GAIA_0005_ENGINE.py")


def download_engine() -> str:
    request = Request(
        ENGINE_URL,
        headers={"User-Agent": "GAIA-Colab/0005"},
    )
    with urlopen(request, timeout=120) as response:
        source = response.read().decode("utf-8")

    if 'VERSION = "GAIA_0004"' not in source:
        raise RuntimeError("GAIA_0004 engine version marker was not found.")
    if 'CACHE_DIR = OUTPUT_DIR / "GAIA_0004_SECTOR_CACHE"' not in source:
        raise RuntimeError("GAIA_0004 cache marker was not found.")

    source = source.replace(
        "# GAIA_0004",
        "# GAIA_0005",
        1,
    )
    source = source.replace(
        'VERSION = "GAIA_0004"',
        'VERSION = "GAIA_0005"',
        1,
    )
    source = source.replace(
        'CACHE_DIR = OUTPUT_DIR / "GAIA_0004_SECTOR_CACHE"',
        'CACHE_DIR = OUTPUT_DIR / "GAIA_0005_SECTOR_CACHE"',
        1,
    )
    source = source.replace(
        'ROWS_PER_SECTOR = int(os.environ.get("GAIA_ROWS_PER_SECTOR", "80"))',
        'ROWS_PER_SECTOR = int(os.environ.get("GAIA_ROWS_PER_SECTOR", "1000"))',
        1,
    )
    source = source.replace(
        "{len(frame):03d} stars",
        "{len(frame):04d} stars",
    )
    return source


def main() -> None:
    os.environ["GAIA_ROWS_PER_SECTOR"] = str(ROWS_PER_SECTOR)
    source = download_engine()
    ENGINE_PATH.write_text(source, encoding="utf-8")
    compiled = compile(source, str(ENGINE_PATH), "exec")
    namespace = {
        "__name__": "__main__",
        "__file__": str(ENGINE_PATH),
        "__package__": None,
    }
    exec(compiled, namespace)


if __name__ == "__main__":
    main()
