# GAIA_0006
# Resume the 1,000-star sector map with chunked timeout-resistant Gaia queries.

from __future__ import annotations

import os
from pathlib import Path
from urllib.request import Request, urlopen

VERSION = "GAIA_0006"
ROWS_PER_SECTOR = 1000
ENGINE_URL = (
    "https://raw.githubusercontent.com/gear66me-ui/"
    "GAIA-GitHub-Master/main/GAIA_0004.py"
)
ENGINE_PATH = Path("/content/GAIA_0006_ENGINE.py")


def replacement_query_function() -> str:
    return r'''
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
    collected = pd.DataFrame()

    if cache_path.exists() and cache_path.stat().st_size > 200:
        cached = pd.read_csv(cache_path)
        cached = normalize_sector(cached, pd)
        cached = cached.drop_duplicates(subset=["source_id"]).reset_index(drop=True)
        if len(cached) >= ROWS_PER_SECTOR:
            return cached.iloc[:ROWS_PER_SECTOR].copy(), "CACHE", 0
        collected = cached.copy()

    l_upper_operator = "<=" if l_index == LONGITUDE_SECTORS - 1 else "<"
    b_upper_operator = "<=" if b_index == LATITUDE_SECTORS - 1 else "<"
    maximum_distance_pc = MAX_DISTANCE_KPC * 1000.0

    sector_seed = (
        (b_index * LONGITUDE_SECTORS + l_index) * 37
    ) % RANDOM_WINDOWS
    window_order = [
        (sector_seed + offset) % RANDOM_WINDOWS
        for offset in range(RANDOM_WINDOWS)
    ]

    successful_chunks = 0
    attempted_chunks = 0
    last_error = None

    for window_index in window_order:
        if len(collected) >= ROWS_PER_SECTOR:
            break

        random_min = int(
            window_index * RANDOM_INDEX_MAX / RANDOM_WINDOWS
        )
        random_max = int(
            (window_index + 1) * RANDOM_INDEX_MAX / RANDOM_WINDOWS
        )
        rows_needed = ROWS_PER_SECTOR - len(collected)
        chunk_limit = min(CHUNK_ROWS, rows_needed)

        query = f"""
        SELECT TOP {chunk_limit}
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
          AND gs.random_index >= {random_min}
          AND gs.random_index < {random_max}
          AND gs.ruwe < {MAX_RUWE:.6f}
          AND gs.phot_g_mean_mag IS NOT NULL
          AND gs.bp_rp IS NOT NULL
          AND ap.distance_gspphot IS NOT NULL
          AND ap.distance_gspphot > 0.0
          AND ap.distance_gspphot <= {maximum_distance_pc:.6f}
          AND ap.lum_flame IS NOT NULL
          AND ap.lum_flame > 0.0
        """

        attempted_chunks += 1
        chunk = None
        for attempt in range(1, CHUNK_RETRIES + 1):
            try:
                job = Gaia.launch_job(
                    query,
                    dump_to_file=False,
                    output_format="csv",
                    verbose=False,
                )
                chunk = normalize_sector(
                    job.get_results().to_pandas(), pd
                )
                break
            except Exception as error:
                last_error = error
                print(
                    f"  CHUNK {attempted_chunks:03d} | "
                    f"window {window_index:03d} | "
                    f"retry {attempt}/{CHUNK_RETRIES} | "
                    f"{compact_error(error)}"
                )
                if attempt < CHUNK_RETRIES:
                    time.sleep(4 * attempt)

        if chunk is None or chunk.empty:
            continue

        successful_chunks += 1
        collected = pd.concat(
            [collected, chunk],
            ignore_index=True,
        )
        collected = collected.drop_duplicates(
            subset=["source_id"]
        ).reset_index(drop=True)
        collected.to_csv(cache_path, index=False)

        print(
            f"  CHUNK {attempted_chunks:03d} | "
            f"window {window_index:03d} | "
            f"+{len(chunk):03d} | total {len(collected):04d}"
        )

    if collected.empty:
        raise RuntimeError(
            f"Sector L{l_index} B{b_index} produced no rows. "
            f"Last error: {compact_error(last_error)}"
        )

    collected = collected.iloc[:ROWS_PER_SECTOR].copy()
    collected.to_csv(cache_path, index=False)

    if len(collected) < ROWS_PER_SECTOR:
        print(
            f"  SECTOR PARTIAL | {len(collected):04d}/"
            f"{ROWS_PER_SECTOR:04d} stars after "
            f"{attempted_chunks} chunk windows"
        )
        return collected, "CHUNKED_PARTIAL", successful_chunks

    return collected, "CHUNKED_TAP", successful_chunks


'''


def build_engine() -> str:
    request = Request(
        ENGINE_URL,
        headers={"User-Agent": "GAIA-Colab/0006"},
    )
    with urlopen(request, timeout=120) as response:
        source = response.read().decode("utf-8")

    required_markers = [
        'VERSION = "GAIA_0004"',
        'CACHE_DIR = OUTPUT_DIR / "GAIA_0004_SECTOR_CACHE"',
        'ROWS_PER_SECTOR = int(os.environ.get("GAIA_ROWS_PER_SECTOR", "80"))',
        "def query_one_sector(",
        "def collect_balanced_catalog(",
    ]
    for marker in required_markers:
        if marker not in source:
            raise RuntimeError(f"Engine marker not found: {marker}")

    source = source.replace("# GAIA_0004", "# GAIA_0006", 1)
    source = source.replace(
        'VERSION = "GAIA_0004"',
        'VERSION = "GAIA_0006"',
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
        'QUERY_RETRIES = int(os.environ.get("GAIA_QUERY_RETRIES", "4"))',
        'QUERY_RETRIES = int(os.environ.get("GAIA_QUERY_RETRIES", "4"))\n'
        'CHUNK_ROWS = int(os.environ.get("GAIA_CHUNK_ROWS", "250"))\n'
        'CHUNK_RETRIES = int(os.environ.get("GAIA_CHUNK_RETRIES", "3"))\n'
        'RANDOM_WINDOWS = int(os.environ.get("GAIA_RANDOM_WINDOWS", "256"))\n'
        'RANDOM_INDEX_MAX = int(os.environ.get("GAIA_RANDOM_INDEX_MAX", "2000000000"))',
        1,
    )
    source = source.replace(
        "{len(frame):03d} stars",
        "{len(frame):04d} stars",
    )

    function_start = source.index("def query_one_sector(")
    function_end = source.index("def collect_balanced_catalog(")
    source = (
        source[:function_start]
        + replacement_query_function()
        + source[function_end:]
    )
    return source


def main() -> None:
    os.environ["GAIA_ROWS_PER_SECTOR"] = str(ROWS_PER_SECTOR)
    source = build_engine()
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
