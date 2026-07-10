# GAIA_UPLOAD_0001
# Upload the newest GAIA PNG from Colab to the GAIA GitHub repository.

from __future__ import annotations

import base64
import getpass
import os
from datetime import datetime
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import requests

VERSION = "GAIA_UPLOAD_0001"
OWNER = "gear66me-ui"
REPOSITORY = "GAIA-GitHub-Master"
BRANCH = "main"
SOURCE_DIRECTORY = Path("/content/GAIA_OUTPUT")
REMOTE_DIRECTORY = "OUTPUT_PNG"
TOKEN_SECRET_NAME = "GITHUB_TOKEN"
API_VERSION = "2026-03-10"
MAX_FILE_SIZE_BYTES = 95 * 1024 * 1024


def colombia_timestamp() -> str:
    return datetime.now(ZoneInfo("America/Bogota")).strftime(
        "%Y-%m-%d %H:%M:%S America/Bogota"
    )


def get_github_token() -> str:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if token:
        return token

    try:
        from google.colab import userdata

        token = userdata.get(TOKEN_SECRET_NAME)
        if token:
            return token.strip()
    except Exception:
        pass

    token = getpass.getpass(
        "GitHub token (hidden; requires Contents read/write): "
    ).strip()
    if not token:
        raise RuntimeError("No GitHub token was supplied.")
    return token


def newest_png() -> Path:
    if not SOURCE_DIRECTORY.exists():
        raise FileNotFoundError(
            f"Source directory does not exist: {SOURCE_DIRECTORY}"
        )

    png_files = [
        path
        for path in SOURCE_DIRECTORY.rglob("*.png")
        if path.is_file()
    ]
    if not png_files:
        raise FileNotFoundError(
            f"No PNG files were found under {SOURCE_DIRECTORY}"
        )

    return max(png_files, key=lambda path: path.stat().st_mtime)


def upload_file(source_path: Path, token: str) -> tuple[str, str, str]:
    file_size = source_path.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        raise RuntimeError(
            f"File is too large for this uploader: {file_size:,} bytes"
        )

    remote_path = f"{REMOTE_DIRECTORY}/{source_path.name}"
    encoded_remote_path = quote(remote_path, safe="/")
    api_url = (
        f"https://api.github.com/repos/{OWNER}/{REPOSITORY}/contents/"
        f"{encoded_remote_path}"
    )

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": API_VERSION,
    }

    existing_response = requests.get(
        api_url,
        headers=headers,
        params={"ref": BRANCH},
        timeout=60,
    )

    existing_sha = None
    if existing_response.status_code == 200:
        existing_sha = existing_response.json().get("sha")
    elif existing_response.status_code != 404:
        raise RuntimeError(
            "GitHub lookup failed: "
            f"HTTP {existing_response.status_code} | "
            f"{existing_response.text[:300]}"
        )

    encoded_content = base64.b64encode(
        source_path.read_bytes()
    ).decode("ascii")

    payload = {
        "message": f"Upload {source_path.name} from Google Colab",
        "content": encoded_content,
        "branch": BRANCH,
    }
    if existing_sha:
        payload["sha"] = existing_sha

    upload_response = requests.put(
        api_url,
        headers=headers,
        json=payload,
        timeout=180,
    )
    if upload_response.status_code not in (200, 201):
        raise RuntimeError(
            "GitHub upload failed: "
            f"HTTP {upload_response.status_code} | "
            f"{upload_response.text[:500]}"
        )

    result = upload_response.json()
    github_url = result["content"]["html_url"]
    raw_url = (
        f"https://raw.githubusercontent.com/{OWNER}/{REPOSITORY}/"
        f"{BRANCH}/{remote_path}"
    )
    action = "UPDATED" if existing_sha else "CREATED"
    return action, github_url, raw_url


def main() -> None:
    print(f"CODE OUTPUT: {VERSION}")
    try:
        source_path = newest_png()
        token = get_github_token()
        action, github_url, raw_url = upload_file(source_path, token)

        print()
        print(f"{'STATUS':<24} {action}")
        print(f"{'LOCAL PNG':<24} {source_path}")
        print(f"{'FILE SIZE BYTES':<24} {source_path.stat().st_size:,}")
        print(f"{'GITHUB LINK':<24} {github_url}")
        print(f"{'RAW IMAGE LINK':<24} {raw_url}")
    except Exception as error:
        print()
        print("STATUS: FAILED")
        print(f"ERROR: {' '.join(str(error).split())}")
    finally:
        print()
        print(colombia_timestamp())
        print(f"END OF CODE OUTPUT: {VERSION}")


if __name__ == "__main__":
    main()
