#!/usr/bin/env python3
"""Download and extract the benchmark dataset bundle.

The archive is expected to already contain the final directory layout::

    lica-benchmarks-dataset/
        lica-data/
        benchmarks/

Usage::

    python scripts/download_data.py
    python scripts/download_data.py --out-dir /tmp
    python scripts/download_data.py --from-zip ~/lica-benchmarks-dataset.zip
"""

from __future__ import annotations

import argparse
import shutil
import ssl
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

LICA_DATA_URL = "https://storage.googleapis.com/lica-ml/lica-benchmarks-dataset.zip"
DEFAULT_OUT_DIR = Path("data")
BUNDLE_NAME = "lica-benchmarks-dataset"
LICA_DATA_DIR = "lica-data"


def _download(url: str, dest: Path) -> None:
    """Download *url* to *dest*, trying curl first then urllib."""
    if shutil.which("curl"):
        print(f"Downloading (curl) {url}")
        subprocess.check_call(
            ["curl", "-fSL", "--progress-bar", "-o", str(dest), url],
        )
        return

    print(f"Downloading (urllib) {url}")
    ctx = ssl.create_default_context()
    try:
        resp = urlopen(Request(url, headers={"User-Agent": "lica-bench/0.1"}), context=ctx)
    except ssl.SSLCertVerificationError as exc:
        print(
            "WARNING: TLS certificate verification failed; retrying without verification "
            "(traffic is not authenticated). Prefer fixing your Python/OpenSSL trust store, "
            "or install curl and re-run so the download uses curl instead.\n"
            f"  Detail: {exc}",
            file=sys.stderr,
        )
        ctx = ssl._create_unverified_context()
        resp = urlopen(Request(url, headers={"User-Agent": "lica-bench/0.1"}), context=ctx)

    total = resp.headers.get("Content-Length")
    downloaded = 0
    with open(dest, "wb") as f:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / int(total) * 100
                print(f"\r  {downloaded / 1e6:.1f} / {int(total) / 1e6:.1f} MB ({pct:.0f}%)", end="", flush=True)
            else:
                print(f"\r  {downloaded / 1e6:.1f} MB", end="", flush=True)
    print()
    resp.close()


def download_and_extract(
    out_dir: Path,
    *,
    url: str = LICA_DATA_URL,
    from_zip: Path | None = None,
) -> Path:
    """Download (or read) the zip and lay out ``lica-benchmarks-dataset/``.

    Returns the path to ``<out-dir>/lica-benchmarks-dataset/``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_root = out_dir / BUNDLE_NAME
    lica_data_dir = bundle_root / LICA_DATA_DIR
    metadata_csv = lica_data_dir / "metadata.csv"

    if metadata_csv.is_file():
        print(f"Already exists: {bundle_root}")
        return bundle_root

    if from_zip:
        zip_path = from_zip
    else:
        zip_path = out_dir / "lica-benchmarks-dataset.zip"
        _download(url, zip_path)

    print(f"Extracting {zip_path} → {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)

    if not from_zip:
        zip_path.unlink(missing_ok=True)

    if not metadata_csv.is_file():
        print(
            f"ERROR: expected {metadata_csv} after extraction. "
            "The archive does not match the expected dataset layout.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Done: {bundle_root}")
    return bundle_root


def main() -> None:
    parser = argparse.ArgumentParser(
        description=f"Download / unpack into {BUNDLE_NAME}/ ({LICA_DATA_DIR}/ + benchmarks/)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT_DIR,
        help=f"Parent directory for {BUNDLE_NAME}/ (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--from-zip", type=Path, default=None,
        help="Path to a local lica-benchmarks-dataset.zip (skip download)",
    )
    args = parser.parse_args()
    download_and_extract(args.out_dir, from_zip=args.from_zip)


if __name__ == "__main__":
    main()
