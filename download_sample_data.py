"""
download_sample_data.py — download demo FLM/TEM images for flm-tem-alignment

Usage:
    python download_sample_data.py           # downloads both specimens
    python download_sample_data.py g3_l3     # downloads only JEY002_G3_L3
    python download_sample_data.py g3_l8     # downloads only JEY002_G3_L8

Data hosted on OSF: https://osf.io/459jg/
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent

DATASETS = {
    "g3_l3": {
        "url": "https://osf.io/pgurb/download",
        "zip": BASE / "jey_002_g3_l3.zip",
        "dest": BASE / "jey_002_g3_l3",
    },
    "g3_l8": {
        "url": "https://osf.io/sn46y/download",
        "zip": BASE / "jey_002_g3_l8.zip",
        "dest": BASE / "jey_002_g3_l8",
    },
}


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        mb = downloaded / 1_048_576
        total_mb = total_size / 1_048_576
        print(f"\r  {pct:.1f}%  {mb:.0f} / {total_mb:.0f} MB", end="", flush=True)


def download_and_extract(key: str, info: dict) -> None:
    dest = info["dest"]
    if dest.exists():
        print(f"  {dest.name}: already present, skipping")
        return

    zip_path = info["zip"]
    print(f"Downloading {dest.name} (~1 GB) ...")
    urllib.request.urlretrieve(info["url"], zip_path, reporthook=_progress)
    print()

    print(f"  Extracting ...")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)

    zip_path.unlink()
    print(f"  Done -> {dest}")


def main() -> None:
    keys = sys.argv[1:] or list(DATASETS.keys())
    for key in keys:
        if key not in DATASETS:
            print(f"Unknown dataset '{key}'. Choose from: {list(DATASETS.keys())}")
            sys.exit(1)
        download_and_extract(key, DATASETS[key])
    print("\nAll done.")


if __name__ == "__main__":
    main()
