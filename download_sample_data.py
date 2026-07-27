"""
download_sample_data.py — download demo FLM/TEM images for flm-tem-alignment

Usage:
    python download_sample_data.py           # downloads both specimens
    python download_sample_data.py g3_l3     # downloads only JEY002_G3_L3
    python download_sample_data.py g3_l8     # downloads only JEY002_G3_L8

Data hosted on OSF: https://osf.io/459jg/
"""

import sys
import zipfile
from pathlib import Path

import requests

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


def download_and_extract(key: str, info: dict) -> None:
    dest = info["dest"]
    if dest.exists():
        print(f"  {dest.name}: already present, skipping")
        return

    zip_path = info["zip"]
    print(f"Downloading {dest.name} ...")
    with requests.get(info["url"], stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                downloaded += len(chunk)
                mb = downloaded / 1_048_576
                if total:
                    pct = min(downloaded / total * 100, 100)
                    total_mb = total / 1_048_576
                    print(f"\r  {pct:.1f}%  {mb:.0f} / {total_mb:.0f} MB", end="", flush=True)
                else:
                    print(f"\r  {mb:.0f} MB downloaded...", end="", flush=True)
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
