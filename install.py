"""
install.py — one-time setup for flm-tem-alignment

Run from the repo root:
    python install.py

What it does:
  1. Clones segment_anything_2, LightGlue, SwinIR at pinned commits
  2. pip install -e for SAM2 and LightGlue
  3. Downloads SAM2 and SwinIR weight files (skips if already present)

Pinned commits (tested and known-good):
  segment_anything_2 : 2b90b9f  (facebookresearch/sam2)
  LightGlue          : eb42fee  (cvg/LightGlue)
  SwinIR             : 6545850  (JingyunLiang/SwinIR)

SAM2 weight source   : https://dl.fbaipublicfiles.com/segment_anything_2/092824/
SwinIR weight source : https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/
"""

import subprocess
import sys
import urllib.request
from pathlib import Path

BASE = Path(__file__).resolve().parent

SAM_REPO      = BASE / "segment_anything_2"
LIGHTGLUE_REPO = BASE / "LightGlue"
SWINIR_REPO   = BASE / "SwinIR"

SAM_CHECKPOINT_DIR   = SAM_REPO   / "checkpoints"
SWINIR_CHECKPOINT_DIR = SWINIR_REPO / "checkpoints"

SAM_BASE_URL   = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
SWINIR_BASE_URL = "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0"

REPO_PINS = {
    "segment_anything_2": {
        "url": "https://github.com/facebookresearch/sam2.git",
        "commit": "2b90b9f",
        "path": SAM_REPO,
    },
    "LightGlue": {
        "url": "https://github.com/cvg/LightGlue.git",
        "commit": "eb42fee",
        "path": LIGHTGLUE_REPO,
    },
    "SwinIR": {
        "url": "https://github.com/JingyunLiang/SwinIR.git",
        "commit": "6545850",
        "path": SWINIR_REPO,
    },
}

SAM2_WEIGHTS = [
    "sam2.1_hiera_tiny.pt",
    "sam2.1_hiera_small.pt",
    "sam2.1_hiera_large.pt",
]

SWINIR_WEIGHTS = [
    "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth",
    "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
]


def _run(cmd: str) -> bool:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [WARN] failed: {cmd!r}\n  {result.stderr[:400]}")
    return result.returncode == 0


def clone_at_commit(name: str, url: str, commit: str, dest: Path) -> None:
    if dest.exists():
        print(f"  {name}: already present at {dest}")
        return
    print(f"  Cloning {name} …")
    ok = _run(f'git clone "{url}" "{dest}"')
    if ok:
        _run(f'git -C "{dest}" checkout {commit}')
        print(f"  {name}: pinned to {commit}")


def pip_install_editable(path: Path) -> None:
    print(f"  pip install -e {path.name} …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(path)])


def download_weight(filename: str, dest_dir: Path, base_url: str) -> None:
    dest = dest_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        mb = dest.stat().st_size / 1_048_576
        print(f"  {filename}: already present ({mb:.1f} MB)")
        return

    url = f"{base_url}/{filename}"
    print(f"  Downloading {filename} …")
    try:
        urllib.request.urlretrieve(url, dest)
        mb = dest.stat().st_size / 1_048_576
        print(f"  {filename}: done ({mb:.1f} MB)")
    except Exception as exc:
        print(f"  [ERROR] could not download {filename}: {exc}")
        if dest.exists():
            dest.unlink()


def main() -> None:
    print("\n=== Step 1: clone repos ===")
    for name, info in REPO_PINS.items():
        clone_at_commit(name, info["url"], info["commit"], info["path"])

    print("\n=== Step 2: install packages ===")
    pip_install_editable(SAM_REPO)
    pip_install_editable(LIGHTGLUE_REPO)
    print("  SwinIR has no setup.py — loaded via sys.path in model_setup.py")

    print("\n=== Step 3: download SAM2 weights ===")
    SAM_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for w in SAM2_WEIGHTS:
        download_weight(w, SAM_CHECKPOINT_DIR, SAM_BASE_URL)

    print("\n=== Step 4: download SwinIR weights ===")
    SWINIR_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for w in SWINIR_WEIGHTS:
        download_weight(w, SWINIR_CHECKPOINT_DIR, SWINIR_BASE_URL)

    print("\n=== Step 5: install the napari plugin ===")
    plugin_dir = BASE / "plugin"
    if plugin_dir.exists():
        pip_install_editable(plugin_dir)
    else:
        print("  [WARN] plugin/ directory not found — skipping")

    print("\n=== Done ===")
    print("LightGlue weights (SuperPoint) download on first model load — that is normal.")
    print("Open napari and load the 'roi-detect' plugin to get started.")


if __name__ == "__main__":
    main()