"""
install.py — one-time setup for flm-tem-alignment

Called by setup.bat / setup.sh in two stages:
  uv run --no-sync python install.py clone    # clone model repos (must run BEFORE uv sync)
  uv run python install.py weights  # download model weights (run AFTER uv sync)

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

SAM_REPO       = BASE / "segment_anything_2"
LIGHTGLUE_REPO = BASE / "LightGlue"
SWINIR_REPO    = BASE / "SwinIR"

SAM_CHECKPOINT_DIR    = SAM_REPO    / "checkpoints"
SWINIR_CHECKPOINT_DIR = SWINIR_REPO / "checkpoints"

SAM_BASE_URL    = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
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
        print(f"  {name}: already present")
        return
    print(f"  Cloning {name} ...")
    ok = _run(f'git clone "{url}" "{dest}"')
    if ok:
        _run(f'git -C "{dest}" checkout {commit}')
        print(f"  {name}: pinned to {commit}")


def download_weight(filename: str, dest_dir: Path, base_url: str) -> None:
    dest = dest_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        mb = dest.stat().st_size / 1_048_576
        print(f"  {filename}: already present ({mb:.1f} MB)")
        return

    url = f"{base_url}/{filename}"
    print(f"  Downloading {filename} ...")
    try:
        urllib.request.urlretrieve(url, dest)
        mb = dest.stat().st_size / 1_048_576
        print(f"  {filename}: done ({mb:.1f} MB)")
    except Exception as exc:
        print(f"  [ERROR] could not download {filename}: {exc}")
        if dest.exists():
            dest.unlink()


def cmd_clone() -> None:
    print("\n=== Cloning model repos ===")
    for name, info in REPO_PINS.items():
        clone_at_commit(name, info["url"], info["commit"], info["path"])
    print("  SwinIR has no Python package — loaded via sys.path at runtime")


def cmd_weights() -> None:
    print("\n=== Downloading SAM2 weights ===")
    SAM_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for w in SAM2_WEIGHTS:
        download_weight(w, SAM_CHECKPOINT_DIR, SAM_BASE_URL)

    print("\n=== Downloading SwinIR weights ===")
    SWINIR_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    for w in SWINIR_WEIGHTS:
        download_weight(w, SWINIR_CHECKPOINT_DIR, SWINIR_BASE_URL)

    print("\nLightGlue weights (SuperPoint) download on first model load — that is normal.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    if cmd == "clone":
        cmd_clone()
    elif cmd == "weights":
        cmd_weights()
    else:
        print("Usage: python install.py clone | weights")
        sys.exit(1)
