import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_grad_enabled(False)

# ── Path constants ─────────────────────────────────────────────────────────────
DRIVE_BASE = Path('/content/drive/MyDrive')

SAM_REPO = DRIVE_BASE / 'segment_anything_2'
SAM_CHECKPOINT = SAM_REPO / 'checkpoints'


LIGHTGLUE_REPO = DRIVE_BASE / 'LightGlue'

SWINIR_REPO = DRIVE_BASE / 'SwinIR'
SWINIR_CHECKPOINT = SWINIR_REPO / 'checkpoints'

INPUT_DIR = DRIVE_BASE / 'input'

OUTPUT_DIR       = DRIVE_BASE / 'output'

SAM_BASE_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/092824"
SWINIR_BASE_URL = 'https://github.com/JingyunLiang/SwinIR/releases/download/v0.0'


sam2_checkpoints = {
    "tiny": f"sam2.1_hiera_tiny.pt",
    "small": f"sam2.1_hiera_small.pt",
    "large": f"sam2.1_hiera_large.pt",
}

swinir_checkpoints = {
    "real_sr": "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
    "lightweight": "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth",
}

sam2_cfgs = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml"
}


# ── Repo setup ─────────────────────────────────────────────────────────────────
def _run(cmd: str) -> bool:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[WARN] command failed: {cmd!r}\n{result.stderr[:300]}")
    return result.returncode == 0

def download_file(url, dest_path):
    dest_path = Path(dest_path) # make into path if not already
    print(f"Downloading {dest_path.name}...")
    try:
        cmd = f'wget --user-agent="Mozilla/5.0" -O "{dest_path}" "{url}"'
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print(f"Downloaded: {dest_path.name}")
        return True
    except Exception as e:
        print(f"Failed to download {dest_path.name}: {e}")
        return False

def check_checkpoints(model_name, checkpoint_base_path, checkpoint_base_url):
    checkpoint_path = Path(f"{checkpoint_base_path}/{model_name}")
    checkpoint_url = f"{checkpoint_base_url}/{model_name}"

    # Check if file exists AND is not empty
    if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        print(f"{model_name} already exists ({size_mb:.1f} MB)")
    else:
        # Remove empty file if it exists
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        print(f"{model_name} missing, downloading...")
        success = download_file(checkpoint_url, checkpoint_path)

        if success and checkpoint_path.stat().st_size > 0:
            size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
            print(f"  Size: {size_mb:.1f} MB")
        else:
            print(f"  Download failed or file is empty")
            if checkpoint_path.exists():
                checkpoint_path.unlink()  # Clean up failed download

def download_repos_and_setup() -> None:
    if not SAM_REPO.exists():
        print("Cloning segment-anything-2 (sam2)...")
        _run(f"git clone https://github.com/facebookresearch/sam2.git {SAM_REPO}")
    else:
        print(f"SAM2 repo already exists at {SAM_REPO}")

    # Install sam2 package
    print("Installing sam2 package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(SAM_REPO)])

    # Add repo to path so imports work

    if str(SAM_REPO) not in sys.path:
        sys.path.insert(0, str(SAM_REPO))

    # --- 2. Check and download SAM 2.1 checkpoints ---
    SAM_CHECKPOINT.mkdir(parents=True, exist_ok=True)

    # Check each checkpoint and download if missing
    print(f"\nChecking sam2 checkpoints in: {SAM_CHECKPOINT}")
    for model_name, model_type in sam2_checkpoints.items():
        check_checkpoints(model_type, SAM_CHECKPOINT, SAM_BASE_URL)

    print("\nAll sam2 checkpoints verified!")

    # LightGlue
    if not LIGHTGLUE_REPO.exists():
        print("Cloning LightGlue …")
        _run(f"git clone https://github.com/cvg/LightGlue.git {LIGHTGLUE_REPO}")

    if str(LIGHTGLUE_REPO) not in sys.path:
        sys.path.insert(0, str(LIGHTGLUE_REPO))

    print("\nInstalling LightGlue package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(LIGHTGLUE_REPO)])

    # SwinIR – no setup.py, just add to sys.path
    if not SWINIR_REPO.exists():
        print("\nCloning SwinIR …")
        _run(f"git clone https://github.com/JingyunLiang/SwinIR.git {SWINIR_REPO}")

    SWINIR_CHECKPOINT.mkdir(parents=True, exist_ok=True)
    print("SwinIR – no setup.py, just adding to sys.path...")
    if str(SWINIR_REPO) not in sys.path:
        sys.path.insert(0, str(SWINIR_REPO))

    # Check each checkpoint and download if missing
    print(f"\nChecking swinir checkpoints in: {SWINIR_CHECKPOINT}")
    for model_type, model_name in swinir_checkpoints.items():
        check_checkpoints(model_name, SWINIR_CHECKPOINT, SWINIR_BASE_URL)

    print("\nAll swinir checkpoints verified!")

    print("\nAll repos ready.")


# ── Model loaders ──────────────────────────────────────────────────────────────
def load_swinir_model(model_type: str = "real_sr"):
    """Load the lightweight or real_sr SwinIR x4 SR model."""
    import torch
    from models.network_swinir import SwinIR as net  # requires SWINIR_REPO on sys.path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert model_type in ("real_sr", "lightweight"), "only real_sr and lightweight model_type are allowed"
    if model_type == "lightweight":
        model = net(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="pixelshuffledirect",
            resi_connection="1conv",
        )
    elif model_type == "real_sr":
        # these setting are the bigger SR model
        model = net(
            upscale=4,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6,6,6,6,6,6,6,6,6],
            embed_dim=240,
            num_heads=[8,8,8,8,8,8,8,8,8],
            mlp_ratio=2,
            upsampler='nearest+conv',
            resi_connection='3conv'
        )

    checkpoint = SWINIR_CHECKPOINT / swinir_checkpoints[model_type]
    pretrained = torch.load(checkpoint, map_location="cpu")
    key = "params_ema" if "params_ema" in pretrained else "params"
    model.load_state_dict(pretrained[key], strict=True)
    model.eval().to(device)

    print(f"SwinIR loaded from {checkpoint.name}")
    return model

def load_sam2_model(model_type: str = "small", device: str = None):
    """Load SAM 2 Model by type - large, small, tiny and return (model, mask_generator)"""
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else "cpu"

    SAM_CHECKPOINT = SAM_REPO / 'checkpoints'
    model_checkpoint = sam2_checkpoints[model_type]
    model = f"{SAM_CHECKPOINT}/{model_checkpoint}"
    model_cfg = sam2_cfgs[model_type]
    sam2_model = build_sam2(model_cfg, model, device=device)

    predictor = SAM2ImagePredictor(sam2_model)

    print(f"SAM2 loaded from {model_checkpoint}")

    return sam2_model, predictor

def load_lightglue_models(filter_threshold: float = 0.05, depth_confidence=-1, width_confidence=-1):
    """Load SuperPoint + LightGlue from local checkpoints."""
    import torch
    from lightglue import LightGlue, SuperPoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SuperPoint with local weights
    extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)

    # LightGlue with local weights
    matcher = LightGlue(
        features="superpoint",
        depth_confidence=-1,
        width_confidence=-1,
        filter_threshold=filter_threshold,
    ).eval().to(device)

    print("SuperPoint + LightGlue loaded from local checkpoints")
    return extractor, matcher
