import subprocess
import sys
from pathlib import Path
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DIR = Path(r"C:\Users\sar31\Documents\GitHub\flm_tem_alignment")
SWINIR_REPO = DEFAULT_DIR / 'SwinIR'
SWINIR_MODEL_ZOO = SWINIR_REPO / 'checkpoints'
SAM_REPO = DEFAULT_DIR / 'segment_anything_2'
LIGHTGLUE_REPO = DEFAULT_DIR / 'LightGlue'

SWINIR_MODEL = SWINIR_MODEL_ZOO / "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth"

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

# SWINIR_MODEL = SWINIR_MODEL_ZOO / '003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth'     # SwinIR – no setup.py, just add to sys.path
sys.path.insert(0, str(SWINIR_REPO))
sys.path.insert(0, str(SAM_REPO))
sys.path.insert(0, str(LIGHTGLUE_REPO))

def load_swinir_model(checkpoint: Path = SWINIR_MODEL):
    """Load the lightweight SwinIR x4 SR model."""
    import torch
    # breakpoint()
    from models.network_swinir import SwinIR as net  # requires SWINIR_REPO on sys.path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # these setting are for the light weight model
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

    # # these setting are the bigger SR model 
    # model = net(
    #     upscale=4, in_chans=3, img_size=64, window_size=8,
    #     img_range=1., depths=[6,6,6,6,6,6,6,6,6], embed_dim=240,
    #     num_heads=[8,8,8,8,8,8,8,8,8],
    #     mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv'
    # )

    pretrained = torch.load(checkpoint, map_location="cpu")
    key = "params_ema" if "params_ema" in pretrained else "params"
    model.load_state_dict(pretrained[key], strict=True)
    model.eval().to(device)
    print(f"✓ SwinIR loaded from {checkpoint.name}")
    return model

# ── SwinIR batch upscale ───────────────────────────────────────────────────────
def swinir_upscale(
    model,
    images: list[np.ndarray],
    window_size: int = 8,
    max_batch_size: int = 4,
) -> list[np.ndarray | None]:
    """
    Upscale a list of BGR uint8 numpy arrays with SwinIR x4.

    Returns a list of the same length; entries are BGR uint8 at 4× resolution,
    or None for any input that was None.
    """

    device = next(model.parameters()).device
    results: list[np.ndarray | None] = [None] * len(images)
    valid_items = [(i, img) for i, img in enumerate(images) if img is not None]

    for start in range(0, len(valid_items), max_batch_size):
        chunk = valid_items[start : start + max_batch_size]
        indices, imgs = zip(*chunk)

        batch = torch.stack(
            [
                torch.from_numpy(img[:, :, [2, 1, 0]]).permute(2, 0, 1).float() / 255.0
                for img in imgs
            ]
        ).to(device)

        _, _, h, w = batch.shape
        h_pad = (h // window_size + 1) * window_size - h
        w_pad = (w // window_size + 1) * window_size - w
        batch = torch.cat([batch, torch.flip(batch, [2])], 2)[:, :, : h + h_pad, :]
        batch = torch.cat([batch, torch.flip(batch, [3])], 3)[:, :, :, : w + w_pad]

        with torch.no_grad():
            output = model(batch)

        output = output[:, :, : h * 4, : w * 4].clamp(0, 1).cpu()

        for local_i, orig_idx in enumerate(indices):
            out_np = output[local_i].permute(1, 2, 0).numpy()
            out_np = out_np[:, :, [2, 1, 0]]  # RGB → BGR
            results[orig_idx] = (out_np * 255).round().astype(np.uint8)

        del batch, output
        import torch as _torch
        _torch.cuda.empty_cache()

    return results

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

if __name__ == "__main__":
    ...
    # load_lightglue_models()
    # ff_bb_path = DEFAULT_DIR/ 'output' / 'filtered_bbox'
    # img_arr = []
    # for img_path in ff_bb_path.glob("*.png"):
    #     img = cv2.imread(img_path, 0) # open as gray scale
    #     if len(img.shape) == 2:
    #         img = np.stack([img] * 3, axis=-1) # need to create RGB since swinir needs to have 3 channel images
    #     img_arr.append(img)
    # swinir_model = load_swinir_model()
    # upscaled = swinir_upscale(
    #     swinir_model,
    #     img_arr,
    #     window_size=8,
    #     max_batch_size=4
    # )

    # ff_bb_upscaled = DEFAULT_DIR / 'output' / 'ff_bb_upscaled'
    # ff_bb_upscaled.mkdir(exist_ok=True)

    # for i, img in enumerate(upscaled):
    #     cv2.imwrite(ff_bb_upscaled / f'{str(i).zfill(3)}.png', img)