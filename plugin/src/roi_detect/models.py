import sys
from pathlib import Path
import torch
import cv2
import numpy as np

DEFAULT_DIR = Path(__file__).resolve().parent.parent.parent.parent  # repo root
SWINIR_REPO = DEFAULT_DIR / 'SwinIR'
SWINIR_MODEL_ZOO = SWINIR_REPO / 'checkpoints'
SAM_REPO = DEFAULT_DIR / 'segment_anything_2'
LIGHTGLUE_REPO = DEFAULT_DIR / 'LightGlue'

SWINIR_MODEL = SWINIR_MODEL_ZOO / "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth"

sam2_checkpoints = {
    "tiny": "sam2.1_hiera_tiny.pt",
    "small": "sam2.1_hiera_small.pt",
    "large": "sam2.1_hiera_large.pt",
}

swinir_checkpoints = {
    "real_sr": "003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth",
    "lightweight": "002_lightweightSR_DIV2K_s64w8_SwinIR-S_x4.pth",
}

sam2_cfgs = {
    "tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
    "small": "configs/sam2.1/sam2.1_hiera_s.yaml",
    "large": "configs/sam2.1/sam2.1_hiera_l.yaml",
}

sys.path.insert(0, str(SWINIR_REPO))
sys.path.insert(0, str(SAM_REPO))
sys.path.insert(0, str(LIGHTGLUE_REPO))


def load_swinir_model(checkpoint: Path = SWINIR_MODEL):
    from models.network_swinir import SwinIR as net
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net(
        upscale=4, in_chans=3, img_size=64, window_size=8, img_range=1.0,
        depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
        mlp_ratio=2, upsampler="pixelshuffledirect", resi_connection="1conv",
    )
    pretrained = torch.load(checkpoint, map_location="cpu")
    key = "params_ema" if "params_ema" in pretrained else "params"
    model.load_state_dict(pretrained[key], strict=True)
    model.eval().to(device)
    print(f"✓ SwinIR loaded from {checkpoint.name}")
    return model


def swinir_upscale(
    model,
    images: list[np.ndarray],
    window_size: int = 8,
    max_batch_size: int = 4,
) -> list[np.ndarray | None]:
    device = next(model.parameters()).device
    results: list[np.ndarray | None] = [None] * len(images)
    valid_items = [(i, img) for i, img in enumerate(images) if img is not None]

    for start in range(0, len(valid_items), max_batch_size):
        chunk = valid_items[start: start + max_batch_size]
        indices, imgs = zip(*chunk)

        batch = torch.stack([
            torch.from_numpy(img[:, :, [2, 1, 0]]).permute(2, 0, 1).float() / 255.0
            for img in imgs
        ]).to(device)

        _, _, h, w = batch.shape
        h_pad = (h // window_size + 1) * window_size - h
        w_pad = (w // window_size + 1) * window_size - w
        batch = torch.cat([batch, torch.flip(batch, [2])], 2)[:, :, :h + h_pad, :]
        batch = torch.cat([batch, torch.flip(batch, [3])], 3)[:, :, :, :w + w_pad]

        with torch.no_grad():
            output = model(batch)

        output = output[:, :, :h * 4, :w * 4].clamp(0, 1).cpu()

        for local_i, orig_idx in enumerate(indices):
            out_np = output[local_i].permute(1, 2, 0).numpy()
            out_np = out_np[:, :, [2, 1, 0]]
            results[orig_idx] = (out_np * 255).round().astype(np.uint8)

        del batch, output
        torch.cuda.empty_cache()

    return results


def load_sam2_model(model_type: str = "small", device: str = None):
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else "cpu"

    checkpoint = f"{SAM_REPO / 'checkpoints'}/{sam2_checkpoints[model_type]}"
    sam2_model = build_sam2(sam2_cfgs[model_type], checkpoint, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    print(f"✓ SAM2 loaded from {sam2_checkpoints[model_type]}")
    return sam2_model, predictor


def load_sam2_auto_model(model_type: str = "small", device: str = None):
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else "cpu"

    checkpoint = f"{SAM_REPO / 'checkpoints'}/{sam2_checkpoints[model_type]}"
    sam2_model = build_sam2(sam2_cfgs[model_type], checkpoint, device=device)
    print(f"✓ SAM2 auto mask generator loaded from {sam2_checkpoints[model_type]}")
    return sam2_model


def load_lightglue_models(filter_threshold: float = 0.05, depth_confidence=-1, width_confidence=-1):
    from lightglue import LightGlue, SuperPoint

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
    matcher = LightGlue(
        features="superpoint",
        depth_confidence=depth_confidence,
        width_confidence=width_confidence,
        filter_threshold=filter_threshold,
    ).eval().to(device)
    print("✓ SuperPoint + LightGlue loaded")
    return extractor, matcher


def upscale_and_save(input_dir: Path | str, output_dir: Path | str, window_size: int = 8, max_batch_size: int = 4):
    input_dir = Path(input_dir)
    img_arr = []
    for img_path in input_dir.glob("*.png"):
        img = cv2.imread(str(img_path), 0)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)
        img_arr.append(img)
    swinir_model = load_swinir_model()
    upscaled = swinir_upscale(swinir_model, img_arr, window_size=window_size, max_batch_size=max_batch_size)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    for i, img in enumerate(upscaled):
        cv2.imwrite(str(output_dir / f'{str(i).zfill(4)}.png'), img)


def create_tensor_from_mask(mask, device):
    if mask.ndim == 2:
        mask = np.stack([mask, mask, mask], axis=-1)
    image = np.transpose(mask, (2, 0, 1))
    return torch.from_numpy(image).float().unsqueeze(0).to(device)


def get_keypoint_matches(extractor, matcher, image0, image1, conf_thresh):
    from lightglue.utils import rbd
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)
    matcher.conf.filter_threshold = conf_thresh
    matches01 = matcher({"image0": feats0, "image1": feats1})
    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    return kpts0, kpts1, kpts0[matches[..., 0]], kpts1[matches[..., 1]], matches01


def estimate_transform(kpts0, kpts1):
    M, inliers = cv2.estimateAffinePartial2D(
        kpts0.astype(np.float32),
        kpts1.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )
    if M is None:
        return None, None, 0
    scale = np.sqrt(M[0, 0] ** 2 + M[0, 1] ** 2)
    return M, inliers, scale


def apply_transform_overlay(img_source, img_target, M):
    h, w = img_target.shape[:2]
    warped = cv2.warpAffine(img_source, M, (w, h), flags=cv2.INTER_LINEAR)

    def _norm(img):
        if img.max() > 0:
            return (img.astype(float) / img.max() * 255).astype(np.uint8)
        return img.astype(np.uint8)

    target_norm = _norm(img_target)
    warped_norm = _norm(warped)

    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 0] = target_norm
    overlay[:, :, 1] = warped_norm
    overlay[:, :, 1] = np.clip(overlay[:, :, 1].astype(int) + warped_norm.astype(int), 0, 255).astype(np.uint8)

    return overlay, warped, target_norm, warped_norm
