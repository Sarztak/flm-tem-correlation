# alignment_helpers.py
import numpy as np
from skimage import io
from scipy import ndimage
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

def load_image(path):
    img = io.imread(path)
    return img

def center_crop(img, crop_h=512, crop_w=512):
    """Return center crop of the input image."""
    h, w = img.shape[:2]
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    return img[start_h:start_h + crop_h, start_w:start_w + crop_w]

def detect_dots(img, sigma=2, size=10, threshold_percentile=90):
    smoothed = ndimage.gaussian_filter(img, sigma=sigma)
    local_max = ndimage.maximum_filter(smoothed, size=size)
    dots = (smoothed == local_max) & (smoothed > np.percentile(smoothed, threshold_percentile))
    return np.argwhere(dots)

def visualize_detected_dots(flm_path, tem_path, flm_thresh=90, tem_thresh=30):
    """Detect and visualize dots on FLM and TEM images."""
    
    # Load
    flm = io.imread(flm_path)
    if len(flm.shape) == 3: flm = flm.mean(axis=2).astype(np.float32)
    tem = io.imread(tem_path)
    if len(tem.shape) == 3: tem = tem.mean(axis=2).astype(np.float32)
    
    # Detect
    flm_dots = detect_dots(flm, threshold_percentile=flm_thresh)
    tem_dots = detect_dots(tem, threshold_percentile=tem_thresh)
    
    print(f"FLM dots: {len(flm_dots)}")
    print(f"TEM dots: {len(tem_dots)}")
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(flm, cmap='gray')
    axes[0].plot(flm_dots[:, 1], flm_dots[:, 0], 'r.', markersize=4, alpha=0.7)
    axes[0].set_title(f'FLM: {len(flm_dots)} dots')
    axes[0].axis('off')
    
    axes[1].imshow(tem, cmap='gray')
    axes[1].plot(tem_dots[:, 1], tem_dots[:, 0], 'r.', markersize=4, alpha=0.7)
    axes[1].set_title(f'TEM: {len(tem_dots)} dots')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('dot_detection.png', dpi=150, bbox_inches='tight')
    plt.close()
def phase_correlation(img1, img2):
    f1, f2 = np.fft.fft2(img1), np.fft.fft2(img2)
    corr = np.fft.ifft2((f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-10))
    corr = corr.real
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    shift = np.array(peak) - np.array(corr.shape)//2
    return -shift[0], -shift[1]

def apply_affine(img, rotation_deg, scale, tx, ty):
    h, w = img.shape
    center = np.array([h//2, w//2])
    angle_rad = np.deg2rad(rotation_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    matrix = scale * np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    offset = center - matrix @ center + np.array([ty, tx])
    return ndimage.affine_transform(img, matrix.T, offset=offset)

def mutual_information(img1, img2, bins=50):
    i1, i2 = img1.flatten(), img2.flatten()
    valid = np.isfinite(i1) & np.isfinite(i2)
    i1, i2 = i1[valid], i2[valid]
    hist, _, _ = np.histogram2d(i1, i2, bins=bins)
    pxy = hist / hist.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    nzs = pxy > 0
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / (px[:, None] * py[None, :])[nzs]))
    hx, hy = -np.sum(px * np.log(px + 1e-10)), -np.sum(py * np.log(py + 1e-10))
    nmi = 2 * mi / (hx + hy + 1e-10)
    return nmi

def optimize_alignment(flm, tem, init_params):
    def objective(params):
        r, s, tx, ty = params
        t = apply_affine(tem, r, s, tx, ty)
        h, w = min(flm.shape[0], t.shape[0]), min(flm.shape[1], t.shape[1])
        return -mutual_information(flm[:h, :w], t[:h, :w])
    
    result = minimize(objective, init_params, method='Nelder-Mead')
    return result.x

def visualize_result(flm, tem, r, s, tx, ty):
    t = apply_affine(tem, r, s, tx, ty)
    h, w = min(flm.shape[0], t.shape[0]), min(flm.shape[1], t.shape[1])
    flm_n = (flm - flm.min()) / (flm.max() - flm.min() + 1e-8)
    t_n   = (t   - t.min()   ) / (t.max()   - t.min()   + 1e-8)
    overlay = np.zeros((h, w, 3))
    overlay[:, :, 0] = flm_n[:h, :w]
    overlay[:, :, 1] = t_n[:h, :w]
    plt.imsave('alignment_result.png', overlay, dpi=150)

def create_edge_tem(path: Path, threshold1=100, threshold2=200) -> np.array:
    tem_img = load_image(path)

    # make into B&W image
    if len(tem_img.shape) == 3:
        tem_img = cv2.cvtColor(tem_img, cv2.COLOR_RGB2GRAY)

    # shrink the size by about 8x
    h, w = tem_img.shape 

    # downsample using INTER_AREA that averages all the pixels in the source region to the output pixel if shrinking by 8x then a block of 8x8 is mapped to a single pixel in the output pixel

    tem_downsampled = cv2.resize(
        tem_img,
        (w // 8, h // 8),
        interpolation=cv2.INTER_AREA,
    )

    # then find the edges using canny filter
    edges = cv2.Canny(
        tem_downsampled, threshold1=threshold1, threshold2=threshold2
    )

    # find the contours
    contours, _ = cv2.findContours(
        edges, 
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(tem_downsampled)
    mask_with_contour = cv2.drawContours(mask, [largest], -1, 255, thickness=1)
    return mask_with_contour


