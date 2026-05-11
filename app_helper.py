import json
import torch
import numpy as np
import cv2
import gc
from pathlib import Path
from skimage import io
from magicgui import magic_factory
from magicgui.widgets import ComboBox, PushButton, Container
import napari
from napari.utils.notifications import show_info
import networkx as nx
from collections import defaultdict
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import minimize
from skimage import exposure, io, measure
from skimage.filters import threshold_otsu
from qtpy.QtWidgets import QLabel, QWidget, QVBoxLayout
import sys 

def get_tile_flm_bbox(flm_height: int, flm_width: int, 
                      tem_height: int, tem_width: int,
                      flm_pixel_nm: float = 121.0, 
                      tem_pixel_nm: float = 6.9, 
                      tile_scale: float = 1.0) -> list[list[int]]:
    
    tile_h = int((tem_height * tem_pixel_nm) / flm_pixel_nm * tile_scale)
    tile_w = int((tem_width * tem_pixel_nm) / flm_pixel_nm * tile_scale)
    step_y, step_x = tile_h // 2, tile_w // 2

    tiles = []
    for y in range(0, flm_height, step_y):
        for x in range(0, flm_width, step_x):
            if y + tile_h > flm_height or x + tile_w > flm_width:
                continue
            
            # Append the bounding box as a list of coordinates
            tiles.append([x, y, x + tile_w, y + tile_h])
    
    if not tiles:  # in case no tile is found
        tiles.append([0, 0, tile_w, tile_h])
    
    return tiles

def has_interior_peak(group_rois):
    """Returns True if the Laplacian peaks at an interior frame, not at the edges."""
    sorted_by_frame = sorted(group_rois, key=lambda r: r["frame_idx"])
    values = [r["laplacian"] for r in sorted_by_frame]
    peak_idx = np.argmax(values)
    # peak must not be at the first or last frame
    return 0 < peak_idx < len(values) - 1

def merge_bboxes(bboxes, tem_height_flm: float, tem_width_flm: float, flm_img_h, flm_img_w) -> list:
    merge_dist = int(max(tem_height_flm, tem_width_flm))
    expanded = [
        (max(0, r0 - merge_dist), max(0, c0 - merge_dist), min(r1 + merge_dist, flm_img_h), min(c1 + merge_dist, flm_img_w))
        for r0, c0, r1, c1 in bboxes
    ]
    expanded.sort(key=lambda x: x[0])
    merged = [expanded[0]]
    for r0, c0, r1, c1 in expanded[1:]:
        pr0, pc0, pr1, pc1 = merged[-1]
        if r0 <= pr1 and c0 <= pc1:
            merged[-1] = (min(pr0, r0), min(pc0, c0), max(pr1, r1), max(pc1, c1))
        else:
            merged.append([r0, c0, r1, c1])
    return merged

def find_roi_with_origins(
    img_flm: np.ndarray,
    tem_h: int,
    tem_w: int,
    flm_pixel_nm: float = 121.0,
    tem_pixel_nm: float = 6.9,
    pad_factor: int = 2,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """
    Like find_roi_bl_gr but also returns the (col_offset, row_offset) of each
    crop in the full FLM image — needed for back-projection later.

    Returns
    -------
    crops   : list of 2-D grayscale uint8 crops (reflection channel)
    origins : list of (col_start, row_start) tuples in full-image pixel coords
    """
    tem_height_flm = (tem_h * tem_pixel_nm) / flm_pixel_nm
    tem_width_flm  = (tem_w * tem_pixel_nm) / flm_pixel_nm

    pad_y = int(tem_height_flm) * pad_factor
    pad_x = int(tem_width_flm)  * pad_factor

    green = img_flm[:, :, 0].astype(float)
    blue  = img_flm[:, :, 2].astype(float)
    bl_gr = green + blue

    thresh   = threshold_otsu(bl_gr)
    roi_mask = (bl_gr > thresh).astype(int)

    labelled = measure.label(roi_mask, connectivity=2)
    props    = measure.regionprops(labelled)
    bboxes   = [p.bbox for p in props]
    merged   = merge_bboxes(bboxes, tem_height_flm, tem_width_flm)

    crops, origins = [], []
    for b in merged:
        min_row, min_col, max_row, max_col = b
        r0 = max(0, min_row - pad_y)
        c0 = max(0, min_col - pad_x)
        r1 = min(img_flm.shape[0], max_row + pad_y)
        c1 = min(img_flm.shape[1], max_col + pad_x)
        # reflection channel = index 1
        crop = img_flm[r0:r1, c0:c1, 1]
        crops.append(crop)
        origins.append((c0, r0))   # (x_offset, y_offset) in full image

    return crops, origins

def get_all_rois(flm_stack, tem_h, tem_w, flm_pixel_nm, tem_pixel_nm, pad_factor=2):
    # collect all ROIs across all frames with their bounding boxes and laplacian
    all_rois = []  # list of {frame_idx, origin, bbox, laplacian}

    for frame_idx in range(flm_stack.shape[0]):
        flm_frame = flm_stack[frame_idx]
        crops, origins = find_roi_with_origins(flm_frame, tem_h, tem_w, flm_pixel_nm, tem_pixel_nm, pad_factor=pad_factor)

        for crop, (ox, oy) in zip(crops, origins):
            if crop.max() == crop.min():
                continue
            h, w = crop.shape[:2]
            crop_u8 = ((crop - crop.min()) / (crop.max() - crop.min()) * 255).astype(np.uint8)
            lap = cv2.Laplacian(crop_u8, cv2.CV_64F).var()
            mean_intensity = crop.mean()
            all_rois.append({
                "frame_idx": frame_idx,
                "origin":    (ox, oy),
                "bbox":      (ox, oy, ox + w, oy + h),  # (x0, y0, x1, y1)
                "laplacian": lap,
                "mean_intensity": mean_intensity,
                "area":      w * h,
            })

    del flm_stack
    gc.collect()

    return all_rois


def iou(b1, b2):
    ix0 = max(b1[0], b2[0])
    iy0 = max(b1[1], b2[1])
    ix1 = min(b1[2], b2[2])
    iy1 = min(b1[3], b2[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    a1    = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2    = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)

def find_best_roi_frame_idx(all_rois, iou_threshold=0.3):
    # group by spatial overlap using IoU
    
    groups = []  # list of lists of roi indices

    G = nx.Graph()
    G.add_nodes_from(range(len(all_rois)))

    for i in range(len(all_rois)):
        for j in range(i + 1, len(all_rois)):
            if iou(all_rois[i]["bbox"], all_rois[j]["bbox"]) >= iou_threshold:
                G.add_edge(i, j)

    groups = list(nx.connected_components(G))

    best_per_group = []
    for group in groups:
        best_idx = max(group, key=lambda i: all_rois[i]["laplacian"] * all_rois[i]["mean_intensity"])
        best_roi = all_rois[best_idx]
        best_per_group.append(best_roi)

    # filter groups: remove those images where the peak does not occur in the middle
    valid_best_per_group = []
    for group, best_roi in zip(groups, best_per_group):
        group_rois = [all_rois[i] for i in group]
        if has_interior_peak(group_rois):
            valid_best_per_group.append(best_roi)
            print(f"Valid ROI: best frame={best_roi['frame_idx']}  laplacian={best_roi['laplacian']:.2f}  bbox={best_roi['bbox']}")
        else:
            print(f"Rejected ROI (no interior peak): bbox={best_roi['bbox']}")

    frames_to_process = sorted(set(r["frame_idx"] for r in valid_best_per_group))
    print(f"\nFrames to process: {frames_to_process}")
    return valid_best_per_group

def norm(img):
    diff = img.max() - img.min()
    if diff == 0: return np.zeros_like(img, dtype=np.uint8)
    return ((img - img.min()) / diff * 255).astype(np.uint8)

def render_flm_frame(flm_frame):
    h, w = flm_frame.shape[:2] # h, w, c
    composite = np.full((h, w, 3), 0, dtype=np.uint8)

    refl = flm_frame[:, :, 1]
    green = flm_frame[:, :, 0]
    blue = flm_frame[:, :, 2]

    refl_u8 = norm(refl)
    blue_u8 = norm(blue)
    green_u8 = norm(green)

    composite[:, :, 2] = np.clip(refl_u8.astype(int) + blue_u8, 0, 255)  # Blue channel
    composite[:, :, 1] = np.clip(refl_u8.astype(int) + green_u8, 0, 255) # Green channel
    composite[:, :, 0] = refl_u8

    return composite

def get_tile_flm_bbox_with_pad(flm_height: int, flm_width: int, 
                      tem_height: int, tem_width: int,
                      flm_pixel_nm: float = 121.0, 
                      tem_pixel_nm: float = 6.9, 
                      tile_scale: float = 1.0) -> tuple[list[list[int]], tuple[int,int,int,int]]:
    """
    Returns (tiles, padding) where:
      - tiles: list of [y0, x0, y1, x1] covering the padded image
      - padding: (pad_top, pad_bottom, pad_left, pad_right) to apply to the original image
    """
    tile_h = int((tem_height * tem_pixel_nm) / flm_pixel_nm * tile_scale)
    tile_w = int((tem_width * tem_pixel_nm) / flm_pixel_nm * tile_scale)
    step_y, step_x = tile_h // 2 or 1, tile_w // 2 or 1  # avoid zero step

    # compute padding needed so that (flm_dim + pad) - tile_dim is divisible by step
    def compute_pad(dim, tile, step):
        if dim <= tile:
            # need at least tile size; put all padding to bottom/right
            pad_top = 0
            pad_bottom = tile - dim
            return pad_top, pad_bottom
        rem = (dim - tile) % step
        pad_top = 0
        pad_bottom = (step - rem) if rem != 0 else 0
        return pad_top, pad_bottom

    pad_top, pad_bottom = compute_pad(flm_height, tile_h, step_y)
    pad_left, pad_right = compute_pad(flm_width, tile_w, step_x)

    padded_h = flm_height + pad_top + pad_bottom
    padded_w = flm_width + pad_left + pad_right

    tiles = []
    for y in range(0, padded_h - tile_h + 1, step_y):
        for x in range(0, padded_w - tile_w + 1, step_x):
            y0 = y
            x0 = x
            y1 = y + tile_h
            x1 = x + tile_w
            tiles.append([y0, x0, y1, x1])

    # fallback (shouldn't be needed)
    if not tiles:
        tiles.append([0, 0, tile_h, tile_w])

    padding = (pad_top, pad_bottom, pad_left, pad_right)
    return tiles, padding