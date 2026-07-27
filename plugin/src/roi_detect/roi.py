import gc
import math
import numpy as np
import cv2
import networkx as nx
from pathlib import Path
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from skimage import measure, exposure
from skimage.filters import threshold_otsu


# ── Image normalization ────────────────────────────────────────────────────────

def norm(img):
    diff = img.max() - img.min()
    if diff == 0:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - img.min()) / diff * 255).astype(np.uint8)


# ── FLM frame rendering ────────────────────────────────────────────────────────

def render_flm_frame(flm_frame):
    h, w = flm_frame.shape[:2]
    composite = np.full((h, w, 3), 0, dtype=np.uint8)
    refl  = flm_frame[:, :, 1]
    green = flm_frame[:, :, 0]
    blue  = flm_frame[:, :, 2]
    refl_u8  = norm(refl)
    blue_u8  = norm(blue)
    green_u8 = norm(green)
    composite[:, :, 2] = np.clip(refl_u8.astype(int) + blue_u8,  0, 255)
    composite[:, :, 1] = np.clip(refl_u8.astype(int) + green_u8, 0, 255)
    composite[:, :, 0] = refl_u8
    return composite


def render_all_frames(flm_stack, render_fn):
    rendered = [np.asarray(render_fn(flm_stack[i])) for i in range(flm_stack.shape[0])]
    return np.array(rendered)


def equalize_flm_frame(flm_frame):
    flm_refl_uint8 = norm(flm_frame)
    flm_img_hist_eq = exposure.equalize_hist(flm_refl_uint8)
    return (flm_img_hist_eq * 255).astype(np.uint8)


# ── Bounding box helpers ───────────────────────────────────────────────────────

def _to_xy(bbox):
    y0, x0, y1, x1 = bbox
    return (x0, y0, x1, y1)


def iou(b1, b2):
    ix0 = max(b1[0], b2[0])
    iy0 = max(b1[1], b2[1])
    ix1 = min(b1[2], b2[2])
    iy1 = min(b1[3], b2[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter)


def merge_bboxes(bboxes, tem_height_flm: float, tem_width_flm: float, flm_img_h, flm_img_w) -> list:
    if not bboxes:
        return []
    merge_dist = int(max(tem_height_flm, tem_width_flm))
    expanded = [
        (max(0, r0 - merge_dist), max(0, c0 - merge_dist),
         min(r1 + merge_dist, flm_img_h), min(c1 + merge_dist, flm_img_w))
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


def filter_and_merge_bboxes(bboxes, tem_height_flm, tem_width_flm, flm_h, flm_w):
    min_area_thresh = tem_height_flm * tem_width_flm
    filtered = [
        bbox for bbox in bboxes
        if abs(bbox[2] - bbox[0]) * abs(bbox[3] - bbox[1]) >= min_area_thresh
    ]
    return merge_bboxes(filtered, tem_height_flm, tem_width_flm, flm_h, flm_w)


def group_bboxes_by_iou(bboxes, iou_threshold=0.3):
    G = nx.Graph()
    G.add_nodes_from(range(len(bboxes)))
    bx_xy = [_to_xy(b) for b in bboxes]
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if iou(bx_xy[i], bx_xy[j]) >= iou_threshold:
                G.add_edge(i, j)
    return list(sorted(list(c)) for c in nx.connected_components(G))


# ── ROI mask and detection ─────────────────────────────────────────────────────

def get_roi_mask(img_flm):
    green = img_flm[:, :, 0].astype(float)
    blue  = img_flm[:, :, 2].astype(float)
    bl_gr = green + blue
    thresh = threshold_otsu(bl_gr)
    return (bl_gr > thresh).astype(int)


def get_bbox_from_roi_mask(roi_mask, connectivity=2):
    labelled = measure.label(roi_mask, connectivity=connectivity)
    props = measure.regionprops(labelled)
    return [p.bbox for p in props]


def find_roi_with_origins(
    img_flm: np.ndarray,
    tem_h: int,
    tem_w: int,
    flm_pixel_nm: float = 121.0,
    tem_pixel_nm: float = 6.9,
    pad_factor: int = 2,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    tem_height_flm = (tem_h * tem_pixel_nm) / flm_pixel_nm
    tem_width_flm  = (tem_w * tem_pixel_nm) / flm_pixel_nm
    min_area_thresh = tem_height_flm * tem_width_flm

    green = img_flm[:, :, 0].astype(float)
    blue  = img_flm[:, :, 2].astype(float)
    bl_gr = green + blue

    thresh   = threshold_otsu(bl_gr)
    roi_mask = (bl_gr > thresh).astype(int)

    labelled = measure.label(roi_mask, connectivity=2)
    props    = measure.regionprops(labelled)

    bboxes = []
    for p in props:
        y0, x0, y1, x1 = p.bbox
        if abs(y1 - y0) * abs(x1 - x0) >= min_area_thresh:
            bboxes.append(p.bbox)

    merged = merge_bboxes(bboxes, tem_height_flm, tem_width_flm, img_flm.shape[0], img_flm.shape[1])

    crops, origins = [], []
    for b in merged:
        min_row, min_col, max_row, max_col = b
        crop = img_flm[min_row:max_row, min_col:max_col, 1]
        crops.append(crop)
        origins.append((min_col, min_row))

    return crops, origins


def get_all_rois(flm_stack, tem_h, tem_w, flm_pixel_nm, tem_pixel_nm, pad_factor=2):
    all_rois = []
    for frame_idx in range(flm_stack.shape[0]):
        flm_frame = flm_stack[frame_idx]
        crops, origins = find_roi_with_origins(
            flm_frame, tem_h, tem_w, flm_pixel_nm, tem_pixel_nm, pad_factor=pad_factor
        )
        for crop, (ox, oy) in zip(crops, origins):
            if crop.max() == crop.min():
                continue
            h, w = crop.shape[:2]
            crop_u8 = ((crop - crop.min()) / (crop.max() - crop.min()) * 255).astype(np.uint8)
            lap = cv2.Laplacian(crop_u8, cv2.CV_64F).var()
            all_rois.append({
                "frame_idx":       frame_idx,
                "origin":          (ox, oy),
                "bbox":            (ox, oy, ox + w, oy + h),
                "laplacian":       lap,
                "mean_intensity":  crop.mean(),
                "area":            w * h,
            })
    del flm_stack
    gc.collect()
    return all_rois


# ── Laplacian / z-frame scoring ────────────────────────────────────────────────

def get_laplacian_per_bbox(flm_stack, bboxes):
    laps_per_bbox = []
    flm_refl = flm_stack[:, :, :, 1]
    for bbox in bboxes:
        y0, x0, y1, x1 = bbox
        crops = flm_refl[:, y0:y1, x0:x1]
        mins = crops.min(axis=(1, 2), keepdims=True)
        maxs = crops.max(axis=(1, 2), keepdims=True)
        crops_u8 = ((crops - mins) / (maxs - mins + 1e-8) * 255).astype(np.uint8)
        laps = np.array([cv2.Laplacian(crops_u8[z], cv2.CV_64F).var() for z in range(crops_u8.shape[0])])
        laps_per_bbox.append(laps)
    return laps_per_bbox


def has_interior_peak(group_rois):
    sorted_by_frame = sorted(group_rois, key=lambda r: r["frame_idx"])
    values = [r["laplacian"] for r in sorted_by_frame]
    peak_idx = np.argmax(values)
    return 0 < peak_idx < len(values) - 1


def find_flat_peak(laps, smooth_sigma=2, min_prominence=10):
    smoothed = gaussian_filter1d(laps.astype(float), sigma=smooth_sigma)
    peaks, props = find_peaks(smoothed, prominence=min_prominence, width=2)
    if len(peaks) == 0:
        return None, smoothed
    best = peaks[np.argmax(props["widths"])]
    return int(best), smoothed


# ── Tiling ─────────────────────────────────────────────────────────────────────

def get_tile_flm_bbox(flm_height, flm_width, tem_height, tem_width,
                      flm_pixel_nm=121.0, tem_pixel_nm=6.9, tile_scale=1.0):
    tile_h = int((tem_height * tem_pixel_nm) / flm_pixel_nm * tile_scale)
    tile_w = int((tem_width  * tem_pixel_nm) / flm_pixel_nm * tile_scale)
    step_y, step_x = tile_h // 2, tile_w // 2
    tiles = []
    for y in range(0, flm_height, step_y):
        for x in range(0, flm_width, step_x):
            if y + tile_h > flm_height or x + tile_w > flm_width:
                continue
            tiles.append([x, y, x + tile_w, y + tile_h])
    if not tiles:
        tiles.append([0, 0, tile_w, tile_h])
    return tiles


def get_tile_flm_bbox_with_pad(flm_height, flm_width, tem_height, tem_width,
                                flm_pixel_nm=121.0, tem_pixel_nm=6.9, tile_scale=1.0):
    tile_h = int((tem_height * tem_pixel_nm) / flm_pixel_nm * tile_scale)
    tile_w = int((tem_width  * tem_pixel_nm) / flm_pixel_nm * tile_scale)
    step_y = tile_h // 2 or 1
    step_x = tile_w // 2 or 1

    def compute_pad(dim, tile, step):
        if dim <= tile:
            return 0, tile - dim
        rem = (dim - tile) % step
        return 0, (step - rem) if rem != 0 else 0

    pad_top, pad_bottom = compute_pad(flm_height, tile_h, step_y)
    pad_left, pad_right = compute_pad(flm_width,  tile_w, step_x)
    padded_h = flm_height + pad_top + pad_bottom
    padded_w = flm_width  + pad_left + pad_right

    tiles = []
    for y in range(0, padded_h - tile_h + 1, step_y):
        for x in range(0, padded_w - tile_w + 1, step_x):
            tiles.append([y, x, y + tile_h, x + tile_w])
    if not tiles:
        tiles.append([0, 0, tile_h, tile_w])

    return tiles, (pad_top, pad_bottom, pad_left, pad_right)


def tiles_per_best_frame(bbox_best_frame, global_bboxes, flm_h, flm_w,
                          tem_h, tem_w, flm_pixel_nm, tem_pixel_nm, tile_scale=1.5):
    tiles_and_frame = []
    for b in bbox_best_frame:
        best_bbox_pts = global_bboxes[b["bbox_idx"]]
        bb_y0, bb_x0, bb_y1, bb_x1 = best_bbox_pts
        bbox_h = bb_y1 - bb_y0
        bbox_w = bb_x1 - bb_x0
        tiles_bbox, _ = get_tile_flm_bbox_with_pad(
            flm_height=bbox_h, flm_width=bbox_w,
            tem_height=tem_h, tem_width=tem_w,
            flm_pixel_nm=flm_pixel_nm, tem_pixel_nm=tem_pixel_nm,
            tile_scale=tile_scale,
        )
        tiles_in_ref = [(y0 + bb_y0, x0 + bb_x0, y1 + bb_y0, x1 + bb_x0) for y0, x0, y1, x1 in tiles_bbox]
        tiles_and_frame.append({"frame_idx": b["frame_idx"], "bboxes": tiles_in_ref})
    return tiles_and_frame


# ── TEM preprocessing ──────────────────────────────────────────────────────────

def prepare_tem(tem_img, thresh=130):
    if tem_img.ndim == 3:
        tem_img = tem_img[:, :, 0]
    img_u8 = ((tem_img.astype(float) - tem_img.min()) / (tem_img.max() - tem_img.min() + 1e-8) * 255).astype(np.uint8)
    inverted = 255 - img_u8
    thresh_img = (inverted > thresh).astype(np.uint8) * 255
    thresh_img = np.stack([thresh_img] * 3, axis=-1)
    return img_u8, thresh_img
