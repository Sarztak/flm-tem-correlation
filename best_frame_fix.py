import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from skimage import measure
from skimage.filters import threshold_otsu
from scipy.signal import savgol_filter, find_peaks
from scipy.ndimage import gaussian_filter1d
import numpy as np
import math
import matplotlib.pyplot as plt
import gc
from app_helper import threshold_otsu, iou
import networkx as nx
from skimage import measure
import cv2
from app_helper import merge_bboxes

def _to_xy(bbox):
    # regionprops bbox: (min_row, min_col, max_row, max_col) -> (x0,y0,x1,y1)
    y0, x0, y1, x1 = bbox
    return (x0, y0, x1, y1)


def render_all_frames(flm_stack, render_flm_frame):
    """
    Render every frame in flm_stack using render_flm_frame.
    flm_stack: ndarray (z, h, w, c)
    render_flm_frame: callable that accepts flm_frame (h,w,c) and returns an image (H,W,C) or (H,W)
    Returns: list of rendered frames (as numpy arrays) in same order as z
    """
    rendered = []
    for idx in range(flm_stack.shape[0]):
        frame = flm_stack[idx]
        rendered_frame = render_flm_frame(frame)
        rendered.append(np.asarray(rendered_frame))
    return np.array(rendered)

def crops_for_group(all_rois, group, rendered_stack):
    """
    all_rois: list of dicts with keys 'frame_idx' and 'bbox' where bbox=(x0,y0,x1,y1)
    group: iterable of indices into all_rois (set/list)
    rendered_stack: list/array of rendered frames (indexable by frame_idx)
    Returns list of cropped images (in order of appearance in group)
    """
    crops = []
    for roi_idx in sorted(group):
        roi = all_rois[roi_idx]
        fi = int(roi["frame_idx"])
        x0, y0, x1, y1 = roi["bbox"]
        # ensure integers and clamp to image bounds
        x0, y0, x1, y1 = map(int, (x0, y0, x1, y1))
        rendered = rendered_stack[fi]
        h, w = rendered.shape[:2]
        x0 = max(0, min(x0, w-1))
        x1 = max(0, min(x1, w))
        y0 = max(0, min(y0, h-1))
        y1 = max(0, min(y1, h))
        if x1 <= x0 or y1 <= y0:
            continue
        crop = rendered[y0:y1, x0:x1].copy()
        crops.append((roi_idx, fi, crop))
    return crops

def plot_group_crops(crops, max_cols=4, figsize_per=(3,3), cmap=None):
    """
    crops: list of tuples (roi_idx, frame_idx, crop_array)
    Shows all crops for the group in a grid. Titles show roi and frame.
    """
    n = len(crops)
    if n == 0:
        return
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*figsize_per[0], rows*figsize_per[1]))
    axes = np.array(axes).reshape(-1)
    for ax in axes[n:]:
        ax.axis('off')
    for i, (roi_idx, fi, crop) in enumerate(crops):
        ax = axes[i]
        if crop.ndim == 3 and crop.shape[2] == 4:
            # if RGBA, show as-is
            ax.imshow(crop)
        elif crop.ndim == 3 and crop.shape[2] in (1,3):
            ax.imshow(crop.squeeze())
        else:
            ax.imshow(crop, cmap=cmap)
        ax.set_title(f'roi {roi_idx}\nframe {fi}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()

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

    pad_y = int(tem_height_flm) * pad_factor
    pad_x = int(tem_width_flm)  * pad_factor

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
        height = abs(y1 - y0)
        width = abs(x1 - x0)
        if height * width >= min_area_thresh:
            bboxes.append(p.bbox)

    merged   = merge_bboxes(bboxes, tem_height_flm, tem_width_flm)
    crops, origins = [], []
    pad_x, pad_y = 0, 0
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

def group_bboxes_by_iou(bboxes, iou_threshold=0.3):
    # group by spatial overlap using IoU
    groups = []  # list of lists of roi indices
    L = len(bboxes)
    G = nx.Graph()
    G.add_nodes_from(range(L))
    bx_xy = [_to_xy(b) for b in bboxes]

    for i in range(L):
        for j in range(i + 1, L):
            if iou(bx_xy[i], bx_xy[j]) >= iou_threshold:
                G.add_edge(i, j)

    groups = list(sorted(list(c)) for c in nx.connected_components(G))
    return groups


def get_roi_mask(img_flm):
    green = img_flm[:, :, 0].astype(float)
    blue  = img_flm[:, :, 2].astype(float)
    bl_gr = green + blue

    thresh   = threshold_otsu(bl_gr)
    roi_mask = (bl_gr > thresh).astype(int)
    return roi_mask

def get_bbox_from_roi_mask(roi_mask, connectivity=2):
    labelled = measure.label(roi_mask, connectivity=connectivity)
    props    = measure.regionprops(labelled)
    bboxes = [p.bbox for p in props]
    return bboxes



def plot_bbox_area_hist(bboxes, bins=50, log_scale=False, figsize=(8,4)):
    areas = [(y1 - y0) * (x1 - x0) for (y0, x0, y1, x1) in bboxes]
    areas = np.array(areas)

    plt.figure(figsize=figsize)
    plt.hist(areas, bins=bins, color='C0', edgecolor='k', alpha=0.8)
    plt.xlabel('Area (pixels)')
    plt.ylabel('Count')
    if log_scale:
        plt.yscale('log')
        plt.ylabel('Count (log scale)')
    plt.title(f'Bounding box area distribution (n={len(areas)})')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    return areas


def bbox_center(b):
    y0, x0, y1, x1 = b
    return ( (y0+y1)//2, (x0+x1)//2 )

def show_zoom(fl_image, roi_mask, bboxes, target_idx, pad=60, cmap='gray'):
    H, W = roi_mask.shape
    y0, x0, y1, x1 = bboxes[target_idx]
    r0 = max(0, y0 - pad)
    c0 = max(0, x0 - pad)
    r1 = min(H, y1 + pad)
    c1 = min(W, x1 + pad)

    # find bboxes whose center lies inside the crop
    inside_idxs = []
    for i, b in enumerate(bboxes):
        cy, cx = bbox_center(b)
        if r0 <= cy < r1 and c0 <= cx < c1:
            inside_idxs.append(i)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # show image for context; use reflection channel if fl_image has channels
    if fl_image.ndim == 3:
        # prefer channel 1 (reflection) if available
        if fl_image.shape[2] > 1:
            ax.imshow(fl_image[r0:r1, c0:c1, 1], cmap=cmap)
        else:
            ax.imshow(fl_image[r0:r1, c0:c1], cmap=cmap)
    else:
        ax.imshow(fl_image[r0:r1, c0:c1], cmap=cmap)

    # overlay roi_mask (transparent)
    ax.imshow(roi_mask[r0:r1, c0:c1], cmap='Reds', alpha=0.25)

    for i in inside_idxs:
        by0, bx0, by1, bx1 = bboxes[i]
        # convert to crop coordinates
        cy0, cx0 = by0 - r0, bx0 - c0
        height = (by1 - by0)
        width  = (bx1 - bx0)
        rect = patches.Rectangle((cx0, cy0), width, height,
                                 linewidth=1.2,
                                 edgecolor='yellow' if i==target_idx else 'lime',
                                 facecolor='none')
        ax.add_patch(rect)
        # label with index and area
        ax.text(cx0 + 2, cy0 + 10, f'{i}', color='white',
                fontsize=8, bbox=dict(facecolor='black', alpha=0.6, pad=1))

    ax.set_title(f'Zoom around bbox {target_idx} ({len(inside_idxs)} boxes inside crop)')
    ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_bbox_over_roi_mask(roi_mask, bboxes):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(roi_mask)

    for idx, bbox in enumerate(bboxes):
        y0, x0, y1, x1 = bbox  # regionprops ordering
        height = y1 - y0
        width  = x1 - x0
        rect = patches.Rectangle((x0, y0), width, height,
                                linewidth=1.5, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        # label box with index at top-left corner
        ax.text(x0 + 2, y0 + 12, str(idx), color='yellow', fontsize=10, bbox=dict(facecolor='black', alpha=0.5, pad=1))

    ax.set_title(f'{len(bboxes)} bboxes on roi_mask')

    ax.axis('off')

def filter_and_merge_bboxes(bboxes, tem_height_flm, tem_width_flm):
    min_area_thresh = tem_height_flm * tem_width_flm
    filtered_bboxes = []
    for bbox in bboxes:
        y0, x0, y1, x1 = bbox
        height = abs(y1 - y0)
        width = abs(x1 - x0)
        if height * width >= min_area_thresh:
            filtered_bboxes.append(bbox)

    merged_bboxes = merge_bboxes(filtered_bboxes, tem_height_flm, tem_width_flm)
    return merged_bboxes


def plot_groups_laplacian(groups_vals, figsize=(10, 6), cols=3, sharex=True, sharey=False):
    """
    groups_vals: List[List[float]] — each inner list contains laplacian values for one group (y-values).
                 x-values will be frame indices 0..len-1 for each group.
    figsize: figure size (width, height)
    cols: max columns in the subplot grid
    sharex/sharey: whether subplots share x/y axes
    Returns (fig, axes)
    """
    n = len(groups_vals)
    if n == 0:
        return None, []

    cols = min(cols, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize,
                             squeeze=False, sharex=sharex, sharey=sharey)
    axes = axes.flatten()

    for i, vals in enumerate(groups_vals):
        ax = axes[i]
        if vals is None or len(vals) == 0:
            ax.set_visible(False)
            continue
        x = list(range(len(vals)))
        ax.plot(x, vals, marker='o', linestyle='-')
        ax.set_title(f'Group {i} (n={len(vals)})')
        ax.set_xlabel('frame_idx')
        ax.set_ylabel('laplacian')

    # hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()



def find_flat_peak(laps, smooth_sigma=2, min_prominence=10):
    """
    Returns the frame index of the flattest/most prominent peak.
    Returns None if no interior peak found.
    """
    smoothed = gaussian_filter1d(laps.astype(float), sigma=smooth_sigma)
    
    peaks, props = find_peaks(
        smoothed,
        prominence=min_prominence,  # ignore small bumps
        width=2,                     # peak must span at least 2 frames
    )
    
    if len(peaks) == 0:
        return None, smoothed
    
    # among all peaks pick the widest one (flattest top)
    widths = props["widths"]
    best = peaks[np.argmax(widths)]
    return int(best), smoothed
