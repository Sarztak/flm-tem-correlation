import numpy as np
import cv2
import gc
from pathlib import Path
from skimage import io
from magicgui import magic_factory
import napari
from napari.utils.notifications import show_info
import networkx as nx

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import minimize
from skimage import exposure, io, measure
from skimage.filters import threshold_otsu

DEFAULT_DIR = Path(r"C:\Users\sar31\Documents\GitHub\flm_tem_alignment\jey_002_g3_l3")

def has_interior_peak(group_rois):
    """Returns True if the Laplacian peaks at an interior frame, not at the edges."""
    sorted_by_frame = sorted(group_rois, key=lambda r: r["frame_idx"])
    values = [r["laplacian"] for r in sorted_by_frame]
    peak_idx = np.argmax(values)
    # peak must not be at the first or last frame
    return 0 < peak_idx < len(values) - 1

def merge_bboxes(bboxes, tem_height_flm: float, tem_width_flm: float) -> list:
    merge_dist = int(max(tem_height_flm, tem_width_flm))
    expanded = [
        (r0 - merge_dist, c0 - merge_dist, r1 + merge_dist, c1 + merge_dist)
        for r0, c0, r1, c1 in bboxes
    ]
    expanded.sort(key=lambda x: x[0])
    merged = [expanded[0]]
    for r0, c0, r1, c1 in expanded[1:]:
        pr0, pc0, pr1, pc1 = merged[-1]
        if r0 <= pr1 and c0 <= pc1:
            merged[-1] = (min(pr0, r0), min(pc0, c0), max(pr1, r1), max(pc1, c1))
        else:
            merged.append((r0, c0, r1, c1))
    return merged

def find_roi_with_origins(
    img_flm: np.ndarray,
    img_tem: np.ndarray,
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
    tem_h, tem_w = img_tem.shape[:2]
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

def get_all_rois(flm_stack, img_tem, flm_pixel_nm, tem_pixel_nm, pad_factor=2):
    # collect all ROIs across all frames with their bounding boxes and laplacian
    all_rois = []  # list of {frame_idx, origin, bbox, laplacian}

    for frame_idx in range(flm_stack.shape[0]):
        flm_frame = flm_stack[frame_idx]
        crops, origins = find_roi_with_origins(flm_frame, img_tem, flm_pixel_nm, tem_pixel_nm, pad_factor=pad_factor)

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

    del flm_stack, img_tem
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
        print(
            f"ROI group size={len(group):2d}  "
            f"best frame={best_roi['frame_idx']:2d}  "
            f"laplacian={best_roi['laplacian']:.2f}  "
            f"mean_intensity={best_roi['mean_intensity']:.2f}  "
            f"bbox={best_roi['bbox']}"
        )

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


@magic_factory(
    call_button="Detect Best Frame & ROIs",
    flm_path={
        "label": "FLM Stack Path", 
        "mode": "r", 
        "value": DEFAULT_DIR / "FLM-stack_JEY002_G3_L3.tif"
    },
    tem_path={
        "label": "TEM Image Path", 
        "mode": "r", 
        "value": DEFAULT_DIR / "JEY002_G3_L3_1950x_t-13.tif"
    },
    flm_pixel_nm={"label": "FLM px (nm)", "value": 121.0},
    tem_pixel_nm={"label": "TEM px (nm)", "value": 6.9},
    pad_factor={"label": "padding factor", "value": 1},
    iou_threshold={"label": "IOU threshold", "value": 0.3},
)

def flm_roi_widget(
    viewer: "napari.viewer.Viewer",
    flm_path: Path,
    tem_path: Path,
    flm_pixel_nm: float,
    tem_pixel_nm: float,
    pad_factor: int,
    iou_threshold: float,
):
    # 1. Load Data
    flm_stack = io.imread(str(flm_path))
    img_tem = io.imread(str(tem_path))

    all_rois = get_all_rois(
        flm_stack, 
        img_tem, 
        flm_pixel_nm=flm_pixel_nm, 
        tem_pixel_nm=tem_pixel_nm, 
        pad_factor=pad_factor,
    )


    best_per_group = find_best_roi_frame_idx(all_rois, iou_threshold=iou_threshold)
    h, w = flm_stack.shape[1:3]
    composite = np.full((h, w), 255, dtype=np.uint8)

    for roi in best_per_group:
        frame = flm_stack[roi["frame_idx"]]
        x0, y0, x1, y1 = roi["bbox"]
        crop = frame[y0:y1, x0:x1, 1] # only display reflection channel
        crop_u8 = ((crop - crop.min()) / (crop.max() - crop.min()) * 255).astype(np.uint8)
        composite[y0:y1, x0:x1] = crop_u8

    viewer.add_image(composite, name="best frame per roi")
    viewer.reset_view()

    del flm_stack, img_tem
    gc.collect()