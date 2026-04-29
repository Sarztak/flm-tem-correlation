import numpy as np
import cv2
import gc
from pathlib import Path
from skimage import io
from magicgui import magic_factory
import napari
from napari.utils.notifications import show_info

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.optimize import minimize
from skimage import exposure, io, measure
from skimage.filters import threshold_otsu

DEFAULT_DIR = Path(r"C:\Users\sar31\Documents\GitHub\flm_tem_alignment\jey_002_g3_l3")

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

# Helper function to calculate focus on specific crops
def find_best_frame_idx(flm_stack, img_tem, flm_px, tem_px):
    focus_scores = {}
    for frame_idx in range(flm_stack.shape[0]):
        flm_frame = flm_stack[frame_idx]
        
        # Logic requires finding ROIs first to avoid full-frame noise
        crops, origins = find_roi_with_origins(flm_frame, img_tem, flm_px, tem_px, pad_factor=2)
        
        values = []
        areas = []
        for crop in crops:
            if crop.max() == crop.min():
                continue
            
            crop_u8 = ((crop - crop.min()) / (crop.max() - crop.min()) * 255).astype(np.uint8)
            lap = cv2.Laplacian(crop_u8, cv2.CV_64F).var()
            
            values.append(lap)
            areas.append(crop.shape[0] * crop.shape[1])

        focus_scores[frame_idx] = np.average(values, weights=areas) if values else 0

    sharpest = max(focus_scores, key=focus_scores.get) 
    return sharpest

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
    tem_pixel_nm={"label": "TEM px (nm)", "value": 6.9}
)
def flm_roi_widget(
    viewer: "napari.viewer.Viewer",
    flm_path: Path,
    tem_path: Path,
    flm_pixel_nm: float,
    tem_pixel_nm: float
):
    # 1. Load Data
    flm_stack = io.imread(str(flm_path))
    img_tem = io.imread(str(tem_path))

    # 2. Find Sharpest Frame via Localized ROIs
    best_idx = find_best_frame_idx(flm_stack, img_tem, flm_pixel_nm, tem_pixel_nm)
    best_frame = flm_stack[best_idx]

    # 3. Add to Viewer
    viewer.layers.clear()
    c_axis = 0 if best_frame.shape[0] == 3 else 2
    
    viewer.add_image(
        best_frame, 
        name=[f"Green (F:{best_idx})", f"Reflection (F:{best_idx})", f"Blue (F:{best_idx})"], 
        channel_axis=c_axis,
        colormap=["green", "gray", "blue"],
        blending="additive"
    )

    viewer.add_image(
        best_frame, 
        name=[f"Green (F:{best_idx})", f"Reflection (F:{best_idx})", f"Blue (F:{best_idx})"], 
        channel_axis=c_axis,
        colormap=["green", "gray", "blue"],
        blending="additive"
    )

    viewer.add_shapes(name="Detected ROIs", edge_color="white", face_color="transparent")
    viewer.add_points(name="Target Selection", size=10, face_color="yellow")

    viewer.reset_view()
    show_info(f"Sharpest frame: {best_idx}")

    del flm_stack, img_tem
    gc.collect()