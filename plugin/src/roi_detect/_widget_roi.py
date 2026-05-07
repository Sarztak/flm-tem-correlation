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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = Path(r"C:\Users\sar31\Documents\GitHub\flm_tem_alignment")
DEFAULT_DIR = ROOT_DIR / "jey_002_g3_l3"
OUTPUT_DIR = ROOT_DIR / "output"

# add the root_dir to the path to load models 
sys.path.append(str(ROOT_DIR))

from model_setup import upscale_and_save, load_sam2_model, load_lightglue_models, create_tensor_from_mask, get_keypoint_matches, estimate_transform, apply_transform_overlay

ff_bb_save_dir = OUTPUT_DIR / 'filtered_bbox'
upscaled_ff_bb_save_dir = OUTPUT_DIR / 'upscaled_filtered_bbox'
segmentation_dir = OUTPUT_DIR / 'segmentation'
handoff_dir = OUTPUT_DIR / "handoff"

ff_bb_save_dir.mkdir(exist_ok=True)
upscaled_ff_bb_save_dir.mkdir(exist_ok=True)
segmentation_dir.mkdir(exist_ok=True)
handoff_dir.mkdir(exist_ok=True)

_, predictor = load_sam2_model()
extractor, matcher = load_lightglue_models()
state = dict(flm_idx_img_path=[], bboxes=[], selected_bbox_idx=[])

def show_transform_result(viewer, M, scale):
    widget = QWidget()
    layout = QVBoxLayout()
    
    text = (
        f"<b>Scale:</b> {scale:.4f}<br><br>"
        f"<b>Transform Matrix:</b><br>"
        f"{M[0,0]:.4f}  {M[0,1]:.4f}  {M[0,2]:.4f}<br>"
        f"{M[1,0]:.4f}  {M[1,1]:.4f}  {M[1,2]:.4f}"
    )
    label = QLabel(text)
    layout.addWidget(label)
    widget.setLayout(layout)
    
    viewer.window.add_dock_widget(widget, name="Transform Result", area="right")

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

def prepare_tem(tem_path, thresh=130):
    from skimage import io
    raw = io.imread(tem_path)
    if raw.ndim == 3:
        raw = raw[:, :, 0]
    img_u8 = ((raw.astype(float) - raw.min()) / (raw.max() - raw.min() + 1e-8) * 255).astype(np.uint8)
    inverted = 255 - img_u8
    thresh_img = (inverted > thresh).astype(np.uint8) * 255
    return img_u8, thresh_img  # img_u8 for final overlay, inverted for SAM

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
    tile_scale={"label": "FLM Tile Padding", "value": 2},
)
def flm_roi_widget(
    viewer: "napari.viewer.Viewer",
    flm_path: Path,
    tem_path: Path,
    flm_pixel_nm: float,
    tem_pixel_nm: float,
    pad_factor: int,
    iou_threshold: float,
    tile_scale: float,
):

    # clean all the layers before loading anything
    viewer.layers.clear()

    # 1. Load Data
    flm_stack = io.imread(str(flm_path))
    img_tem = io.imread(str(tem_path))
    tem_h, tem_w = img_tem.shape[:2]
    all_rois = get_all_rois(
        flm_stack, 
        tem_h, tem_w,
        flm_pixel_nm=flm_pixel_nm, 
        tem_pixel_nm=tem_pixel_nm, 
        pad_factor=pad_factor,
    )


    # prepare the tem_img for the next stage
    img_uint8, inv_thresh_tem_img = prepare_tem(tem_path)
    cv2.imwrite(OUTPUT_DIR / "tem.png", img_uint8)
    cv2.imwrite(OUTPUT_DIR / "tem_inv_thresh.png", inv_thresh_tem_img)

    best_per_group = find_best_roi_frame_idx(all_rois, iou_threshold=iou_threshold)
    h, w = flm_stack.shape[1:3]
    composite = np.full((h, w, 3), 0, dtype=np.uint8)

    rectangles = []
    properties = {'label': [], 'frame': [], 'laplacian': []}

    for i, roi in enumerate(best_per_group):
        frame = flm_stack[roi["frame_idx"]]
        x0, y0, x1, y1 = roi["bbox"]
        crop = frame[y0:y1, x0:x1, :] # only display reflection channel

        refl = crop[:, :, 1]
        green = crop[:, :, 0]
        blue = crop[:, :, 2]
        
        refl_u8 = norm(refl)
        blue_u8 = norm(blue)
        green_u8 = norm(green)

        composite[y0:y1, x0:x1, 2] = np.clip(refl_u8.astype(int) + blue_u8, 0, 255)  # Blue channel
        composite[y0:y1, x0:x1, 1] = np.clip(refl_u8.astype(int) + green_u8, 0, 255) # Green channel
        composite[y0:y1, x0:x1, 0] = refl_u8
        # crop_u8 = ((crop - crop.min()) / (crop.max() - crop.min()) * 255).astype(np.uint8)
        # composite[y0:y1, x0:x1, :] = crop_u8

        # napari shapes expects [[row0,col0],[row1,col1]] i.e. [[y0,x0],[y1,x1]]
        rect = np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])
        rectangles.append(rect)
        properties['label'].append(f"ROI {i} | frame {roi['frame_idx']}")
        properties['frame'].append(roi['frame_idx'])
        properties['laplacian'].append(roi['laplacian'])

    # select the middle image of the flm_stack to serve as background image 
    napari_compatible_flm_img = render_flm_frame(flm_stack[len(flm_stack)//2])
    viewer.add_image(napari_compatible_flm_img, name='flm stack')

    viewer.add_image(composite, name="best frame per roi")

    shapes_layer = viewer.add_shapes(
        rectangles,
        shape_type='rectangle',
        edge_color='white',
        face_color='transparent',
        edge_width=2,
        properties=properties,
        text={
            'string': 'label',       # which property to show as text
            'size': 14,
            'color': 'red',
            'anchor': 'upper_left',
        },
        name='Detected ROIs',
    )

    # add an empty points layer for user to annotate
    points_layer = viewer.add_points(
        ndim=2,
        name='User Points',
        size=15,
        face_color='yellow',
    )


    # after user has added points, run this to find which ROI each point is in
    def collect_user_data():
        pts = points_layer.data
        pts_per_region = defaultdict(list)
    
        for pt in pts:
            py, px = pt[0], pt[1]
            for roi_idx, roi in enumerate(best_per_group):
                x0, y0, x1, y1 = roi['bbox']

                if y0 <= py <= y1 and x0 <= px <= x1:
                    # the bounding box is the key and group points in those bounding boxes
                    pts_per_region[roi_idx].append([px, py])
                    break # assuming that a point belongs to a single ROI
        return pts_per_region

    # tile the regions and select those that have the points in them
    def create_tiles():
        filtered_bbox = defaultdict(list)
        pts_per_region = collect_user_data()
        for roi_idx, roi in enumerate(best_per_group):
            if roi_idx in pts_per_region:
                # select only the reflection channel to tile
                x0, y0, x1, y1 = roi['bbox']
                origin_x, origin_y = roi['origin']
                flm_w = x1 - x0
                flm_h = y1 - y0
                tiles_bbox = get_tile_flm_bbox(
                    flm_height=flm_h, flm_width=flm_w, 
                    tem_height=tem_h, tem_width=tem_w,
                    flm_pixel_nm=flm_pixel_nm, tem_pixel_nm=tem_pixel_nm,
                    tile_scale=tile_scale
                )

                # now filter the bounding boxes that have the user selected points            
                # I want to optimize this later on
                for t_x0, t_y0, t_x1, t_y1 in tiles_bbox:
                    # bbox are in the reference frame of the ROI, so they need to be made into absolute coordinates
                    t_x0 += origin_x
                    t_x1 += origin_x
                    t_y0 += origin_y
                    t_y1 += origin_y
                    for [px, py] in pts_per_region[roi_idx]:
                        if t_x0 < px < t_x1 and t_y0 < py < t_y1:
                            filtered_bbox[roi_idx].append([t_x0, t_y0, t_x1, t_y1])
                            break # just find a box per point because if other points are there then same box will be selected so no need to check
        return filtered_bbox 


    def get_points_in_rois():
        pts = points_layer.data  # shape (N, 2) as [row, col] i.e. [y, x]
        results = []
        for pt in pts:
            py, px = pt[0], pt[1]
            for i, roi in enumerate(best_per_group):
                x0, y0, x1, y1 = roi['bbox']
                if y0 <= py <= y1 and x0 <= px <= x1:
                    results.append({'point': pt, 'roi_idx': i, 'roi': roi})
                    # break  # assume one ROI per point
        return results

    # when other images are loaded, it tried to bind the enter key again which causes error, therefore it needs to check before binding
    viewer.bind_key('Enter', None, overwrite=True) 
    @viewer.bind_key('Enter')
    def on_done(viewer):
        results = get_points_in_rois()
        filtered_bbox = create_tiles() # dictionary of filtered tiles per roi_idx

        rectangles = []
        crop_idx = 0
        for roi_idx, roi in enumerate(best_per_group):
            if roi_idx in filtered_bbox:
                bboxes = filtered_bbox[roi_idx]
                flm_frame = flm_stack[roi['frame_idx'], :, :, 1] # use refl channel
                flm_frame_uint8 = norm(flm_frame)
                flm_img_hist_eq = exposure.equalize_hist(flm_frame_uint8)
                flm_img_hist_eq = (flm_img_hist_eq * 255).astype(np.uint8)

                # need to save the bounding boxes and the flm img since they will be used for transformation 
                flm_frame_path = OUTPUT_DIR / f"flm_frame_{roi['frame_idx']}.png"
                cv2.imwrite(flm_frame_path, flm_frame_uint8) 
                state["flm_idx_img_path"].append(str(flm_frame_path))
                state["bboxes"].append(bboxes)

                for (x0, y0, x1, y1) in bboxes:
                    # crop the regions from the refl channel and store them in output folder
                    crop = flm_img_hist_eq[y0:y1, x0:x1]
                    crop = np.stack([crop] * 3, axis=-1)
                    cv2.imwrite(ff_bb_save_dir / f'{str(crop_idx).zfill(4)}.png', crop)
                    crop_idx += 1
                    rectangles.append(np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]]))

        upscale_and_save(ff_bb_save_dir, upscaled_ff_bb_save_dir)
        tiles_bbox_shapes_layer = viewer.add_shapes(
            rectangles,
            shape_type='rectangle',
            edge_color='red',
            face_color='transparent',
            edge_width=2,
        )


        for r in results:
            print(f"Point {r['point']} → ROI {r['roi_idx']} | frame {r['roi']['frame_idx']}")


    points_layer.mode = 'add'
    viewer.reset_view()


    # del flm_stack, img_tem
    gc.collect()

@magic_factory(call_button="Load Segmentation Images")
def segment_widget(viewer: "napari.viewer.Viewer"):
    img_paths = sorted(upscaled_ff_bb_save_dir.glob("*.png"))
    if not img_paths:
        show_info("No images found.")
        return

    images = [io.imread(str(p)) for p in img_paths]
    
    PADDING = 20
    COLS = 3
    rows = int(np.ceil(len(images) / COLS))
    h, w = images[0].shape[:2]
    c = images[0].shape[2] if images[0].ndim == 3 else 1

    canvas_h = rows * h + (rows + 1) * PADDING
    canvas_w = COLS * w + (COLS + 1) * PADDING
    canvas = np.zeros((canvas_h, canvas_w, c), dtype=np.uint8)

    for idx, img in enumerate(images):
        row = idx // COLS
        col = idx % COLS
        y0 = PADDING + row * (h + PADDING)
        x0 = PADDING + col * (w + PADDING)
        canvas[y0:y0+h, x0:x0+w] = img if img.ndim == 3 else img[:, :, np.newaxis]

    viewer.layers.clear()
    viewer.add_image(canvas, name="Segmentation Grid")


    points_layer = viewer.add_points(ndim=2, name="Selection", size=20, face_color="green")

    points_layer.mode = 'add'

    viewer.reset_view()

    viewer.bind_key('Enter', None, overwrite=True) 
    @viewer.bind_key('Enter')
    def on_point_added(event):
        pts = points_layer.data
        if len(pts) == 0:
            return

        # keep only the latest point
        latest = pts[-1]
        points_layer.data = pts[-1:]

        py, px = latest[0], latest[1]
        col = int((px - PADDING) // (w + PADDING))
        row = int((py - PADDING) // (h + PADDING))

        # check if click landed inside an actual image (not on padding)
        x0 = PADDING + col * (w + PADDING)
        y0 = PADDING + row * (h + PADDING)
        if not (x0 <= px < x0 + w and y0 <= py < y0 + h):
            show_info("Click inside an image, not on padding.")
            points_layer.data = np.empty((0, 2))
            return

        idx = row * COLS + col
        if idx >= len(img_paths):
            show_info("No image at that position.")
            points_layer.data = np.empty((0, 2))
            return

        # selected_path = img_paths[idx]
        # show_info(f"selected: {selected_path.name}")
        # print(f"selected image path: {selected_path}")
        points_layer.metadata["selected_idx"] = idx
        state["selected_bbox_idx"].append(idx)

        # hide all the existing layers
        for layer in viewer.layers:
            layer.visible = False 

        # add selected image and show it
        selected_img = images[idx]
        selected_img = (selected_img > 130).astype(np.uint8) * 255

        # in case selected image is not RGB, SAM needs it 
        if len(selected_img.shape) == 2:
            selected_img = np.stack([selected_img] * 3)
            
        selected_flm_layer = viewer.add_image(selected_img, name="FLM Crop Selected for Segmentation")
        selected_flm_layer.visible = True

        # add a fresh annotation layer for annotation on the selected img
        annotation_layer = viewer.add_points(
            ndim=2,
            name="Segmentation Points",
            size=15,
            face_color="green",
        )
        annotation_layer.mode = 'add'
        viewer.reset_view()
        
        # bind Enter to confirm annotation and store points
        viewer.bind_key('Enter', None, overwrite=True)
        @viewer.bind_key('Enter')
        def on_annotation_done(viewer):
            annotation_pts = annotation_layer.data 
            annotation_layer.metadata["points"] = annotation_pts
            print(f"Annotation points: {annotation_pts}")
            show_info(f"{len(annotation_pts)} points confirmed for further processing.")

            predictor.set_image(selected_img)

            coords = np.array([
                [int(pt[1]), int(pt[0])] # the metadata is stored in (y, x) but SAM needs data in (x, y)
                for pt in annotation_layer.metadata["points"]
            ])

            labs = np.array([1] * len(annotation_layer.metadata["points"]))

            with torch.inference_mode():
                masks, scores, _ = predictor.predict(
                    point_coords=coords,
                    point_labels=labs,
                    multimask_output=True
                )
            
            flm_segmentation_mask = masks[np.argmax(scores)]
            flm_segmentation_mask = flm_segmentation_mask.astype(np.uint8) * 255
            viewer.add_image(flm_segmentation_mask.astype(np.uint8) * 255, name="FLM Mask")
            cv2.imwrite(segmentation_dir / 'flm_seg_mask.png', flm_segmentation_mask)

            tem_path = OUTPUT_DIR / "tem_inv_thresh.png"

            if tem_path.exists():
                tem_inv_thresh_img = cv2.imread(tem_path)

                # in case selected image is not RGB, SAM needs it 
                if len(tem_inv_thresh_img.shape) == 2:
                    tem_inv_thresh_img = np.stack([tem_inv_thresh_img] * 3)
                viewer.add_image(tem_inv_thresh_img, name="tem segmentation image")
                annotation_layer_tem = viewer.add_points(ndim=2, name="TEM Segmentation Points", size=15, face_color="red")
                annotation_layer_tem.mode = 'add'
                viewer.reset_view()

                viewer.bind_key('Enter', None, overwrite=True)
                @viewer.bind_key('Enter')
                def on_tem_annotation_done(viewer):
                    annotation_pts = annotation_layer_tem.data 
                    annotation_layer_tem.metadata["points"] = annotation_pts
                    print(f"Annotation points: {annotation_pts}")
                    show_info(f"{len(annotation_pts)} points confirmed for further processing.")

                    predictor.set_image(tem_inv_thresh_img)

                    coords = np.array([
                        [int(pt[1]), int(pt[0])] # the metadata is stored in (y, x) but SAM needs data in (x, y)
                        for pt in annotation_layer_tem.metadata["points"]
                    ])

                    labs = np.array([1] * len(annotation_layer_tem.metadata["points"]))

                    with torch.inference_mode():
                        masks, scores, _ = predictor.predict(
                            point_coords=coords,
                            point_labels=labs,
                            multimask_output=True
                        )
                    
                    tem_segmentation_mask = masks[np.argmax(scores)]
                    tem_segmentation_mask = tem_segmentation_mask.astype(np.uint8) * 255
                    viewer.add_image(tem_segmentation_mask, name="TEM Mask")
                    cv2.imwrite(segmentation_dir / 'tem_seg_mask.png', tem_segmentation_mask)
                    
                    # write all the information present in the state dictionary
                    with open(OUTPUT_DIR / 'state.json', 'w') as w:
                        json.dump(state, w)


# upscaled_ff_bb_save_dir = OUTPUT_DIR / 'upscaled_filtered_bbox'
@magic_factory(
        call_button="Match Keypoints",
        thresh={"label": "Match Threshold", "value": 0.02}
)
def match_widget(viewer: "napari.viewer.Viewer", thresh: float):
    from lightglue import viz2d

    flm_segmentation_mask = cv2.imread(segmentation_dir / 'flm_seg_mask.png') / 255.0
    tem_segmentation_mask = cv2.imread(segmentation_dir / 'tem_seg_mask.png') / 255.0

    t0 = create_tensor_from_mask(flm_segmentation_mask, DEVICE)
    t1t = create_tensor_from_mask(tem_segmentation_mask, DEVICE)

    _, _, mk0, mk1, m01 = get_keypoint_matches(extractor, matcher, t0, t1t, thresh)
    
    mk0 = mk0.cpu().numpy()
    mk1 = mk1.cpu().numpy()

    show_info(f"Found {len(mk0)} matches")
    viz2d.plot_images([t0[0][0], t1t[0][0]])
    viz2d.plot_matches(mk0, mk1, color="lime", lw=0.2)
    fig = plt.gcf()

    fig.savefig(OUTPUT_DIR / 'matches.png')
    img_arr = cv2.imread(OUTPUT_DIR / 'matches.png')
    plt.close(fig)

    viewer.add_image(img_arr, name=f"Keypoint Matches ({len(mk0)})", rgb=True)

    # load the state
    with open(OUTPUT_DIR / 'state.json', 'r') as r:
        state = json.load(r)

    flm_frame_path = state["flm_idx_img_path"][0]
    bboxes = state["bboxes"][0]
    selected_bbox_idx = state["selected_bbox_idx"][0]

    # select the idx (for debugging)
    # flm_path = [p for p in upscaled_ff_bb_save_dir.glob("*.png")][selected_bbox_idx]
    # flm_img = cv2.imread(flm_path)

    flm_img = cv2.imread(flm_frame_path)
    bbox_origin_x, bbox_origin_y, _, _ = bboxes[selected_bbox_idx]

    # transform the keypoints on the flm upscaled image to the original image
    mk0 = mk0 / 4 # images are upscaled 4x
    mk0[:, 0] += bbox_origin_x
    mk0[:, 1] += bbox_origin_y
    
    # estimate transform
    M, _, scale = estimate_transform(mk0, mk1)

    tem_img = cv2.imread(OUTPUT_DIR / 'tem.png')

    # tem_img, and flm_img should both be gray scales images before passing to the function
    if flm_img.ndim == 3:
        flm_img = cv2.cvtColor(flm_img, cv2.COLOR_BGR2GRAY)
    if tem_img.ndim == 3:
        tem_img = cv2.cvtColor(tem_img, cv2.COLOR_BGR2GRAY)

    overlay, _, _, _ = apply_transform_overlay(flm_img, tem_img, M)

    show_info(f"Scale: {scale:.4f} | Matrix:\n{M}")

    M, _, scale = estimate_transform(mk0, mk1)
    show_transform_result(viewer, M, scale)

    viewer.add_image(overlay, name="Overlay", rgb=True)
    viewer.reset_view()