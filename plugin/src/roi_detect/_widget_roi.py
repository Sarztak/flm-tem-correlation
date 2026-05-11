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
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT_DIR = Path(r"C:\Users\sar31\Documents\GitHub\flm_tem_alignment")
DEFAULT_DIR = ROOT_DIR / "jey_002_g3_l3"
OUTPUT_DIR = ROOT_DIR / "output"

# add the root_dir to the path to load models 
sys.path.append(str(ROOT_DIR))

from model_setup import upscale_and_save, load_sam2_model, load_lightglue_models, create_tensor_from_mask, get_keypoint_matches, estimate_transform, apply_transform_overlay

from app_helper import *
from best_frame_fix import *

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
    flm_h, flm_w = flm_stack.shape[1:3]
    tem_height_flm = (tem_h * tem_pixel_nm) / flm_pixel_nm
    tem_width_flm  = (tem_w * tem_pixel_nm) / flm_pixel_nm

    # convert uint16 to unint 8 save it once and use again
    rendered_stack_path = OUTPUT_DIR / "rendered_stack.npy"
    if not (rendered_stack_path).exists():
        rendered_stack = render_all_frames(flm_stack, render_flm_frame)
        np.save(rendered_stack_path, rendered_stack)
    else:
        rendered_stack = np.load(rendered_stack_path)

    roi_masks = [get_roi_mask(flm_stack[i]) for i in range(len(flm_stack))]
    bboxes = [get_bbox_from_roi_mask(roi_mask) for roi_mask in roi_masks]
    filted_and_merged_bboxes = [filter_and_merge_bboxes(b, tem_height_flm / 2, tem_width_flm / 2, flm_h, flm_w) for b in bboxes]

    all_bboxes = []
    for b in filted_and_merged_bboxes:
        all_bboxes.extend(b)
    global_bboxes = filter_and_merge_bboxes(all_bboxes, tem_height_flm / 2, tem_width_flm / 2, flm_h, flm_w)

    laps_per_bbox = get_laplacian_per_bbox(flm_stack, global_bboxes)

    bbox_best_frame = []
    for bbox_idx, laps in enumerate(laps_per_bbox):
        peak_idx, peak_vals = find_flat_peak(laps)
        if peak_idx:
            bbox_best_frame.append({"bbox_idx": bbox_idx, "frame_idx": peak_idx})
            show_info(f"BBox: {bbox_idx} -> Best Frame: {peak_idx}, Val: {peak_vals[peak_idx]}")

    # select by index from global bbox and the frame_idx from bbox_best_frame

    # do the tiling, points from this will be used to filter selected points
    tiles_and_frame = tiles_per_best_frame(
        bbox_best_frame=bbox_best_frame, global_bboxes=global_bboxes,
        flm_h=flm_h, flm_w=flm_w, flm_pixel_nm=flm_pixel_nm,
        tem_h=tem_h, tem_w=tem_w, tem_pixel_nm=tem_pixel_nm,
        tile_scale=tile_scale,
    )

    # flatten the tiles_and_frame from list of dictionary to a list of tuple
    tile_bbox_and_frame_idx = []
    for tf in tiles_and_frame:
        best_frame_idx = tf["frame_idx"]
        for bbox in tf["bboxes"]:
            tile_bbox_and_frame_idx.append((bbox, best_frame_idx))

    rectangles = []
    properties = {'label': [], 'frame': []}

    composite = np.full((flm_h, flm_w, 3), 0, dtype=np.uint8)
    for b in bbox_best_frame:
        best_bbox_idx = b["bbox_idx"]
        best_frame_idx = b["frame_idx"]
            
        best_bbox_pts = global_bboxes[best_bbox_idx]
        bb_y0, bb_x0, bb_y1, bb_x1 = best_bbox_pts

        rect = np.array([[bb_y0, bb_x0], [bb_y0, bb_x1], [bb_y1, bb_x1], [bb_y1, bb_x0]])
        rectangles.append(rect)
        properties['label'].append(f"ROI {best_bbox_idx} | frame {best_frame_idx}")
        properties['frame'].append(best_frame_idx)

        composite[bb_y0:bb_y1, bb_x0:bb_x1] = rendered_stack[best_frame_idx, bb_y0:bb_y1, bb_x0:bb_x1]

    # prepare the tem_img for the next stage
    img_uint8, inv_thresh_tem_img = prepare_tem(img_tem)
    cv2.imwrite(OUTPUT_DIR / "tem.png", img_uint8)
    cv2.imwrite(OUTPUT_DIR / "tem_inv_thresh.png", inv_thresh_tem_img)

    viewer.add_image(rendered_stack[0], name='flm stack')
    viewer.add_image(composite, name="best frame per roi")

    viewer.add_shapes(
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

    points_layer.mode = 'add'
    viewer.reset_view()

    # when other images are loaded, it tried to bind the enter key again which causes error, therefore it needs to check before binding
    viewer.bind_key('Enter', None, overwrite=True) 
    @viewer.bind_key('Enter')
    def on_done(viewer):
        
        user_selected_tile_and_frame = []
        pts = points_layer.data  # shape (N, 2) as [row, col] i.e. [y, x]
        for pt in pts:
            py, px = pt[0], pt[1]
            for bbox, best_frame_idx in tile_bbox_and_frame_idx:
                t_y0, t_x0, t_y1, t_x1 = bbox # this has already been in the frame of image
                if t_x0 < px < t_x1 and t_y0 < py < t_y1:
                    user_selected_tile_and_frame.append((bbox, best_frame_idx))
        
        tile_rectangles = []
        bboxes = []
        for idx, (bbox, best_frame_idx) in enumerate(user_selected_tile_and_frame):
            t_y0, t_x0, t_y1, t_x1 = bbox
            bboxes.append(bbox)
            flm_frame = flm_stack[best_frame_idx, :, :, 1] # use refl channel
            flm_frame_eq = equalize_flm_frame(flm_frame)
            flm_frame_eq_rgb = np.stack([flm_frame_eq] * 3, axis=-1)
            best_roi_cropped = flm_frame_eq_rgb[t_y0:t_y1, t_x0:t_x1]
            tile_rectangles.append(
                np.array([[t_y0, t_x0], [t_y0, t_x1], [t_y1, t_x1], [t_y1, t_x0]])
            )
            cv2.imwrite(ff_bb_save_dir / f'{str(idx).zfill(4)}.png', best_roi_cropped)
            flm_frame_path = OUTPUT_DIR / f"flm_frame_{best_frame_idx}.png"
            cv2.imwrite(flm_frame_path, rendered_stack[best_frame_idx, :, :, 1])
            state["flm_idx_img_path"].append(str(flm_frame_path))
        state["bboxes"].append(bboxes)

        viewer.add_shapes(
            tile_rectangles,
            shape_type='rectangle',
            edge_color='red',
            face_color='transparent',
            edge_width=2,
        )

        upscale_and_save(ff_bb_save_dir, upscaled_ff_bb_save_dir)

@magic_factory(call_button="Load Segmentation Images")
def segment_widget(viewer: "napari.viewer.Viewer"):
    for layer in viewer.layers:
        layer.visible = False

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

        # annotation layer for the background points, color red
        annotation_layer_flm_red = viewer.add_points(
            ndim=2,
            name="FLM Background Segmentation Points",
            size=15,
            face_color="red",
        )
        annotation_layer_flm_red.mode = 'add'

        # annotation layer for the foreground points, color green
        annotation_layer_flm_green = viewer.add_points(
            ndim=2,
            name="FLM Foreground Segmentation Points",
            size=15,
            face_color="green",
        )
        annotation_layer_flm_green.mode = 'add'

        viewer.reset_view()
        
        # bind Enter to confirm annotation and store points
        viewer.bind_key('Enter', None, overwrite=True)
        @viewer.bind_key('Enter')
        def on_annotation_done(viewer):
            annotation_pts_flm_green = annotation_layer_flm_green.data 
            annotation_layer_flm_green.metadata["points"] = annotation_pts_flm_green
            annotation_pts_flm_red = annotation_layer_flm_red.data 
            annotation_layer_flm_red.metadata["points"] = annotation_pts_flm_red
            print(f"Annotation points: {annotation_pts_flm_green}")
            show_info(f"{len(annotation_pts_flm_green)} points confirmed for further processing.")

            predictor.set_image(selected_img)

            coords_flm_green = [
                [int(pt[1]), int(pt[0])] # the metadata is stored in (y, x) but SAM needs data in (x, y)
                for pt in annotation_layer_flm_green.metadata["points"]
            ]

            coords_flm_red = [
                [int(pt[1]), int(pt[0])] # the metadata is stored in (y, x) but SAM needs data in (x, y)
                for pt in annotation_layer_flm_red.metadata["points"]
            ]

            coords = np.array(coords_flm_green + coords_flm_red)
            labs = np.array([1] * len(coords_flm_green) + [0] * len(coords_flm_red))

            with torch.inference_mode():
                masks, scores, _ = predictor.predict(
                    point_coords=coords,
                    point_labels=labs,
                    multimask_output=True
                )
            
            flm_segmentation_mask = masks[np.argmax(scores)]
            flm_segmentation_mask = flm_segmentation_mask.astype(np.uint8) * 255
            cv2.imwrite(segmentation_dir / 'flm_seg_mask.png', flm_segmentation_mask)

            tem_path = OUTPUT_DIR / "tem_inv_thresh.png"

            if tem_path.exists():
                tem_inv_thresh_img = cv2.imread(tem_path)

                # in case selected image is not RGB, SAM needs it 
                if len(tem_inv_thresh_img.shape) == 2:
                    tem_inv_thresh_img = np.stack([tem_inv_thresh_img] * 3)
                viewer.add_image(tem_inv_thresh_img, name="tem segmentation image")

                # annotation layer for the background points, color red
                annotation_layer_tem_red = viewer.add_points(
                    ndim=2,
                    name="TEM Background Segmentation Points",
                    size=15,
                    face_color="red",
                )
                annotation_layer_tem_red.mode = 'add'
                viewer.reset_view()

                # annotation layer for the foreground points, color green
                annotation_layer_tem_green = viewer.add_points(
                    ndim=2,
                    name="TEM Foreground Segmentation Points",
                    size=15,
                    face_color="green",
                )
                annotation_layer_tem_green.mode = 'add'
       
                viewer.bind_key('Enter', None, overwrite=True)
                @viewer.bind_key('Enter')
                def on_tem_annotation_done(viewer):

                    annotation_pts_tem_green = annotation_layer_tem_green.data 
                    annotation_layer_tem_green.metadata["points"] = annotation_pts_tem_green
                    annotation_pts_tem_red = annotation_layer_tem_red.data 
                    annotation_layer_tem_red.metadata["points"] = annotation_pts_tem_red
                    print(f"Annotation points: {annotation_pts_tem_green}")
                    show_info(f"{len(annotation_pts_tem_green)} points confirmed for further processing.")

                    predictor.set_image(tem_inv_thresh_img)

                    coords_tem_green = [
                        [int(pt[1]), int(pt[0])] # the metadata is stored in (y, x) but SAM needs data in (x, y)
                        for pt in annotation_layer_tem_green.metadata["points"]
                    ]

                    coords_tem_red = [
                        [int(pt[1]), int(pt[0])] # the metadata is stored in (y, x) but SAM needs data in (x, y)
                        for pt in annotation_layer_tem_red.metadata["points"]
                    ]

                    coords = np.array(coords_tem_green + coords_tem_red)
                    print(coords)
                    labs = np.array([1] * len(coords_tem_green) + [0] * len(coords_tem_red))

                    with torch.inference_mode():
                        masks, scores, _ = predictor.predict(
                            point_coords=coords,
                            point_labels=labs,
                            multimask_output=True
                        )
                    
                    tem_segmentation_mask = masks[np.argmax(scores)]
                    tem_segmentation_mask = tem_segmentation_mask.astype(np.uint8) * 255
                    # viewer.add_image(tem_segmentation_mask, name="TEM Mask")
                    cv2.imwrite(segmentation_dir / 'tem_seg_mask.png', tem_segmentation_mask)
                    
                    fig, ax = plt.subplots(1, 2, figsize=(20, 15))
                    ax[0].imshow(flm_segmentation_mask, cmap='gray')
                    ax[0].set_title('FLM Segmentation Mask')
                    ax[0].axis('off')
                    ax[1].imshow(tem_segmentation_mask, cmap='gray')
                    ax[1].axis('off')
                    ax[1].set_title('TEM Segmentation Mask')
                    plt.tight_layout()
                    fig.savefig(segmentation_dir / 'FLM_TEM Segmentation Masks.png')
                    flm_tem_seg_img = cv2.imread(segmentation_dir / 'FLM_TEM Segmentation Masks.png', 0)

                    # add to napari
                    for layer in viewer.layers:
                        layer.visible = False
                    viewer.add_image(flm_tem_seg_img, name="Segmented Images")

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

    for layer in viewer.layers:
        layer.visible = False

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
    bbox_origin_y, bbox_origin_x, _, _ = bboxes[selected_bbox_idx] # the bounding boxes are y, x

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