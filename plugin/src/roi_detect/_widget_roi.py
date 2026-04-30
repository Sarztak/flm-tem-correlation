import numpy as np
import cv2
import gc
from pathlib import Path
from skimage import io
from magicgui import magic_factory
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

DEFAULT_DIR = Path(r"C:\Users\sar31\Documents\GitHub\flm_tem_alignment\jey_002_g3_l3")

def get_tile_flm_bbox(flm_height: int, flm_width: int, 
                      tem_height: int, tem_width: int,
                      flm_pixel_nm: float = 121.0, 
                      tem_pixel_nm: float = 6.9, 
                      tile_scale: int = 2) -> list[list[int]]:
    """Given dimensions of two images, it splits the FLM region into overlapping tiles."""
    
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
            merged.append((r0, c0, r1, c1))
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
    tile_scale: int,
):
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


    best_per_group = find_best_roi_frame_idx(all_rois, iou_threshold=iou_threshold)
    h, w = flm_stack.shape[1:3]
    composite = np.full((h, w), 255, dtype=np.uint8)

    rectangles = []
    properties = {'label': [], 'frame': [], 'laplacian': []}

    for i, roi in enumerate(best_per_group):
        frame = flm_stack[roi["frame_idx"]]
        x0, y0, x1, y1 = roi["bbox"]
        crop = frame[y0:y1, x0:x1, 1] # only display reflection channel
        crop_u8 = ((crop - crop.min()) / (crop.max() - crop.min()) * 255).astype(np.uint8)
        composite[y0:y1, x0:x1] = crop_u8

        # napari shapes expects [[row0,col0],[row1,col1]] i.e. [[y0,x0],[y1,x1]]
        rect = np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]])
        rectangles.append(rect)
        properties['label'].append(f"ROI {i} | frame {roi['frame_idx']}")
        properties['frame'].append(roi['frame_idx'])
        properties['laplacian'].append(roi['laplacian'])

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
                print(roi['bbox'], px, py)

                if y0 <= py <= y1 and x0 <= px <= x1:
                    # the bounding box is the key and group points in those bounding boxes
                    pts_per_region[roi_idx].append([px, py])
                    break # assuming that a point belongs to a single ROI
        print(pts_per_region)
        return pts_per_region

    # tile the regions and select those that have the points in them
    def create_tiles():
        filtered_bbox = []
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
                            filtered_bbox.append([t_x0, t_y0, t_x1, t_y1])
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
        
    @viewer.bind_key('Enter')
    def on_done(viewer):
        results = get_points_in_rois()
        filtered_bbox = create_tiles()
        rectangles = [
            np.array([[y0, x0], [y0, x1], [y1, x1], [y1, x0]]) for x0, y0, x1, y1 in filtered_bbox
        ]

        tiles_bbox_shapes_layer = viewer.add_shapes(
            rectangles,
            shape_type='rectangle',
            edge_color='red',
            face_color='transparent',
            edge_width=2,
        )

        for r in results:
            print(f"Point {r['point']} → ROI {r['roi_idx']} | frame {r['roi']['frame_idx']}")

        for r in filtered_bbox:
            print(f"Bbox co-ordinates:{r}")

    points_layer.mode = 'add'
    viewer.reset_view()

    del flm_stack, img_tem
    gc.collect()