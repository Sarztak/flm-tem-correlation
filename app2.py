import gc
import json
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import streamlit as st
import torch
from PIL import Image, ImageDraw
from scipy.ndimage import label
from skimage import exposure, io, measure
from skimage.filters import threshold_otsu
from streamlit_image_coordinates import streamlit_image_coordinates

ROOT_DIR = Path(r"C:\Users\sar31\Documents\GitHub\flm_tem_alignment")
DEFAULT_FLM = ROOT_DIR / "jey_002_g3_l3" / "FLM-stack_JEY002_G3_L3.tif"
DEFAULT_TEM = ROOT_DIR / "jey_002_g3_l3" / "JEY002_G3_L3_1950x_t-13.tif"
OUTPUT_DIR = ROOT_DIR / "output"
FF_BB_DIR = OUTPUT_DIR / "filtered_bbox"
UPSCALED_DIR = OUTPUT_DIR / "upscaled_filtered_bbox"
SEG_DIR = OUTPUT_DIR / "segmentation"

for d in [OUTPUT_DIR, FF_BB_DIR, UPSCALED_DIR, SEG_DIR]:
    d.mkdir(exist_ok=True, parents=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISPLAY_WIDTH = 700
COLS = 3
PADDING = 16

st.set_page_config(layout="wide", page_title="FLM–TEM Alignment", page_icon="🔬")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0d0f12;
    color: #c9d1d9;
}
h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; color: #e6edf3; letter-spacing: -0.02em; }
.stButton > button {
    background: #161b22; border: 1px solid #30363d;
    color: #58a6ff; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem; padding: 0.4rem 1rem;
    transition: all 0.15s;
}
.stButton > button:hover { background: #1f2937; border-color: #58a6ff; }
.stButton > button[kind="primary"] { background: #1f6feb; color: #fff; border-color: #1f6feb; }
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem;
    color: #8b949e; padding: 0.5rem 1.2rem;
}
.stTabs [aria-selected="true"] { color: #58a6ff; border-bottom: 2px solid #58a6ff; }
.stSidebar { background: #0d1117; border-right: 1px solid #21262d; }
.metric-box {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 6px; padding: 0.75rem 1rem; margin-bottom: 0.5rem;
}
.metric-box span { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: #8b949e; }
.metric-box strong { font-family: 'IBM Plex Mono', monospace; font-size: 1rem; color: #e6edf3; }
.step-badge {
    display: inline-block; background: #1f2937;
    border: 1px solid #30363d; border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem;
    color: #58a6ff; padding: 2px 8px; margin-bottom: 0.5rem;
}
.status-ok { color: #3fb950; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; }
.status-warn { color: #d29922; font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; }
hr { border-color: #21262d; }
</style>
""", unsafe_allow_html=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def norm(img):
    d = img.max() - img.min()
    if d == 0: return np.zeros_like(img, dtype=np.uint8)
    return ((img - img.min()) / d * 255).astype(np.uint8)

def render_flm_frame(frame):
    h, w = frame.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    refl, green, blue = norm(frame[:,:,1]), norm(frame[:,:,0]), norm(frame[:,:,2])
    out[:,:,2] = np.clip(refl.astype(int) + blue, 0, 255)
    out[:,:,1] = np.clip(refl.astype(int) + green, 0, 255)
    out[:,:,0] = refl
    return out

def merge_bboxes(bboxes, tem_h_flm, tem_w_flm):
    d = int(max(tem_h_flm, tem_w_flm))
    expanded = [(r0-d, c0-d, r1+d, c1+d) for r0,c0,r1,c1 in bboxes]
    expanded.sort(key=lambda x: x[0])
    merged = [list(expanded[0])]
    for r0,c0,r1,c1 in expanded[1:]:
        pr0,pc0,pr1,pc1 = merged[-1]
        if r0 <= pr1 and c0 <= pc1:
            merged[-1] = [min(pr0,r0), min(pc0,c0), max(pr1,r1), max(pc1,c1)]
        else:
            merged.append([r0,c0,r1,c1])
    return merged

def find_roi_with_origins(img_flm, tem_h, tem_w, flm_px, tem_px, pad_factor=2):
    tem_h_flm = (tem_h * tem_px) / flm_px
    tem_w_flm = (tem_w * tem_px) / flm_px
    pad_y = int(tem_h_flm) * pad_factor
    pad_x = int(tem_w_flm) * pad_factor
    bl_gr = img_flm[:,:,0].astype(float) + img_flm[:,:,2].astype(float)
    mask = (bl_gr > threshold_otsu(bl_gr)).astype(int)
    props = measure.regionprops(measure.label(mask, connectivity=2))
    merged = merge_bboxes([p.bbox for p in props], tem_h_flm, tem_w_flm)
    crops, origins = [], []
    for b in merged:
        r0 = max(0, b[0] - pad_y); c0 = max(0, b[1] - pad_x)
        r1 = min(img_flm.shape[0], b[2] + pad_y); c1 = min(img_flm.shape[1], b[3] + pad_x)
        crops.append(img_flm[r0:r1, c0:c1, 1])
        origins.append((c0, r0))
    return crops, origins

def get_all_rois(flm_stack, tem_h, tem_w, flm_px, tem_px, pad_factor=2):
    all_rois = []
    for fi in range(flm_stack.shape[0]):
        crops, origins = find_roi_with_origins(flm_stack[fi], tem_h, tem_w, flm_px, tem_px, pad_factor)
        for crop, (ox, oy) in zip(crops, origins):
            if crop.max() == crop.min(): continue
            h, w = crop.shape[:2]
            u8 = norm(crop)
            all_rois.append({
                "frame_idx": fi, "origin": (ox, oy),
                "bbox": (ox, oy, ox+w, oy+h),
                "laplacian": cv2.Laplacian(u8, cv2.CV_64F).var(),
                "mean_intensity": crop.mean(), "area": w*h,
            })
    return all_rois

def iou(b1, b2):
    ix0,iy0 = max(b1[0],b2[0]), max(b1[1],b2[1])
    ix1,iy1 = min(b1[2],b2[2]), min(b1[3],b2[3])
    if ix1<=ix0 or iy1<=iy0: return 0.0
    inter = (ix1-ix0)*(iy1-iy0)
    return inter / ((b1[2]-b1[0])*(b1[3]-b1[1]) + (b2[2]-b2[0])*(b2[3]-b2[1]) - inter)

def has_interior_peak(group_rois):
    vals = [r["laplacian"] for r in sorted(group_rois, key=lambda r: r["frame_idx"])]
    pk = np.argmax(vals)
    return 0 < pk < len(vals)-1

def find_best_rois(all_rois, iou_thresh=0.3):
    G = nx.Graph()
    G.add_nodes_from(range(len(all_rois)))
    for i in range(len(all_rois)):
        for j in range(i+1, len(all_rois)):
            if iou(all_rois[i]["bbox"], all_rois[j]["bbox"]) >= iou_thresh:
                G.add_edge(i, j)
    best = []
    for group in nx.connected_components(G):
        best_idx = max(group, key=lambda i: all_rois[i]["laplacian"] * all_rois[i]["mean_intensity"])
        group_rois = [all_rois[i] for i in group]
        if has_interior_peak(group_rois):
            best.append(all_rois[best_idx])
    return best

def get_tile_flm_bbox(flm_h, flm_w, tem_h, tem_w, flm_px, tem_px, tile_scale=1.0):
    th = int((tem_h * tem_px) / flm_px * tile_scale)
    tw = int((tem_w * tem_px) / flm_px * tile_scale)
    sy, sx = th//2, tw//2
    tiles = []
    for y in range(0, flm_h, sy):
        for x in range(0, flm_w, sx):
            if y+th <= flm_h and x+tw <= flm_w:
                tiles.append([x, y, x+tw, y+th])
    return tiles or [[0, 0, tw, th]]

def prepare_tem(tem_path, thresh=130):
    raw = io.imread(str(tem_path))
    if raw.ndim == 3: raw = raw[:,:,0]
    u8 = norm(raw)
    inv = 255 - u8
    return u8, (inv > thresh).astype(np.uint8) * 255

def resize_for_display(img, width=DISPLAY_WIDTH):
    h, w = img.shape[:2]
    scale = width / w
    return cv2.resize(img, (width, int(h * scale))), scale

def draw_points_on_img(img_pil, points, labels):
    img = img_pil.convert("RGB")
    draw = ImageDraw.Draw(img)
    for (x, y), lbl in zip(points, labels):
        color = (60, 220, 100) if lbl == 1 else (220, 60, 60)
        draw.ellipse([(x-7, y-7), (x+7, y+7)], fill=color, outline="white", width=2)
    return img

def run_sam(predictor, img_np, points_display, labels, scale_x, scale_y):
    if img_np.ndim == 2:
        img_np = np.stack([img_np]*3, axis=-1)
    predictor.set_image(img_np)
    coords = np.array([[int(x*scale_x), int(y*scale_y)] for x,y in points_display])
    labs = np.array(labels)
    with torch.inference_mode():
        masks, scores, _ = predictor.predict(point_coords=coords, point_labels=labs, multimask_output=True)
    return masks[np.argmax(scores)]


# ── model loading ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_models():
    import sys
    sys.path.append(str(ROOT_DIR))
    from model_setup import load_sam2_model, load_lightglue_models
    _, predictor = load_sam2_model()
    extractor, matcher = load_lightglue_models()
    return predictor, extractor, matcher

predictor, extractor, matcher = load_models()


# ── session state init ────────────────────────────────────────────────────────

defaults = {
    "flm_stack": None, "tem_raw": None,
    "best_rois": None, "tem_h": None, "tem_w": None,
    "flm_px": 121.0, "tem_px": 6.9,
    "roi_points": [], "roi_labels": [],
    "filtered_bboxes": None, "flm_frame_path": None,
    "upscaled_paths": [],
    "selected_crop_idx": None, "selected_crop_img": None,
    "flm_seg_points": [], "flm_seg_labels": [],
    "flm_mask": None,
    "tem_inv_thresh": None, "tem_u8": None,
    "tem_seg_points": [], "tem_seg_labels": [],
    "tem_mask": None,
    "mk0": None, "mk1": None, "match_fig": None,
    "transform_M": None, "transform_scale": None,
    "overlay": None,
    "last_click_roi": None, "last_click_flm_seg": None, "last_click_tem_seg": None,
    "state_bboxes": [], "state_flm_frame_path": [], "state_selected_bbox_idx": [],
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)


# ── sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🔬 FLM–TEM Alignment")
    st.markdown("---")

    st.markdown("### Files")
    flm_upload = st.file_uploader("FLM Stack (.tif)", type=["tif", "tiff"], key="flm_up")
    tem_upload = st.file_uploader("TEM Image (.tif/.png)", type=["tif", "tiff", "png"], key="tem_up")

    st.markdown("---")
    st.markdown("### Parameters")
    flm_px = st.number_input("FLM pixel (nm)", value=121.0, step=0.5)
    tem_px = st.number_input("TEM pixel (nm)", value=6.9, step=0.1)
    pad_factor = st.number_input("Pad factor", value=1, step=1)
    iou_thresh = st.slider("IOU threshold", 0.1, 0.9, 0.3)
    tile_scale = st.slider("Tile scale", 0.5, 4.0, 2.0)
    tem_thresh = st.slider("TEM invert threshold", 50, 220, 130)

    st.markdown("---")
    st.markdown("### Status")
    def status(label, ok):
        cls = "status-ok" if ok else "status-warn"
        icon = "✓" if ok else "○"
        st.markdown(f'<span class="{cls}">{icon} {label}</span>', unsafe_allow_html=True)

    status("FLM loaded", st.session_state.flm_stack is not None)
    status("TEM loaded", st.session_state.tem_raw is not None)
    status("ROIs detected", st.session_state.best_rois is not None)
    status("Tiles upscaled", len(st.session_state.upscaled_paths) > 0)
    status("FLM mask", st.session_state.flm_mask is not None)
    status("TEM mask", st.session_state.tem_mask is not None)
    status("Matches", st.session_state.mk0 is not None)
    status("Transform", st.session_state.transform_M is not None)

    if st.button("Reset all", type="secondary"):
        for k, v in defaults.items():
            st.session_state[k] = v
        st.rerun()


# ── load files ────────────────────────────────────────────────────────────────

if flm_upload:
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        f.write(flm_upload.read())
        tmp = f.name
    st.session_state.flm_stack = io.imread(tmp)
    os.unlink(tmp)

if tem_upload:
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
        f.write(tem_upload.read())
        tmp = f.name
    u8, inv = prepare_tem(tmp, tem_thresh)
    st.session_state.tem_u8 = u8
    st.session_state.tem_inv_thresh = inv
    st.session_state.tem_raw = u8
    h, w = u8.shape[:2]
    st.session_state.tem_h = h
    st.session_state.tem_w = w
    cv2.imwrite(str(OUTPUT_DIR / "tem.png"), u8)
    cv2.imwrite(str(OUTPUT_DIR / "tem_inv_thresh.png"), inv)
    os.unlink(tmp)


# ── tabs ──────────────────────────────────────────────────────────────────────

flm_tab, tem_tab, match_tab = st.tabs(["FLM Processing", "TEM Processing", "Match & Align"])


# ══════════════════════════════════════════════════════════════════════════════
# FLM TAB
# ══════════════════════════════════════════════════════════════════════════════

with flm_tab:
    flm_steps = st.tabs(["1 · Stack Explorer", "2 · ROI Detection", "3 · Tile Selection", "4 · Segmentation"])

    # ── step 1: stack explorer ────────────────────────────────────────────────
    with flm_steps[0]:
        st.markdown('<div class="step-badge">STEP 1</div>', unsafe_allow_html=True)
        st.markdown("### Stack Explorer")

        if st.session_state.flm_stack is None:
            st.info("Upload an FLM stack in the sidebar.")
        else:
            stack = st.session_state.flm_stack
            n_frames = stack.shape[0]
            st.markdown(f'<div class="metric-box"><span>Dimensions</span><br><strong>{stack.shape}</strong></div>', unsafe_allow_html=True)

            col_ctrl, col_img = st.columns([1, 3])
            with col_ctrl:
                frame_idx = st.slider("Frame (Z)", 0, n_frames-1, n_frames//2)
                channel = st.radio("Channel", ["Composite", "Reflection (1)", "Green (0)", "Blue (2)"])
                show_axes = st.checkbox("Show crosshairs", value=False)

            with col_img:
                frame = stack[frame_idx]
                if channel == "Composite":
                    display = render_flm_frame(frame)
                elif "Reflection" in channel:
                    display = norm(frame[:,:,1])
                elif "Green" in channel:
                    display = norm(frame[:,:,0])
                else:
                    display = norm(frame[:,:,2])

                disp_rgb = Image.fromarray(display if display.ndim == 3 else np.stack([display]*3, -1))
                w_disp, h_disp = disp_rgb.size
                scale = DISPLAY_WIDTH / w_disp
                disp_small = disp_rgb.resize((DISPLAY_WIDTH, int(h_disp * scale)))

                if show_axes:
                    draw = ImageDraw.Draw(disp_small)
                    cx, cy = DISPLAY_WIDTH//2, int(h_disp*scale)//2
                    draw.line([(cx, 0), (cx, int(h_disp*scale))], fill=(255,255,0,120), width=1)
                    draw.line([(0, cy), (DISPLAY_WIDTH, cy)], fill=(255,255,0,120), width=1)

                st.image(disp_small, caption=f"Frame {frame_idx} / {n_frames-1}")

            # depth profile
            st.markdown("#### Laplacian sharpness across stack (reflection channel)")
            if st.button("Compute depth profile"):
                laps = []
                for fi in range(n_frames):
                    u8 = norm(stack[fi, :, :, 1])
                    laps.append(cv2.Laplacian(u8, cv2.CV_64F).var())
                fig, ax = plt.subplots(figsize=(8, 2.5), facecolor="#0d0f12")
                ax.set_facecolor("#0d0f12")
                ax.plot(laps, color="#58a6ff", linewidth=1.5)
                ax.axvline(np.argmax(laps), color="#3fb950", linewidth=1, linestyle="--")
                ax.set_xlabel("Frame", color="#8b949e"); ax.set_ylabel("Laplacian var", color="#8b949e")
                ax.tick_params(colors="#8b949e")
                for spine in ax.spines.values(): spine.set_edgecolor("#21262d")
                plt.tight_layout()
                st.pyplot(fig); plt.close()

    # ── step 2: roi detection ─────────────────────────────────────────────────
    with flm_steps[1]:
        st.markdown('<div class="step-badge">STEP 2</div>', unsafe_allow_html=True)
        st.markdown("### ROI Detection")

        if st.session_state.flm_stack is None or st.session_state.tem_h is None:
            st.info("Load both FLM stack and TEM image first.")
        else:
            if st.button("Detect ROIs", type="primary"):
                with st.spinner("Detecting ROIs across stack..."):
                    stack = st.session_state.flm_stack
                    all_rois = get_all_rois(stack, st.session_state.tem_h, st.session_state.tem_w, flm_px, tem_px, pad_factor)
                    st.session_state.best_rois = find_best_rois(all_rois, iou_thresh)
                    st.session_state.flm_px = flm_px
                    st.session_state.tem_px = tem_px
                    gc.collect()
                st.rerun()

            if st.session_state.best_rois:
                best = st.session_state.best_rois
                st.success(f"Found {len(best)} valid ROIs")

                stack = st.session_state.flm_stack
                h_full, w_full = stack.shape[1:3]

                composite = np.zeros((h_full, w_full, 3), dtype=np.uint8)
                for roi in best:
                    frame = stack[roi["frame_idx"]]
                    x0,y0,x1,y1 = roi["bbox"]
                    crop = frame[y0:y1, x0:x1]
                    composite[y0:y1, x0:x1] = render_flm_frame(crop)

                disp, _ = resize_for_display(composite)
                scale_disp = DISPLAY_WIDTH / w_full

                # draw bounding boxes
                disp_pil = Image.fromarray(disp)
                draw = ImageDraw.Draw(disp_pil)
                for i, roi in enumerate(best):
                    x0,y0,x1,y1 = roi["bbox"]
                    draw.rectangle(
                        [x0*scale_disp, y0*scale_disp, x1*scale_disp, y1*scale_disp],
                        outline=(255,255,255), width=2
                    )
                    draw.text((x0*scale_disp+4, y0*scale_disp+4), f"ROI {i} | f{roi['frame_idx']}", fill=(255,80,80))

                col_img, col_info = st.columns([3, 1])
                with col_img:
                    st.image(disp_pil, caption="Detected ROIs (best frame per group)")
                with col_info:
                    for i, roi in enumerate(best):
                        st.markdown(f"""
                        <div class="metric-box">
                        <span>ROI {i}</span><br>
                        <strong>frame {roi['frame_idx']}</strong><br>
                        <span>lap {roi['laplacian']:.1f}</span>
                        </div>""", unsafe_allow_html=True)

    # ── step 3: tile selection ────────────────────────────────────────────────
    with flm_steps[2]:
        st.markdown('<div class="step-badge">STEP 3</div>', unsafe_allow_html=True)
        st.markdown("### Tile Selection")
        st.markdown(
            "Drop one or more points on the full FLM image. "
            "Each point selects the ROI it falls inside **and** the specific tile within that ROI. "
            "Only tiles that contain a point will be extracted."
        )

        if st.session_state.best_rois is None:
            st.info("Run ROI detection first.")
        else:
            best = st.session_state.best_rois
            stack = st.session_state.flm_stack
            h_full, w_full = stack.shape[1:3]
            scale_disp = DISPLAY_WIDTH / w_full

            # build composite — best frame per ROI region, rest black
            composite = np.zeros((h_full, w_full, 3), dtype=np.uint8)
            for roi in best:
                frame = stack[roi["frame_idx"]]
                x0,y0,x1,y1 = roi["bbox"]
                composite[y0:y1, x0:x1] = render_flm_frame(frame[y0:y1, x0:x1])

            disp, _ = resize_for_display(composite)
            disp_pil = Image.fromarray(disp)
            draw = ImageDraw.Draw(disp_pil)

            # draw ROI bounding boxes
            for i, roi in enumerate(best):
                x0,y0,x1,y1 = roi["bbox"]
                draw.rectangle(
                    [x0*scale_disp, y0*scale_disp, x1*scale_disp, y1*scale_disp],
                    outline=(200,200,200), width=1
                )
                draw.text((x0*scale_disp+4, y0*scale_disp+4), f"ROI {i}", fill=(255,200,0))

            # draw already-placed points
            for px_d, py_d in st.session_state.roi_points:
                draw.ellipse([(px_d-7, py_d-7), (px_d+7, py_d+7)], fill=(255,220,0), outline="white", width=2)

            col_img, col_ctrl = st.columns([3, 1])
            with col_img:
                click = streamlit_image_coordinates(disp_pil, key="roi_click", width=DISPLAY_WIDTH)
                if click and click != st.session_state.last_click_roi:
                    st.session_state.last_click_roi = click
                    st.session_state.roi_points.append([click["x"], click["y"]])
                    st.rerun()

            with col_ctrl:
                if st.session_state.roi_points:
                    # show which ROIs are hit
                    pts_in_roi = defaultdict(list)
                    for px_d, py_d in st.session_state.roi_points:
                        px_full, py_full = px_d / scale_disp, py_d / scale_disp
                        for i, roi in enumerate(best):
                            x0,y0,x1,y1 = roi["bbox"]
                            if x0 <= px_full <= x1 and y0 <= py_full <= y1:
                                pts_in_roi[i].append([px_full, py_full])

                    st.markdown(f'<span class="status-ok">✓ {len(st.session_state.roi_points)} point(s) placed</span>', unsafe_allow_html=True)
                    for roi_idx, pts in pts_in_roi.items():
                        roi = best[roi_idx]
                        st.markdown(f"""<div class="metric-box">
                        <span>ROI {roi_idx} · frame {roi['frame_idx']}</span><br>
                        <strong>{len(pts)} point(s)</strong>
                        </div>""", unsafe_allow_html=True)

                    if not pts_in_roi:
                        st.warning("No points landed inside any ROI.")

                if st.button("Undo last point") and st.session_state.roi_points:
                    st.session_state.roi_points.pop()
                    st.rerun()

                if st.button("Clear points"):
                    st.session_state.roi_points = []
                    st.rerun()

                if st.button("Extract & Upscale Tiles", type="primary"):
                    pts_in_roi = defaultdict(list)
                    for px_d, py_d in st.session_state.roi_points:
                        px_full, py_full = px_d / scale_disp, py_d / scale_disp
                        for i, roi in enumerate(best):
                            x0,y0,x1,y1 = roi["bbox"]
                            if x0 <= px_full <= x1 and y0 <= py_full <= y1:
                                pts_in_roi[i].append([px_full, py_full])

                    if not pts_in_roi:
                        st.warning("Place at least one point inside an ROI.")
                    else:
                        with st.spinner("Extracting and upscaling tiles..."):
                            import sys; sys.path.append(str(ROOT_DIR))
                            from model_setup import upscale_and_save
                            for f in FF_BB_DIR.glob("*.png"): f.unlink()
                            for f in UPSCALED_DIR.glob("*.png"): f.unlink()

                            crop_idx = 0
                            all_bboxes = []
                            flm_frame_paths = []

                            for roi_idx, user_pts in pts_in_roi.items():
                                roi = best[roi_idx]
                                x0,y0,x1,y1 = roi["bbox"]
                                ox, oy = roi["origin"]
                                flm_frame = stack[roi["frame_idx"], :, :, 1]
                                flm_u8 = norm(flm_frame)
                                flm_eq = (exposure.equalize_hist(flm_u8) * 255).astype(np.uint8)

                                frame_path = OUTPUT_DIR / f"flm_frame_{roi['frame_idx']}.png"
                                cv2.imwrite(str(frame_path), flm_u8)
                                flm_frame_paths.append(str(frame_path))

                                # generate all tiles in ROI-local coords, then offset to absolute
                                tiles = get_tile_flm_bbox(
                                    y1-y0, x1-x0,
                                    st.session_state.tem_h, st.session_state.tem_w,
                                    flm_px, tem_px, tile_scale
                                )

                                roi_bboxes = []
                                for tx0,ty0,tx1,ty1 in tiles:
                                    # convert to absolute image coordinates
                                    abs_x0 = tx0 + ox; abs_x1 = tx1 + ox
                                    abs_y0 = ty0 + oy; abs_y1 = ty1 + oy
                                    # keep only tiles that contain at least one user point
                                    for [px_full, py_full] in user_pts:
                                        if abs_x0 < px_full < abs_x1 and abs_y0 < py_full < abs_y1:
                                            crop = flm_eq[abs_y0:abs_y1, abs_x0:abs_x1]
                                            crop3 = np.stack([crop]*3, -1)
                                            cv2.imwrite(str(FF_BB_DIR / f"{str(crop_idx).zfill(4)}.png"), crop3)
                                            roi_bboxes.append([abs_x0, abs_y0, abs_x1, abs_y1])
                                            crop_idx += 1
                                            break  # one match per tile is enough

                                all_bboxes.append(roi_bboxes)

                            st.session_state.state_bboxes = all_bboxes
                            st.session_state.state_flm_frame_path = flm_frame_paths

                            upscale_and_save(FF_BB_DIR, UPSCALED_DIR)
                            st.session_state.upscaled_paths = sorted(UPSCALED_DIR.glob("*.png"))

                        st.success(f"Extracted {crop_idx} tiles, upscaled 4×")
                        st.rerun()

    # ── step 4: flm segmentation ──────────────────────────────────────────────
    with flm_steps[3]:
        st.markdown('<div class="step-badge">STEP 4</div>', unsafe_allow_html=True)
        st.markdown("### FLM Segmentation")
        st.markdown("Click to select a tile, then add SAM2 points and segment.")

        if not st.session_state.upscaled_paths:
            st.info("Extract tiles in Step 3 first.")
        else:
            img_paths = st.session_state.upscaled_paths
            images = [io.imread(str(p)) for p in img_paths]
            ih, iw = images[0].shape[:2]
            rows = int(np.ceil(len(images) / COLS))

            canvas_h = rows * ih + (rows+1) * PADDING
            canvas_w = COLS * iw + (COLS+1) * PADDING
            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            for i, img in enumerate(images):
                r, c = divmod(i, COLS)
                y0 = PADDING + r*(ih+PADDING); x0 = PADDING + c*(iw+PADDING)
                canvas[y0:y0+ih, x0:x0+iw] = img if img.ndim==3 else np.stack([img]*3,-1)

            if st.session_state.selected_crop_idx is None:
                st.markdown("**Select a tile by clicking:**")
                grid_scale = DISPLAY_WIDTH / canvas_w
                grid_disp = cv2.resize(canvas, (DISPLAY_WIDTH, int(canvas_h * grid_scale)))
                click = streamlit_image_coordinates(Image.fromarray(grid_disp), key="grid_click", width=DISPLAY_WIDTH)
                if click and click != st.session_state.last_click_flm_seg:
                    st.session_state.last_click_flm_seg = click
                    px, py = click["x"] / grid_scale, click["y"] / grid_scale
                    col = int((px - PADDING) // (iw + PADDING))
                    row = int((py - PADDING) // (ih + PADDING))
                    x0c = PADDING + col*(iw+PADDING); y0c = PADDING + row*(ih+PADDING)
                    if x0c <= px < x0c+iw and y0c <= py < y0c+ih:
                        tidx = row*COLS + col
                        if tidx < len(images):
                            st.session_state.selected_crop_idx = tidx
                            st.session_state.selected_crop_img = images[tidx]
                            st.session_state.flm_seg_points = []
                            st.session_state.flm_seg_labels = []
                            st.rerun()
            else:
                tidx = st.session_state.selected_crop_idx
                sel_img = st.session_state.selected_crop_img
                orig_h, orig_w = sel_img.shape[:2]
                disp_h_sel = int(orig_h * (DISPLAY_WIDTH / orig_w))
                scale_x = orig_w / DISPLAY_WIDTH
                scale_y = orig_h / disp_h_sel
                sel_pil = Image.fromarray(sel_img if sel_img.ndim==3 else np.stack([sel_img]*3,-1))
                sel_disp = sel_pil.resize((DISPLAY_WIDTH, disp_h_sel))

                col_img, col_ctrl = st.columns([3, 1])
                with col_ctrl:
                    st.markdown(f'<div class="metric-box"><span>Tile</span><br><strong>{tidx}</strong></div>', unsafe_allow_html=True)
                    seg_mode = st.radio("Point", ["Add ✓", "Remove ✗"], key="flm_seg_mode")
                    if st.button("Undo", key="flm_undo") and st.session_state.flm_seg_points:
                        st.session_state.flm_seg_points.pop()
                        st.session_state.flm_seg_labels.pop()
                        st.rerun()
                    if st.button("Clear points", key="flm_clr"):
                        st.session_state.flm_seg_points = []
                        st.session_state.flm_seg_labels = []
                        st.rerun()
                    if st.button("↩ Back to grid", key="flm_back"):
                        st.session_state.selected_crop_idx = None
                        st.rerun()
                    if st.button("Segment", type="primary", key="flm_seg_btn"):
                        if not st.session_state.flm_seg_points:
                            st.warning("Add at least one point.")
                        else:
                            with st.spinner("Running SAM2..."):
                                img_np = sel_img if sel_img.ndim==3 else np.stack([sel_img]*3,-1)
                                # apply threshold like napari widget
                                img_thresh = (img_np > 130).astype(np.uint8) * 255
                                mask = run_sam(predictor, img_thresh,
                                               st.session_state.flm_seg_points,
                                               st.session_state.flm_seg_labels,
                                               scale_x, scale_y)
                                st.session_state.flm_mask = mask
                                cv2.imwrite(str(SEG_DIR / "flm_seg_mask.png"), mask.astype(np.uint8)*255)
                            st.rerun()

                with col_img:
                    display_img = draw_points_on_img(sel_disp, st.session_state.flm_seg_points, st.session_state.flm_seg_labels)
                    click = streamlit_image_coordinates(display_img, key="flm_seg_click", width=DISPLAY_WIDTH)
                    if click and click != st.session_state.last_click_flm_seg:
                        st.session_state.last_click_flm_seg = click
                        lbl = 1 if "Add" in seg_mode else 0
                        st.session_state.flm_seg_points.append([click["x"], click["y"]])
                        st.session_state.flm_seg_labels.append(lbl)
                        st.rerun()

                    if st.session_state.flm_mask is not None:
                        st.markdown("**Mask:**")
                        st.image(st.session_state.flm_mask.astype(np.uint8)*255, clamp=True)


# ══════════════════════════════════════════════════════════════════════════════
# TEM TAB
# ══════════════════════════════════════════════════════════════════════════════

with tem_tab:
    st.markdown("### TEM Segmentation")

    if st.session_state.tem_inv_thresh is None:
        st.info("Upload a TEM image in the sidebar.")
    else:
        inv = st.session_state.tem_inv_thresh
        orig_h, orig_w = inv.shape[:2]
        disp_h = int(orig_h * (DISPLAY_WIDTH / orig_w))
        scale_x = orig_w / DISPLAY_WIDTH
        scale_y = orig_h / disp_h

        inv_rgb = np.stack([inv]*3, -1) if inv.ndim==2 else inv
        inv_pil = Image.fromarray(inv_rgb).resize((DISPLAY_WIDTH, disp_h))

        col_img, col_ctrl = st.columns([3, 1])
        with col_ctrl:
            st.markdown(f'<div class="metric-box"><span>TEM size</span><br><strong>{orig_w}×{orig_h}</strong></div>', unsafe_allow_html=True)
            tem_mode = st.radio("Point", ["Add ✓", "Remove ✗"], key="tem_mode")
            if st.button("Undo", key="tem_undo") and st.session_state.tem_seg_points:
                st.session_state.tem_seg_points.pop()
                st.session_state.tem_seg_labels.pop()
                st.rerun()
            if st.button("Clear", key="tem_clr"):
                st.session_state.tem_seg_points = []
                st.session_state.tem_seg_labels = []
                st.session_state.tem_mask = None
                st.rerun()
            if st.button("Segment TEM", type="primary"):
                if not st.session_state.tem_seg_points:
                    st.warning("Add at least one point.")
                else:
                    with st.spinner("Running SAM2..."):
                        img_np = inv_rgb
                        mask = run_sam(predictor, img_np,
                                       st.session_state.tem_seg_points,
                                       st.session_state.tem_seg_labels,
                                       scale_x, scale_y)
                        st.session_state.tem_mask = mask
                        cv2.imwrite(str(SEG_DIR / "tem_seg_mask.png"), mask.astype(np.uint8)*255)
                    st.rerun()

        with col_img:
            display_img = draw_points_on_img(inv_pil, st.session_state.tem_seg_points, st.session_state.tem_seg_labels)
            click = streamlit_image_coordinates(display_img, key="tem_seg_click", width=DISPLAY_WIDTH)
            if click and click != st.session_state.last_click_tem_seg:
                st.session_state.last_click_tem_seg = click
                lbl = 1 if "Add" in tem_mode else 0
                st.session_state.tem_seg_points.append([click["x"], click["y"]])
                st.session_state.tem_seg_labels.append(lbl)
                st.rerun()

            if st.session_state.tem_mask is not None:
                st.markdown("**Mask:**")
                st.image(st.session_state.tem_mask.astype(np.uint8)*255, clamp=True)


# ══════════════════════════════════════════════════════════════════════════════
# MATCH TAB
# ══════════════════════════════════════════════════════════════════════════════

with match_tab:
    st.markdown("### Keypoint Matching & Alignment")

    if st.session_state.flm_mask is None or st.session_state.tem_mask is None:
        st.info("Complete FLM and TEM segmentation first.")
    else:
        import sys; sys.path.append(str(ROOT_DIR))
        from model_setup import create_tensor_from_mask, get_keypoint_matches, estimate_transform, apply_transform_overlay
        from lightglue import viz2d

        col1, col2, col3 = st.columns(3)
        thresh = st.slider("Match threshold", 0.0, 0.2, 0.02, 0.005)

        with col1:
            if st.button("Match Keypoints", type="primary"):
                with st.spinner("Matching..."):
                    t0 = create_tensor_from_mask(st.session_state.flm_mask, DEVICE)
                    t1t = create_tensor_from_mask(st.session_state.tem_mask, DEVICE)
                    _, _, mk0, mk1, _ = get_keypoint_matches(extractor, matcher, t0, t1t, thresh)
                    st.session_state.mk0 = mk0.cpu().numpy()
                    st.session_state.mk1 = mk1.cpu().numpy()

                    plt.close("all")
                    viz2d.plot_images([t0[0][0].cpu(), t1t[0][0].cpu()])
                    viz2d.plot_matches(mk0, mk1, color="lime", lw=0.2)
                    fig = plt.gcf()
                    fig.canvas.draw()
                    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:,:,:3]
                    st.session_state.match_fig = buf
                    plt.close(fig)
                st.rerun()

        with col2:
            if st.button("Estimate Transform") and st.session_state.mk0 is not None:
                with st.spinner("Estimating..."):
                    mk0 = st.session_state.mk0.copy()
                    mk1 = st.session_state.mk1.copy()

                    if st.session_state.state_bboxes and st.session_state.state_selected_bbox_idx:
                        bbox_idx = st.session_state.state_selected_bbox_idx[0]
                        bboxes = st.session_state.state_bboxes[0]
                        bx0, by0, _, _ = bboxes[bbox_idx]
                        mk0 = mk0 / 4
                        mk0[:, 0] += bx0
                        mk0[:, 1] += by0

                    M, inliers, scale = estimate_transform(mk0, mk1)
                    st.session_state.transform_M = M
                    st.session_state.transform_scale = scale

                    if M is not None:
                        flm_img = cv2.imread(str(st.session_state.state_flm_frame_path[0])) if st.session_state.state_flm_frame_path else None
                        tem_img = st.session_state.tem_u8

                        if flm_img is not None:
                            if flm_img.ndim == 3: flm_img = cv2.cvtColor(flm_img, cv2.COLOR_BGR2GRAY)
                            overlay, _, _, _ = apply_transform_overlay(flm_img, tem_img, M)
                            st.session_state.overlay = overlay
                st.rerun()

        with col3:
            if st.session_state.transform_M is not None:
                M = st.session_state.transform_M
                s = st.session_state.transform_scale
                st.markdown(f"""<div class="metric-box">
                <span>Scale</span><br><strong>{s:.5f}</strong>
                </div>""", unsafe_allow_html=True)

        if st.session_state.match_fig is not None:
            st.markdown("#### Keypoint matches")
            st.markdown(f'<span class="status-ok">✓ {len(st.session_state.mk0)} matches found</span>', unsafe_allow_html=True)
            st.image(st.session_state.match_fig)

        if st.session_state.transform_M is not None:
            M = st.session_state.transform_M
            st.markdown("#### Transform matrix")
            st.code(f"{M[0,0]:.5f}  {M[0,1]:.5f}  {M[0,2]:.5f}\n{M[1,0]:.5f}  {M[1,1]:.5f}  {M[1,2]:.5f}", language=None)

        if st.session_state.overlay is not None:
            st.markdown("#### Aligned overlay")
            tem_img = st.session_state.tem_u8
            overlay = st.session_state.overlay

            fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0d0f12")
            for ax in axes:
                ax.set_facecolor("#0d0f12")
                ax.tick_params(colors="#8b949e")
                for sp in ax.spines.values(): sp.set_edgecolor("#21262d")
            axes[0].imshow(tem_img, cmap="gray"); axes[0].set_title("TEM", color="#c9d1d9"); axes[0].axis("off")
            axes[1].imshow(overlay); axes[1].set_title("Overlay (Gray=TEM, Green=FLM)", color="#c9d1d9"); axes[1].axis("off")
            plt.tight_layout()
            st.pyplot(fig); plt.close()