import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
from streamlit_image_coordinates import streamlit_image_coordinates
from model_setup import load_lightglue_models, load_sam2_model
from lightglue.utils import rbd
from lightglue import viz2d

st.set_page_config(layout="wide")
st.title("SAM2 + LightGlue")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISPLAY_WIDTH = 640

@st.cache_resource
def load_all_tools():
    extractor, matcher = load_lightglue_models()
    sam2_model, predictor = load_sam2_model()
    return extractor, matcher, predictor

extractor, matcher, predictor = load_all_tools()

for i in ["1", "2"]:
    for key in ["points", "labels", "mask", "img_pil", "last_click"]:
        st.session_state.setdefault(f"{key}_{i}", [] if "points" in key or "labels" in key else None)

def create_tensor_from_mask(mask):
    # Mask is 2D bool/uint8, expand to 3-channel float tensor
    if mask.ndim == 2:
        mask = np.stack([mask, mask, mask], axis=-1)
    # HWC -> CHW
    image = np.transpose(mask, (2, 0, 1))
    return torch.from_numpy(image).float().unsqueeze(0).to(DEVICE)

def get_keypoint_matches(image0, image1, conf_thresh):
    feats0 = extractor.extract(image0)
    feats1 = extractor.extract(image1)

    # Update threshold on existing matcher
    matcher.conf.filter_threshold = conf_thresh

    matches01 = matcher({
        "image0": feats0,
        "image1": feats1,
    })

    feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    return kpts0, kpts1, kpts0[matches[..., 0]], kpts1[matches[..., 1]], matches01

def draw_points(base_img, img_id):
    img = base_img.convert("RGB")
    draw = ImageDraw.Draw(img)
    for (dx, dy), lbl in zip(st.session_state[f"points_{img_id}"], st.session_state[f"labels_{img_id}"]):
        color = (0, 220, 0) if lbl == 1 else (220, 0, 0)
        draw.ellipse([(dx-8, dy-8), (dx+8, dy+8)], fill=color, outline="white", width=2)
    return img

def run_sam_logic(img_id):
    if not st.session_state[f"points_{img_id}"]:
        st.warning("No points")
        return

    img_pil = st.session_state[f"img_pil_{img_id}"]
    img_np = np.array(img_pil)

    # Handle grayscale
    if img_np.ndim == 2:
        img_np = np.stack([img_np, img_np, img_np], axis=-1)

    orig_w, orig_h = img_pil.size
    display_h = int(orig_h * (DISPLAY_WIDTH / orig_w))
    scale_x = orig_w / DISPLAY_WIDTH
    scale_y = orig_h / display_h

    predictor.set_image(img_np)

    # Scale display coords to original image coords
    coords = np.array([
        [int(x * scale_x), int(y * scale_y)]
        for x, y in st.session_state[f"points_{img_id}"]
    ])
    labs = np.array(st.session_state[f"labels_{img_id}"])

    with torch.inference_mode():
        masks, scores, _ = predictor.predict(
            point_coords=coords,
            point_labels=labs,
            multimask_output=True
        )
    st.session_state[f"mask_{img_id}"] = masks[np.argmax(scores)]

t1, t2, t3 = st.tabs(["Image 1", "Image 2", "Match"])

for i in ["1", "2"]:
    with (t1 if i == "1" else t2):
        up = st.file_uploader(f"Img {i}", type=["jpg", "png"], key=f"up_{i}")
        if up:
            # Force RGB on load
            img = Image.open(up)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            st.session_state[f"img_pil_{i}"] = img

            w, h = img.size
            disp_h = int(h * (DISPLAY_WIDTH / w))
            disp = img.resize((DISPLAY_WIDTH, disp_h))

            c1, c2 = st.columns([1, 3])
            with c1:
                mode = st.radio(f"Mode {i}", ["Add", "Remove", "Delete"], key=f"m_{i}")
                if st.button(f"Segment {i}", key=f"s_{i}"):
                    run_sam_logic(i)
                    st.rerun()

                if st.button(f"Undo {i}", key=f"u_{i}") and st.session_state[f"points_{i}"]:
                    st.session_state[f"points_{i}"].pop()
                    st.session_state[f"labels_{i}"].pop()
                    st.rerun()

                if st.button(f"Clear {i}", key=f"c_{i}"):
                    st.session_state[f"points_{i}"] = []
                    st.session_state[f"labels_{i}"] = []
                    st.session_state[f"mask_{i}"] = None
                    st.rerun()

            with c2:
                click = streamlit_image_coordinates(
                    draw_points(disp, i),
                    key=f"cl_{i}",
                    width=DISPLAY_WIDTH
                )

                if click and click != st.session_state[f"last_click_{i}"]:
                    st.session_state[f"last_click_{i}"] = click
                    cx, cy = click["x"], click["y"]

                    if mode == "Delete" and st.session_state[f"points_{i}"]:
                        dists = [(cx-x)**2 + (cy-y)**2 for x, y in st.session_state[f"points_{i}"]]
                        idx = np.argmin(dists)
                        st.session_state[f"points_{i}"].pop(idx)
                        st.session_state[f"labels_{i}"].pop(idx)
                        st.rerun()
                    else:
                        st.session_state[f"points_{i}"].append([cx, cy])
                        st.session_state[f"labels_{i}"].append(1 if mode == "Add" else 0)
                        st.rerun()

                if st.session_state[f"mask_{i}"] is not None:
                    st.image(st.session_state[f"mask_{i}"], clamp=True, caption="Mask")

# Add after existing imports at top
import cv2

# Add these functions before the Streamlit code

def estimate_transform(kpts0, kpts1):
    """Estimate affine transformation from kpts0 to kpts1"""
    M, inliers = cv2.estimateAffinePartial2D(
        kpts0.astype(np.float32),
        kpts1.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    )

    if M is None:
        return None, None, 0

    # Extract scale from transformation matrix
    scale = np.sqrt(M[0,0]**2 + M[0,1]**2)

    return M, inliers, scale

def apply_transform_overlay(img_source, img_target, M, alpha=0.5):
    h, w = img_target.shape[:2]
    warped = cv2.warpAffine(img_source, M, (w, h), flags=cv2.INTER_LINEAR)

    # normalize both to uint8
    if img_target.max() > 0:
        target_norm = (img_target.astype(float) / img_target.max() * 255).astype(np.uint8)
    else:
        target_norm = img_target.astype(np.uint8)

    if warped.max() > 0:
        warped_norm = (warped.astype(float) / warped.max() * 255).astype(np.uint8)
    else:
        warped_norm = warped.astype(np.uint8)

    # overlay: target as gray, warped source as green
    overlay = np.zeros((h, w, 3), dtype=np.uint8)
    overlay[:, :, 0] = target_norm   # R = target gray
    overlay[:, :, 1] = np.maximum(target_norm, warped_norm)  # G = both (gray + green)
    overlay[:, :, 2] = target_norm   # B = target gray

    # add warped source purely into green channel
    overlay[:, :, 1] = np.clip(overlay[:, :, 1].astype(int) + warped_norm.astype(int), 0, 255).astype(np.uint8)

    return overlay, warped, target_norm, warped_norm

# Update the Match tab
with t3:
    if st.session_state.mask_1 is not None and st.session_state.mask_2 is not None:
        thresh = st.slider("Filter Threshold", 0.0, 0.2, 0.05, 0.01)

        col1, col2 = st.columns(2)
        with col1:
            run_match = st.button("Match Keypoints")
        with col2:
            run_transform = st.button("Estimate Transform")

        if run_match:
            t0 = create_tensor_from_mask(st.session_state.mask_1)
            t1t = create_tensor_from_mask(st.session_state.mask_2)

            _, _, mk0, mk1, m01 = get_keypoint_matches(t0, t1t, thresh)

            # Store matches in session state
            st.session_state['matched_kpts_1'] = mk0.cpu().numpy()
            st.session_state['matched_kpts_2'] = mk1.cpu().numpy()

            st.write(f"Found {len(mk0)} matches")

            # Store the match figure
            plt.close('all')
            viz2d.plot_images([t0[0][0].cpu(), t1t[0][0].cpu()])
            viz2d.plot_matches(mk0.cpu(), mk1.cpu(), color="lime", lw=0.2)
            st.session_state['match_fig'] = plt.gcf()

        # Always show matches if they exist
        if 'match_fig' in st.session_state:
            st.subheader("Keypoint Matches")
            st.pyplot(st.session_state['match_fig'])

        if run_transform and 'matched_kpts_1' in st.session_state:
            kpts1 = st.session_state['matched_kpts_1']
            kpts2 = st.session_state['matched_kpts_2']

            M, inliers, scale = estimate_transform(kpts1, kpts2)

            if M is None:
                st.error("Failed to estimate transform")
            else:
                st.success(f"Transform estimated with {np.sum(inliers)} inliers")
                st.write(f"**Scale factor:** {scale:.5f}")
                st.write("**Transformation Matrix:**")
                st.code(f"{M}")

                # Store transform
                st.session_state['transform_M'] = M
                st.session_state['transform_scale'] = scale

        # Always show transform overlay if it exists
        if 'transform_M' in st.session_state:
            st.subheader("Aligned Overlay")

            M = st.session_state['transform_M']
            scale = st.session_state['transform_scale']

            # Get original images (not masks)
            img1 = np.array(st.session_state.img_pil_1.convert('L'))
            img2 = np.array(st.session_state.img_pil_2.convert('L'))

            # Apply transform and create overlay
            alpha_blend = st.slider("Overlay transparency", 0.0, 1.0, 0.5, 0.05)
            overlay, warped, target_norm, warped_norm = apply_transform_overlay(img1, img2, M)

                # Display results
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            axes[0].imshow(img2, cmap='gray')
            axes[0].set_title("Target (Image 2)")
            axes[0].axis('off')

            axes[1].imshow(warped, cmap='gray')
            axes[1].set_title(f"Warped Source (scale={scale:.4f})")
            axes[1].axis('off')

            axes[2].imshow(overlay)
            axes[2].set_title("Overlay (Gray=Target, Green=Source)")
            axes[2].axis('off')

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("Need both masks")
