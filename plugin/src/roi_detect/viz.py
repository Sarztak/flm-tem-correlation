import math
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def crops_for_group(all_rois, group, rendered_stack):
    crops = []
    for roi_idx in sorted(group):
        roi = all_rois[roi_idx]
        fi = int(roi["frame_idx"])
        x0, y0, x1, y1 = map(int, roi["bbox"])
        rendered = rendered_stack[fi]
        h, w = rendered.shape[:2]
        x0 = max(0, min(x0, w - 1))
        x1 = max(0, min(x1, w))
        y0 = max(0, min(y0, h - 1))
        y1 = max(0, min(y1, h))
        if x1 <= x0 or y1 <= y0:
            continue
        crops.append((roi_idx, fi, rendered[y0:y1, x0:x1].copy()))
    return crops


def plot_group_crops(crops, max_cols=4, figsize_per=(3, 3), cmap=None):
    n = len(crops)
    if n == 0:
        return
    cols = min(max_cols, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * figsize_per[0], rows * figsize_per[1]))
    axes = np.array(axes).reshape(-1)
    for ax in axes[n:]:
        ax.axis('off')
    for i, (roi_idx, fi, crop) in enumerate(crops):
        ax = axes[i]
        if crop.ndim == 3 and crop.shape[2] in (1, 3):
            ax.imshow(crop.squeeze())
        else:
            ax.imshow(crop, cmap=cmap)
        ax.set_title(f'roi {roi_idx}\nframe {fi}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_groups_laplacian(groups_vals, figsize=(10, 6), cols=3, sharex=True, sharey=False):
    n = len(groups_vals)
    if n == 0:
        return None, []
    cols = min(cols, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False, sharex=sharex, sharey=sharey)
    axes = axes.flatten()
    for i, vals in enumerate(groups_vals):
        ax = axes[i]
        if vals is None or len(vals) == 0:
            ax.set_visible(False)
            continue
        ax.plot(range(len(vals)), vals, marker='o', linestyle='-')
        ax.set_title(f'Group {i} (n={len(vals)})')
        ax.set_xlabel('frame_idx')
        ax.set_ylabel('laplacian')
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()


def plot_bbox_area_hist(bboxes, bins=50, log_scale=False, figsize=(8, 4)):
    areas = np.array([(y1 - y0) * (x1 - x0) for (y0, x0, y1, x1) in bboxes])
    plt.figure(figsize=figsize)
    plt.hist(areas, bins=bins, color='C0', edgecolor='k', alpha=0.8)
    plt.xlabel('Area (pixels)')
    plt.ylabel('Count (log scale)' if log_scale else 'Count')
    if log_scale:
        plt.yscale('log')
    plt.title(f'Bounding box area distribution (n={len(areas)})')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    return areas


def bbox_center(b):
    y0, x0, y1, x1 = b
    return ((y0 + y1) // 2, (x0 + x1) // 2)


def show_zoom(fl_image, roi_mask, bboxes, target_idx, pad=60, cmap='gray'):
    H, W = roi_mask.shape
    y0, x0, y1, x1 = bboxes[target_idx]
    r0, c0 = max(0, y0 - pad), max(0, x0 - pad)
    r1, c1 = min(H, y1 + pad), min(W, x1 + pad)

    inside_idxs = [
        i for i, b in enumerate(bboxes)
        if r0 <= bbox_center(b)[0] < r1 and c0 <= bbox_center(b)[1] < c1
    ]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    if fl_image.ndim == 3 and fl_image.shape[2] > 1:
        ax.imshow(fl_image[r0:r1, c0:c1, 1], cmap=cmap)
    else:
        ax.imshow(fl_image[r0:r1, c0:c1], cmap=cmap)
    ax.imshow(roi_mask[r0:r1, c0:c1], cmap='Reds', alpha=0.25)

    for i in inside_idxs:
        by0, bx0, by1, bx1 = bboxes[i]
        rect = patches.Rectangle(
            (bx0 - c0, by0 - r0), bx1 - bx0, by1 - by0,
            linewidth=1.2,
            edgecolor='yellow' if i == target_idx else 'lime',
            facecolor='none',
        )
        ax.add_patch(rect)
        ax.text(bx0 - c0 + 2, by0 - r0 + 10, f'{i}', color='white',
                fontsize=8, bbox=dict(facecolor='black', alpha=0.6, pad=1))

    ax.set_title(f'Zoom around bbox {target_idx} ({len(inside_idxs)} boxes inside crop)')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


def plot_bbox_over_roi_mask(roi_mask, bboxes):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(roi_mask)
    for idx, (y0, x0, y1, x1) in enumerate(bboxes):
        rect = patches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                  linewidth=1.5, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x0 + 2, y0 + 12, str(idx), color='yellow', fontsize=10,
                bbox=dict(facecolor='black', alpha=0.5, pad=1))
    ax.set_title(f'{len(bboxes)} bboxes on roi_mask')
    ax.axis('off')


def show_anns(anns, borders=True):
    np.random.seed(3)
    if not anns:
        return
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((*sorted_anns[0]['segmentation'].shape, 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1)
    ax.imshow(img)
