import os
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from scipy.ndimage import gaussian_filter

def detect_grid_lines(img, sigma=3, threshold=0.2, min_distance=50, min_angle=10, num_peaks=5):
    """Detect straight lines using Canny edge + Hough transform."""
    # Smooth to reduce noise
    if len(img.shape) == 3: img = img.mean(axis=2).astype(np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-12)
    smoothed = gaussian_filter(img_norm, sigma=sigma)
    
    # Detect edges
    edges = canny(
        smoothed, low_threshold=threshold*0.5, high_threshold=threshold
    ).astype(int)

    # Hough transform
    h, theta, d = hough_line(edges)
    
    # Find peaks in Hough space
    lines = []
    for _, angle_val, dist_val in zip(*hough_line_peaks(h, theta, d,
                                                        min_distance=min_distance,
                                                        min_angle=min_angle,
                                                        threshold=0.4 * h.max(),
                                                        num_peaks=num_peaks)):
        lines.append((angle_val, dist_val))
    return lines, edges

def overlay_hough_lines(img, lines, value=255):
    """
    Draw Hough lines directly into a numpy image.

    img   : 2D numpy array (grayscale)
    lines : list of (theta, d)
    value : pixel value for lines

    returns: new numpy array with lines overlaid
    """
    out = img.copy()
    h, w = out.shape

    for theta, d in lines:
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        points = []

        # Intersections with image borders
        if abs(sin_t) > 1e-6:
            y = d / sin_t
            if 0 <= y < h:
                points.append((0, y))
            y = (d - (w - 1) * cos_t) / sin_t
            if 0 <= y < h:
                points.append((w - 1, y))

        if abs(cos_t) > 1e-6:
            x = d / cos_t
            if 0 <= x < w:
                points.append((x, 0))
            x = (d - (h - 1) * sin_t) / cos_t
            if 0 <= x < w:
                points.append((x, h - 1))

        if len(points) >= 2:
            (x0, y0), (x1, y1) = points[:2]
            x0, y0, x1, y1 = map(int, [x0, y0, x1, y1])

            # Bresenham-style rasterization
            rr = np.linspace(y0, y1, 1000).astype(int)
            cc = np.linspace(x0, x1, 1000).astype(int)
            valid = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
            out[rr[valid], cc[valid]] = value

    return out

def visualize_detected_lines(img, lines, title="Detected Lines"):
    """Visualize Hough lines overlaid on image and save as is."""

    fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)  # Preserve top-to-bottom orientation
    for angle, dist in lines:
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        points = []

        # Intersections with image boundaries
        h, w = img.shape
        edges = [
            (0, 0, w, 0),         # top
            (w, 0, w, h),         # right
            (w, h, 0, h),         # bottom
            (0, h, 0, 0)          # left
        ]

        for x1, y1, x2, y2 in edges:
            # Line equation: dx = x2 - x1, dy = y2 - y1
            dx, dy = x2 - x1, y2 - y1
            # Parametric form: P = (x1 + t*dx, y1 + t*dy)
            # Solve for intersection with: x*cos_a + y*sin_a = dist
            denom = cos_a * dx + sin_a * dy
            if abs(denom) > 1e-6:
                t = (dist - cos_a * x1 - sin_a * y1) / denom
                if 0 <= t <= 1:
                    x = x1 + t * dx
                    y = y1 + t * dy
                    points.append((x, y))

        if len(points) >= 2:
            p0, p1 = points[0], points[1]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r-', linewidth=1)

    # Save exactly as displayed — no resizing, no auto-scaling
    plt.tight_layout(pad=0)
    plt.savefig(f'lines_{title}.png', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualize_edges_only(img_path, sigma_range=[1, 2, 3], threshold_range=[0.1, 0.2, 0.3]):
    """Visualize ONLY edge detection results, no Hough lines."""
    os.makedirs("edge_tests", exist_ok=True)
    
    img = io.imread(img_path)
    if len(img.shape) == 3: img = img.mean(axis=2).astype(np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-12)
    
    for sigma in sigma_range:
        for thresh in threshold_range:
            smoothed = gaussian_filter(img_norm, sigma=sigma)
            edges = canny(smoothed, low_threshold=thresh*0.5, high_threshold=thresh)
            
            plt.figure(figsize=(10, 10))
            plt.imshow(img_norm, cmap='gray')
            plt.contour(edges, colors='red', linewidths=1, levels=[0.5])
            plt.title(f'σ={sigma}, threshold={thresh} (edges=red)')
            plt.axis('off')
            
            plt.savefig(f'edge_tests/edges_σ{sigma}_t{thresh}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    print("Edge detection saved to edge_tests/")