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

def visualize_detected_lines(img, lines, title="Detected Lines"):
    """Visualize Hough lines that intersect the image."""

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

    h, w = img.shape
    lines_drawn = 0

    for angle, dist in lines:
        # Normal vector
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        # Find intersections with image boundaries
        points = []

        # Left edge: x = 0
        if abs(sin_a) > 1e-6:
            y = (dist - 0 * cos_a) / sin_a
            if 0 <= y <= h - 1:
                points.append((0, y))

        # Right edge: x = w - 1
        if abs(sin_a) > 1e-6:
            y = (dist - (w - 1) * cos_a) / sin_a
            if 0 <= y <= h - 1:
                points.append((w - 1, y))

        # Top edge: y = 0
        if abs(cos_a) > 1e-6:
            x = (dist - 0 * sin_a) / cos_a
            if 0 <= x <= w - 1:
                points.append((x, 0))

        # Bottom edge: y = h - 1
        if abs(cos_a) > 1e-6:
            x = (dist - (h - 1) * sin_a) / cos_a
            if 0 <= x <= w - 1:
                points.append((x, h - 1))

        # Draw if at least two distinct intersection points
        if len(points) >= 2:
            p0, p1 = points[0], points[1]
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], 'r-', linewidth=1)
            lines_drawn += 1

    print(f"Drawn {lines_drawn}/{len(lines)} lines (others outside image)")
    plt.savefig(f'lines_{title}.png', dpi=150, bbox_inches='tight')
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