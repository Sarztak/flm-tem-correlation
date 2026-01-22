import os
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from scipy.ndimage import gaussian_filter

def detect_grid_lines(img, sigma=3, threshold=0.2, min_distance=50, min_angle=10, num_peaks=50):
    """Detect straight lines using Canny edge + Hough transform."""
    # Smooth to reduce noise
    if len(img.shape) == 3: img = img.mean(axis=2).astype(np.float32)
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-12)
    smoothed = gaussian_filter(img_norm, sigma=sigma)
    
    # Enhanced contrast for grid
    # smoothed = (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-12)
    
    # Detect edges
    edges = canny(smoothed, low_threshold=threshold*0.5, high_threshold=threshold)
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
    """Visualize Hough lines overlaid on image."""

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

    h, w = img.shape
    for angle, dist in lines:
        # Compute endpoints of line segment within image bounds
        if np.isclose(np.cos(angle), 0):  # Near vertical
            y0, y1 = 0, h - 1
            x0 = x1 = dist / (np.sin(angle) + 1e-12)
        elif np.isclose(np.sin(angle), 0):  # Near horizontal
            x0, x1 = 0, w - 1
            y0 = y1 = dist / (np.cos(angle) + 1e-12)
        else:
            x0 = 0
            y0 = (dist - x0 * np.cos(angle)) / (np.sin(angle) + 1e-12)
            x1 = w - 1
            y1 = (dist - x1 * np.cos(angle)) / (np.sin(angle) + 1e-12)
            y0, y1 = np.clip([y0, y1], 0, h - 1)

        ax.plot([x0, x1], [y0, y1], 'r-', linewidth=1)

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