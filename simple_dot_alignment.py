import numpy as np
import cv2
from skimage import io
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

def load_image(path):
    img = io.imread(path)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img

def detect_circles(img, min_radius=10, max_radius=100):
    """Detect circles using Hough transform"""
    # Normalize to 8-bit
    if img.dtype != np.uint8:
        img_8bit = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    else:
        img_8bit = img
    
    # Blur to reduce noise
    blurred = cv2.GaussianBlur(img_8bit, (9, 9), 2)
    
    # Detect circles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_radius * 2,
        param1=50,
        param2=30,
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    if circles is not None:
        circles = circles[0]
        centers = circles[:, :2]  # x, y coordinates
        print(f"Detected {len(centers)} circles")
        return centers
    else:
        print("No circles detected")
        return np.array([])

def match_points(pts1, pts2, max_distance=50):
    """Match corresponding points between two sets"""
    if len(pts1) == 0 or len(pts2) == 0:
        return np.array([]), np.array([])
    
    # Compute pairwise distances
    distances = cdist(pts1, pts2)
    
    # Find optimal assignment (Hungarian algorithm)
    row_ind, col_ind = linear_sum_assignment(distances)
    
    # Filter by distance threshold
    valid = distances[row_ind, col_ind] < max_distance
    matched1 = pts1[row_ind[valid]]
    matched2 = pts2[col_ind[valid]]
    
    print(f"Matched {len(matched1)} point pairs")
    return matched1, matched2

def estimate_affine_from_points(pts_src, pts_dst):
    """Estimate affine transform from point correspondences"""
    # Need at least 3 points
    if len(pts_src) < 3:
        print("Not enough points for affine estimation")
        return None
    
    # Use OpenCV's estimateAffinePartial2D with RANSAC
    matrix, inliers = cv2.estimateAffinePartial2D(
        pts_src.astype(np.float32),
        pts_dst.astype(np.float32),
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0
    )
    
    if matrix is not None:
        n_inliers = np.sum(inliers)
        print(f"Affine estimation: {n_inliers}/{len(pts_src)} inliers")
        print(f"Transform matrix:\n{matrix}")
        return matrix
    else:
        print("Affine estimation failed")
        return None

def apply_transform(img, matrix):
    """Apply affine transformation"""
    h, w = img.shape[:2]
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR)

def visualize_alignment(flm, tem, matrix, pts_flm=None, pts_tem=None):
    """Visualize the alignment result"""
    # Transform TEM to FLM space
    tem_aligned = apply_transform(tem, matrix)
    
    # Normalize
    flm_norm = ((flm - flm.min()) / (flm.max() - flm.min()) * 255).astype(np.uint8)
    tem_norm = ((tem_aligned - tem_aligned.min()) / (tem_aligned.max() - tem_aligned.min()) * 255).astype(np.uint8)
    
    # Create overlay
    overlay = np.zeros((*flm.shape, 3), dtype=np.uint8)
    overlay[:, :, 0] = flm_norm
    overlay[:, :, 1] = tem_norm
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    # Original images with detected points
    axes[0, 0].imshow(flm, cmap='gray')
    if pts_flm is not None and len(pts_flm) > 0:
        axes[0, 0].scatter(pts_flm[:, 0], pts_flm[:, 1], c='red', s=30, alpha=0.5)
    axes[0, 0].set_title(f'FLM with {len(pts_flm) if pts_flm is not None else 0} detected circles')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(tem, cmap='gray')
    if pts_tem is not None and len(pts_tem) > 0:
        axes[0, 1].scatter(pts_tem[:, 0], pts_tem[:, 1], c='blue', s=30, alpha=0.5)
    axes[0, 1].set_title(f'TEM with {len(pts_tem) if pts_tem is not None else 0} detected circles')
    axes[0, 1].axis('off')
    
    # Aligned result
    axes[1, 0].imshow(tem_aligned, cmap='gray')
    axes[1, 0].set_title('TEM Aligned to FLM')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title('Overlay (Red=FLM, Green=TEM)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('circle_alignment.png', dpi=150, bbox_inches='tight')
    print("Saved to circle_alignment.png")
    plt.close()

# Main workflow
if __name__ == "__main__":
    # Load images
    flm_path = "flm_image.tiff"
    tem_path = "tem_image.tiff"
    
    flm = load_image(flm_path)
    tem = load_image(tem_path)
    
    print("=== Detecting circles in FLM ===")
    circles_flm = detect_circles(flm, min_radius=5, max_radius=30)
    
    print("\n=== Detecting circles in TEM ===")
    circles_tem = detect_circles(tem, min_radius=5, max_radius=30)
    
    if len(circles_flm) > 0 and len(circles_tem) > 0:
        print("\n=== Matching circles ===")
        # Try to match based on relative positions
        matched_flm, matched_tem = match_points(circles_flm, circles_tem, max_distance=100)
        
        if len(matched_flm) >= 3:
            print("\n=== Estimating transform ===")
            matrix = estimate_affine_from_points(matched_tem, matched_flm)
            
            if matrix is not None:
                print("\n=== Visualizing ===")
                visualize_alignment(flm, tem, matrix, circles_flm, circles_tem)
                
                # Save transform for later use
                np.savetxt('affine_matrix.txt', matrix)
                print("Saved transform to affine_matrix.txt")
            else:
                print("Failed to estimate transform")
        else:
            print(f"Not enough matched points: {len(matched_flm)}")
    else:
        print("Circle detection failed")
