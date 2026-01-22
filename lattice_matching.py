import numpy as np
from skimage import io
from scipy import ndimage, fft
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def load_image(path):
    """Load and normalize image"""
    img = io.imread(path)
    if len(img.shape) == 3:
        img = img.mean(axis=2)
    img = img.astype(np.float32)
    return (img - img.mean()) / (img.std() + 1e-8)

def find_lattice_vectors(img):
    """
    Find lattice vectors from FFT peaks.
    Returns two dominant periodic directions and spacings.
    """
    # FFT to find periodicities
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    magnitude = np.abs(f_shift)
    
    # Remove DC component
    h, w = magnitude.shape
    center = (h // 2, w // 2)
    y, x = np.ogrid[:h, :w]
    mask = ((x - center[1])**2 + (y - center[0])**2) > 25
    magnitude = magnitude * mask
    
    # Find peaks (excluding center)
    from scipy.ndimage import maximum_filter
    local_max = (magnitude == maximum_filter(magnitude, size=20))
    peaks = np.where(local_max & (magnitude > np.percentile(magnitude, 99.5)))
    
    # Get peak positions relative to center
    peak_coords = []
    for py, px in zip(peaks[0], peaks[1]):
        dy, dx = py - center[0], px - center[1]
        if np.sqrt(dx**2 + dy**2) > 10:  # Skip near-center
            peak_coords.append((dy, dx, magnitude[py, px]))
    
    # Sort by magnitude and take top peaks
    peak_coords.sort(key=lambda x: x[2], reverse=True)
    
    if len(peak_coords) < 2:
        print("Could not find lattice peaks")
        return None
    
    # First two strongest peaks define lattice
    v1 = np.array([peak_coords[0][0], peak_coords[0][1]])
    v2 = np.array([peak_coords[1][0], peak_coords[1][1]])
    
    # Convert frequency to spatial spacing
    spacing1 = h / np.linalg.norm(v1)
    spacing2 = h / np.linalg.norm(v2)
    angle1 = np.arctan2(v1[0], v1[1])
    angle2 = np.arctan2(v2[0], v2[1])
    
    print(f"Lattice spacing: {spacing1:.1f}, {spacing2:.1f} pixels")
    print(f"Lattice angles: {np.rad2deg(angle1):.1f}°, {np.rad2deg(angle2):.1f}°")
    
    return {
        'vectors': (v1, v2),
        'spacings': (spacing1, spacing2),
        'angles': (angle1, angle2),
        'fft_magnitude': magnitude
    }

def estimate_transform_from_lattice(lattice_flm, lattice_tem):
    """
    Estimate rotation and scale from lattice parameters.
    """
    if lattice_flm is None or lattice_tem is None:
        return 0, 1.0
    
    # Compare primary lattice vectors
    angle_flm = lattice_flm['angles'][0]
    angle_tem = lattice_tem['angles'][0]
    rotation = np.rad2deg(angle_tem - angle_flm)
    
    spacing_flm = lattice_flm['spacings'][0]
    spacing_tem = lattice_tem['spacings'][0]
    scale = spacing_tem / spacing_flm
    
    print(f"\nEstimated rotation: {rotation:.1f}°")
    print(f"Estimated scale: {scale:.3f}")
    
    return rotation, scale

def detect_dots(img, threshold_percentile=90):
    """Detect dot centers using local maxima"""
    from scipy.ndimage import gaussian_filter, maximum_filter
    
    # Smooth slightly
    smoothed = gaussian_filter(img, sigma=2)
    
    # Find local maxima
    local_max = maximum_filter(smoothed, size=10)
    dots = (smoothed == local_max) & (smoothed > np.percentile(smoothed, threshold_percentile))
    
    coords = np.argwhere(dots)
    print(f"Detected {len(coords)} dots")
    
    return coords

def phase_correlation(img1, img2):
    """Find translation offset using phase correlation"""
    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)
    
    cross_power = (f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-10)
    corr = np.fft.ifft2(cross_power).real
    
    # Find peak
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    
    # Convert to offset
    dy = peak[0] if peak[0] < corr.shape[0] // 2 else peak[0] - corr.shape[0]
    dx = peak[1] if peak[1] < corr.shape[1] // 2 else peak[1] - corr.shape[1]
    
    return dy, dx

def apply_affine(img, rotation_deg, scale, tx, ty):
    """Apply affine transformation to image"""
    h, w = img.shape
    center = np.array([h // 2, w // 2])
    
    # Build affine matrix
    angle_rad = np.deg2rad(rotation_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    # Rotation + scale matrix
    matrix = scale * np.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    # Apply around center with translation
    offset = center - matrix @ center + np.array([ty, tx])
    
    return ndimage.affine_transform(img, matrix, offset=offset, order=1)

def mutual_information(img1, img2):
    """Calculate normalized mutual information"""
    # Flatten and remove invalid values
    i1 = img1.flatten()
    i2 = img2.flatten()
    valid = np.isfinite(i1) & np.isfinite(i2)
    i1, i2 = i1[valid], i2[valid]
    
    # Compute joint histogram
    hist_2d, _, _ = np.histogram2d(i1, i2, bins=50)
    
    # Compute probabilities
    pxy = hist_2d / hist_2d.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    
    # MI calculation
    px_py = px[:, None] * py[None, :]
    nzs = pxy > 0
    mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / (px_py[nzs] + 1e-10)))
    
    # Normalize
    hx = -np.sum(px * np.log(px + 1e-10))
    hy = -np.sum(py * np.log(py + 1e-10))
    nmi = 2 * mi / (hx + hy + 1e-10)
    
    return nmi

def optimize_alignment(flm, tem, init_rotation, init_scale):
    """Optimize alignment using mutual information"""
    print("\nOptimizing alignment...")
    
    def objective(params):
        rotation, scale, tx, ty = params
        tem_transformed = apply_affine(tem, rotation, scale, tx, ty)
        
        # Crop to common region
        h, w = min(flm.shape[0], tem_transformed.shape[0]), min(flm.shape[1], tem_transformed.shape[1])
        nmi = mutual_information(flm[:h, :w], tem_transformed[:h, :w])
        
        return -nmi  # Maximize MI = minimize -MI
    
    # Initial parameters
    x0 = [init_rotation, init_scale, 0, 0]
    
    # Optimize
    result = minimize(
        objective,
        x0,
        method='Nelder-Mead',
        options={'maxiter': 100, 'disp': True}
    )
    
    rotation, scale, tx, ty = result.x
    print(f"\nOptimized: rotation={rotation:.2f}°, scale={scale:.3f}, translation=({tx:.1f}, {ty:.1f})")
    
    return rotation, scale, tx, ty

def visualize_result(flm, tem, rotation, scale, tx, ty):
    """Visualize alignment result"""
    tem_aligned = apply_affine(tem, rotation, scale, tx, ty)
    
    # Normalize for display
    flm_norm = (flm - flm.min()) / (flm.max() - flm.min() + 1e-10)
    tem_norm = (tem_aligned - tem_aligned.min()) / (tem_aligned.max() - tem_aligned.min() + 1e-10)
    
    # Create overlay
    h, w = min(flm.shape[0], tem_aligned.shape[0]), min(flm.shape[1], tem_aligned.shape[1])
    overlay = np.zeros((h, w, 3))
    overlay[:, :, 0] = flm_norm[:h, :w]
    overlay[:, :, 1] = tem_norm[:h, :w]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(flm, cmap='gray')
    axes[0].set_title('FLM (Fixed)')
    axes[0].axis('off')
    
    axes[1].imshow(tem_aligned, cmap='gray')
    axes[1].set_title('TEM (Aligned)')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (Red=FLM, Green=TEM)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('alignment_result.png', dpi=150, bbox_inches='tight')
    print("Saved to alignment_result.png")
    plt.close()

# Main workflow
if __name__ == "__main__":
    # Load images
    flm = load_image("./images/grid_1/flm.tif")
    tem = load_image("./images/grid_1/tem.tif")
    
    print("=== Analyzing FLM ===")
    lattice_flm = find_lattice_vectors(flm)
    
    print("\n=== Analyzing TEM ===")
    lattice_tem = find_lattice_vectors(tem)
    
    # Estimate initial transform from lattice
    rotation, scale = estimate_transform_from_lattice(lattice_flm, lattice_tem)
    
    # Optimize using mutual information
    rotation, scale, tx, ty = optimize_alignment(flm, tem, rotation, scale)
    
    # Visualize
    visualize_result(flm, tem, rotation, scale, tx, ty)
    
    # Output transform matrix
    angle_rad = np.deg2rad(rotation)
    matrix = scale * np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), tx],
        [np.sin(angle_rad), np.cos(angle_rad), ty],
        [0, 0, 1]
    ])
    print(f"\nFinal affine matrix:\n{matrix}")
