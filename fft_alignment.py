import numpy as np
from skimage import io
from scipy import fft, ndimage
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt

def find_lattice_vectors(img, title=""):
    """
    Find lattice vectors from FFT peaks.
    Visualizes the FFT magnitude and detected peaks.
    Returns lattice parameters or None.
    """
    # Compute FFT
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    magnitude = np.abs(f_shift)

    # Remove DC peak
    h, w = magnitude.shape
    center = (h // 2, w // 2)
    y, x = np.ogrid[:h, :w]
    mask = ((x - center[1])**2 + (y - center[0])**2) > 25
    magnitude_masked = magnitude * mask

    # Detect peaks
    local_max = (magnitude_masked == maximum_filter(magnitude_masked, size=20))
    threshold = np.percentile(magnitude_masked, 99.5)
    peaks = np.where(local_max & (magnitude_masked > threshold))

    # Visualize FFT and peaks
    plt.figure(figsize=(8, 8))
    plt.imshow(np.log(1 + magnitude), cmap='viridis')
    plt.colorbar(label='Log Magnitude')
    plt.plot(center[1], center[0], 'r+', markersize=12, mew=2, label='DC')
    for py, px in zip(peaks[0], peaks[1]):
        plt.plot(px, py, 'c.', markersize=6)
        # if (py - center[0])**2 + (px - center[1])**2 > 100:  # Label distant
        #     plt.text(px + 3, py + 3, f'({py - center[0]}, {px - center[1]})',
        #              color='white', fontsize=6)
    plt.title(f'FFT Peaks — {title}')
    plt.savefig(f'fft_peaks_{title}.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Extract peak coordinates relative to center
    peak_coords = []
    for py, px in zip(peaks[0], peaks[1]):
        dy, dx = py - center[0], px - center[1]
        r = np.hypot(dx, dy)
        if r > 10:
            peak_coords.append((dy, dx, magnitude[py, px]))

    # Sort by strength
    peak_coords.sort(key=lambda x: x[2], reverse=True)

    if len(peak_coords) < 2:
        print(f"[{title}] Not enough FFT peaks found.")
        return None

    v1 = np.array([peak_coords[0][0], peak_coords[0][1]])
    v2 = np.array([peak_coords[1][0], peak_coords[1][1]])

    spacing1 = h / np.linalg.norm(v1)
    angle1 = np.arctan2(v1[0], v1[1])

    print(f"[{title}] Lattice spacing: {spacing1:.1f} px")
    print(f"[{title}] Lattice angle: {np.rad2deg(angle1):+.1f}°")

    return {
        'vector': v1,
        'spacing': spacing1,
        'angle': angle1,
        'peaks': peaks,
        'magnitude': magnitude
    }


def estimate_transform_from_fft(lattice_flm, lattice_tem):
    """Estimate rotation and scale from FFT lattice results."""
    if lattice_flm is None or lattice_tem is None:
        print("Cannot estimate transform: missing FFT results.")
        return None

    # Avoid 90° symmetry ambiguity
    angle_diff = np.rad2deg(lattice_tem['angle'] - lattice_flm['angle'])
    angle_wrapped = (angle_diff + 180) % 360 - 180  # Range: [-180, 180]
    
    # Correct if near ±90° (square grid symmetry)
    if abs(angle_wrapped - 90) < 10:
        angle_wrapped -= 90
    elif abs(angle_wrapped + 90) < 10:
        angle_wrapped += 90

    rotation = angle_wrapped
    scale = lattice_tem['spacing'] / lattice_flm['spacing']

    print(f"\n→ Estimated rotation: {rotation:.1f}°")
    print(f"→ Estimated scale: {scale:.3f}")

    return rotation, scale


# --- Main ---
if __name__ == "__main__":
    # Load and normalize
    flm = io.imread("./images/grid_1/flm.tif").astype(np.float32)
    if len(flm.shape) == 3: flm = flm.mean(axis=2)
    flm = (flm - flm.mean()) / (flm.std() + 1e-8)

    tem = io.imread("./images/grid_1/tem.tif").astype(np.float32)
    if len(tem.shape) == 3: tem = tem.mean(axis=2)
    tem = (tem - tem.mean()) / (tem.std() + 1e-8)

    from helper import detect_dots
    flm_dots = detect_dots(flm)  # List of (y,x) coordinates
    tem_dots = detect_dots(tem, threshold_percentile=30)  # Lower threshold for TEM holes
    print(f"FLM dots: {len(flm_dots)}, TEM dots: {len(tem_dots)}") 
    # # Analyze each
    # print("=== FLM FFT Analysis ===")
    # lattice_flm = find_lattice_vectors(flm, "FLM")

    # print("\n=== TEM FFT Analysis ===")
    # lattice_tem = find_lattice_vectors(tem, "TEM")

    # # Estimate transform
    # print("\n=== Transform Estimation ===")
    # estimate_transform_from_fft(lattice_flm, lattice_tem)