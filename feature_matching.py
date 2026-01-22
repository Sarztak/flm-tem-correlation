import numpy as np
import SimpleITK as sitk
from skimage import io
from scipy import ndimage
import matplotlib.pyplot as plt

def load_as_sitk(path):
    """Load image as SimpleITK image"""
    img = io.imread(path)
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    if len(img.shape) == 3:
        img = img.mean(axis=2)
    return sitk.GetImageFromArray(img)

def log_polar_transform(img_array):
    """Convert to log-polar for rotation/scale estimation"""
    h, w = img_array.shape
    center = (h // 2, w // 2)
    max_radius = min(center)
    
    # Create log-polar coordinate grid
    theta = np.linspace(0, 2*np.pi, 360)
    log_r = np.linspace(0, np.log(max_radius), max_radius)
    
    r = np.exp(log_r)
    x = center[1] + np.outer(r, np.cos(theta))
    y = center[0] + np.outer(r, np.sin(theta))
    
    # Sample image at log-polar coordinates
    lp_img = ndimage.map_coordinates(img_array, [y, x], order=1, mode='constant')
    return lp_img

def estimate_rotation_scale(flm_sitk, tem_sitk):
    """Estimate rotation and scale using log-polar FFT"""
    flm = sitk.GetArrayFromImage(flm_sitk)
    tem = sitk.GetArrayFromImage(tem_sitk)
    
    # Convert to log-polar
    lp_flm = log_polar_transform(flm)
    lp_tem = log_polar_transform(tem)
    
    # Phase correlation in log-polar space
    f1 = np.fft.fft2(lp_flm)
    f2 = np.fft.fft2(lp_tem)
    cross_power = (f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-10)
    corr = np.fft.ifft2(cross_power).real
    
    # Find peak
    scale_idx, rotation_idx = np.unravel_index(np.argmax(corr), corr.shape)
    
    # Convert to rotation angle and scale factor
    rotation = (rotation_idx / 360.0) * 360.0  # degrees
    scale = np.exp(scale_idx * np.log(min(flm.shape) / 2) / lp_flm.shape[0])
    
    print(f"Log-polar estimate: rotation={rotation:.1f}°, scale={scale:.3f}")
    return rotation, scale

def register_affine_mi(flm_sitk, tem_sitk, initial_rotation=0, initial_scale=1.0):
    """
    Register using affine transform with Mutual Information metric.
    
    Args:
        flm_sitk: Fixed image (FLM)
        tem_sitk: Moving image (TEM)
        initial_rotation: Initial rotation estimate (degrees)
        initial_scale: Initial scale estimate
    """
    # Setup registration
    registration = sitk.ImageRegistrationMethod()
    
    # Mutual Information metric
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.2)
    
    # Optimizer
    registration.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    
    # Multi-resolution
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    
    # Initialize with rotation and scale
    initial_transform = sitk.AffineTransform(2)
    center = [s * 0.5 for s in flm_sitk.GetSize()]
    
    # Set rotation
    angle_rad = np.deg2rad(initial_rotation)
    matrix = [
        initial_scale * np.cos(angle_rad), -initial_scale * np.sin(angle_rad),
        initial_scale * np.sin(angle_rad), initial_scale * np.cos(angle_rad)
    ]
    initial_transform.SetMatrix(matrix)
    initial_transform.SetCenter(center)
    
    registration.SetInitialTransform(initial_transform, inPlace=False)
    
    # Interpolator
    registration.SetInterpolator(sitk.sitkLinear)
    
    # Execute
    print("Running affine registration with Mutual Information...")
    final_transform = registration.Execute(flm_sitk, tem_sitk)
    
    print(f"Optimizer stop condition: {registration.GetOptimizerStopConditionDescription()}")
    print(f"Final metric value: {registration.GetMetricValue():.4f}")
    
    return final_transform

def register_bspline_mi(flm_sitk, tem_sitk, affine_transform):
    """
    Refine with B-spline deformable registration.
    
    Args:
        flm_sitk: Fixed image
        tem_sitk: Moving image
        affine_transform: Initial affine transform
    """
    # Resample moving image with affine
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(flm_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(affine_transform)
    tem_affine = resampler.Execute(tem_sitk)
    
    # Setup B-spline registration
    registration = sitk.ImageRegistrationMethod()
    
    # Mutual Information
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.2)
    
    # B-spline transform
    grid_size = [8, 8]  # Control point grid
    bspline = sitk.BSplineTransformInitializer(flm_sitk, grid_size)
    registration.SetInitialTransform(bspline, inPlace=True)
    
    # Optimizer
    registration.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=100
    )
    
    # Multi-resolution
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[1, 0])
    
    registration.SetInterpolator(sitk.sitkLinear)
    
    print("Running B-spline deformable registration...")
    final_bspline = registration.Execute(flm_sitk, tem_affine)
    
    print(f"Final metric value: {registration.GetMetricValue():.4f}")
    
    # Composite transform
    composite = sitk.CompositeTransform(2)
    composite.AddTransform(affine_transform)
    composite.AddTransform(final_bspline)
    
    return composite

def visualize_registration(flm_sitk, tem_sitk, transform):
    """Visualize registration result"""
    # Resample moving image
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(flm_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    tem_registered = resampler.Execute(tem_sitk)
    
    # Convert to arrays
    flm = sitk.GetArrayFromImage(flm_sitk)
    tem_reg = sitk.GetArrayFromImage(tem_registered)
    
    # Normalize for visualization
    flm = (flm - flm.min()) / (flm.max() - flm.min())
    tem_reg = (tem_reg - tem_reg.min()) / (tem_reg.max() - tem_reg.min())
    
    # Create RGB overlay
    overlay = np.zeros((*flm.shape, 3))
    overlay[:, :, 0] = flm  # FLM in red
    overlay[:, :, 1] = tem_reg  # TEM in green
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(flm, cmap='gray')
    axes[0].set_title('FLM (Fixed)')
    axes[0].axis('off')
    
    axes[1].imshow(tem_reg, cmap='gray')
    axes[1].set_title('TEM (Registered)')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay (Red=FLM, Green=TEM)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Load images
    flm_path = "flm_image.tiff"
    tem_path = "tem_image.tiff"
    
    flm = load_as_sitk(flm_path)
    tem = load_as_sitk(tem_path)
    
    # Step 1: Coarse estimate via log-polar
    rotation, scale = estimate_rotation_scale(flm, tem)
    
    # Step 2: Affine registration with MI
    affine_transform = register_affine_mi(flm, tem, rotation, scale)
    
    # Step 3: Optional B-spline refinement for non-rigid deformation
    # final_transform = register_bspline_mi(flm, tem, affine_transform)
    final_transform = affine_transform  # Use affine only for now
    
    # Visualize
    visualize_registration(flm, tem, final_transform)
    
    # Extract affine matrix for your plugin
    if isinstance(final_transform, sitk.AffineTransform):
        matrix = np.array(final_transform.GetMatrix()).reshape(2, 2)
        translation = np.array(final_transform.GetTranslation())
        print(f"\nAffine matrix:\n{matrix}")
        print(f"Translation: {translation}")
