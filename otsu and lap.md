# Selecting the Sharpest Z Plane and Localizing Regions of Interest

## Laplacian Sharpness Detection

### The intuition

A fluorescence microscopy z-stack is a series of images taken at different focal depths. At most depths the structure of interest is out of focus and appears blurry. Only at one particular depth does it appear sharp. The challenge is finding that depth automatically, without looking at every slice manually.

Sharpness in an image is determined by how much fine detail and edge information it contains. A blurry image has smooth, gradual transitions between pixel values. A sharp image has abrupt transitions — edges — where pixel values change rapidly over a short distance.

The Laplacian operator is a mathematical tool that measures exactly this: how rapidly pixel values are changing in the local neighbourhood of each pixel.

### The mathematics

The Laplacian of an image $I$ at a pixel $(x, y)$ is the sum of the second-order partial derivatives:

$$\nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2}$$

In discrete terms, for a pixel and its immediate neighbours, this is approximated by a convolution with the kernel:

$$\begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix}$$

What this kernel does is subtract the value of the central pixel from the average of its neighbours. Where the image is smooth and pixel values are similar across the neighbourhood, the result is close to zero. Where there is a sharp edge and values change rapidly, the result is large in magnitude.

### Why variance of the Laplacian correlates with sharpness

After applying the Laplacian to an image you get a response map where edges and fine details produce large values and flat regions produce values near zero. Taking the **variance** of this response map gives a single scalar that summarises how much edge information is present across the whole image.

- High variance: the image contains strong, well-defined edges — it is sharp
- Low variance: most Laplacian responses are near zero — the image is blurry

### How this is used here

For each region of interest identified by Otsu thresholding, the Laplacian variance is computed independently for every slice in the z-stack. The slice with the highest Laplacian variance is selected as the sharpest representation of that region.

This is done per region of interest rather than for the whole image because the sharpest focal plane for one region of interest may not be the sharpest for another — different structures can sit at different depths within the same sample.

```python
import cv2
import numpy as np

def laplacian_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

# For a stack of images (list of numpy arrays):
sharpness_scores = [laplacian_sharpness(slice) for slice in z_stack]
best_slice_index = np.argmax(sharpness_scores)
```

---

## Otsu Thresholding

### The intuition

Before computing sharpness, we need to know where in the FM image to look. The entire image contains background noise and regions with no biological activity. We are only interested in regions where fluorescent proteins have accumulated, visible as bright spots in the green (GFP) and blue (BFP) channels.

The problem is: what counts as bright? Setting a threshold manually is arbitrary and does not generalise across different images or experiments. Otsu thresholding solves this by finding the threshold automatically from the pixel intensity distribution of the image itself.

### The mathematics

Every image has a histogram of pixel intensities. Otsu's method treats this as a two-class problem: pixels belong either to the background (low intensity) or the foreground (high intensity, i.e. fluorescent signal). The goal is to find the threshold $t$ that best separates these two classes.

The criterion is to minimise the **weighted intra-class variance**, which is equivalent to maximising the **inter-class variance**:

$$\sigma^2_b(t) = w_0(t) \cdot w_1(t) \cdot [\mu_0(t) - \mu_1(t)]^2$$

Where:
- $w_0(t)$ and $w_1(t)$ are the proportions of pixels below and above threshold $t$
- $\mu_0(t)$ and $\mu_1(t)$ are the mean intensities of the two classes

Otsu's method searches across all possible threshold values and selects the $t$ that maximises $\sigma^2_b(t)$. This is the point where the two classes are most separated from each other.

### Why this works for localizing fluorescent regions

Fluorescence images have a natural bimodal intensity distribution: a large population of dim background pixels and a smaller population of bright signal pixels. Otsu thresholding reliably finds the boundary between these two populations without any manual tuning.

The result is a binary mask where pixels belonging to fluorescent regions are set to 1 and background pixels are set to 0. This mask defines the regions of interest that are then passed to the Laplacian sharpness selection step.

### An important assumption

This approach assumes that the scientist is interested in regions that show fluorescent activity, which is a valid assumption: the purpose of labelling with GFP or BFP is precisely to mark the structures or events under investigation. Regions with no fluorescent signal are not regions of interest and can be excluded from the search space entirely.

```python
import cv2
import numpy as np

def otsu_mask(image_channel):
    # Normalise to 8-bit if needed
    channel = cv2.normalize(image_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    threshold_value, binary_mask = cv2.threshold(
        channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return binary_mask, threshold_value

# Apply to green and blue channels separately, then combine
green_mask, _ = otsu_mask(fm_image[:, :, 1])  # green channel
blue_mask, _  = otsu_mask(fm_image[:, :, 0])  # blue channel
combined_roi_mask = cv2.bitwise_or(green_mask, blue_mask)
```