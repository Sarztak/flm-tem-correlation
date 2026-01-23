from helper import *
from detect_lines import *
from skimage.io import imsave
def main():
    # visualize_detected_dots(,)
    flm_img = load_image("./images/grid_1/flm.tif", )
    # tem_img = load_image("./images/grid_1/tem.tif", )
    flm_img = center_crop(flm_img)
    lines, edges = detect_grid_lines(
        flm_img, 
        sigma=4,
        threshold=0.1,
        min_angle=50, 
        min_distance=25,
        num_peaks=10,
    )
    out = overlay_hough_lines(flm_img, lines)
    imsave("flm_512.png", out.astype(np.uint8))
    # visualize_detected_lines(flm_img, lines, "TEM")
    # visualize_edges_only("./images/grid_1/flm.tif", threshold_range=[0.1, 0.2, 0.3], sigma_range=[6, 7, 8])
if __name__ == "__main__":
    main()
