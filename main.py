from helper import *
from detect_lines import *
def main():
    # visualize_detected_dots(,)
    flm_img = load_image("./images/grid_1/flm.tif", )
    # tem_img = load_image("./images/grid_1/tem.tif", )
    lines, edges = detect_grid_lines(
        flm_img, 
        sigma=4,
        threshold=0.1,
        min_angle=50, 
        min_distance=25,
        num_peaks=10,
    )
    visualize_detected_lines(flm_img, lines, "FLM")
    # visualize_edges_only("./images/grid_1/flm.tif", threshold_range=[0.1, 0.2, 0.3], sigma_range=[6, 7, 8])
if __name__ == "__main__":
    main()
