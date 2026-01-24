from helper import *
from detect_lines import *
from skimage.io import imsave
from scipy.ndimage import rotate
from PIL import Image
def main():
    # visualize_detected_dots(,)
    # flm_img = load_image("./images/grid_1/flm.tif", )
    tem_img = load_image("./images/grid_1/tem.tif", )
    # flm_img = center_crop(flm_img)
    lines, edges = detect_grid_lines(
        tem_img, 
        sigma=4,
        threshold=0.1,
        min_angle=50, 
        min_distance=25,
        num_peaks=200,
    )
    grp_lines = group_lines(lines)
    grp_info = [(len(v), np.mean(v)) for v in grp_lines.values()]
    grp_info = sorted(grp_info, key=lambda x: x[0], reverse=True) # this give me largest group of angles
    # select the largest 2
    a1 = grp_info[0][1]
    a2 = grp_info[1][1]
    # check for the difference in angle and it should be about 90 degrees or 270 with some tolerance say 3 degrees
    diff = np.abs(a1 - a2)
    tol = 3 
    diff_90 = 'yes' if np.abs(diff - 90) > tol else 'no'
    # rotate the image
    angle = (90 - a2) # assuming a2 is in first quadrant
    rot_img = rotate(tem_img, -angle, reshape=False) # need anticlockwise rot
    rot_img_pil = Image.fromarray(rot_img)
    rot_img_pil.save('rot_tem.png')
    breakpoint()
    out = overlay_hough_lines(flm_img, lines)
    imsave("flm_512.png", out.astype(np.uint8))
    # visualize_detected_lines(flm_img, lines, "TEM")
    # visualize_edges_only("./images/grid_1/flm.tif", threshold_range=[0.1, 0.2, 0.3], sigma_range=[6, 7, 8])
if __name__ == "__main__":
    main()
