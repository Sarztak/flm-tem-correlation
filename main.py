from helper import *
from detect_lines import *
from skimage.io import imsave
from scipy.ndimage import rotate
import cv2
from PIL import Image
def get_angles(img):
    lines, edges = detect_grid_lines(
        img, 
        sigma=4,
        threshold=0.1,
        min_angle=50, 
        min_distance=25,
        num_peaks=200,
    )
    grp_lines = group_lines(lines)
    grp_info = [(len(v), np.mean(v)) for v in grp_lines.values()]
    grp_info = sorted(grp_info, key=lambda x: x[0], reverse=True) # this give me largest group of angles
    return grp_info

def main():
    # visualize_detected_dots(,)
    flm_img = load_image("./images/grid_1/flm.tif", )
    tem_img = load_image("./images/grid_1/tem.tif", )
    # flm_img = center_crop(flm_img)
    # select the largest 2
    flm_grp_info = get_angles(flm_img)
    tem_grp_info = get_angles(tem_img)

    flm_a1 = flm_grp_info[0][1]
    flm_a2 = flm_grp_info[1][1]

    tem_a1 = tem_grp_info[0][1]
    tem_a2 = tem_grp_info[1][1]

    breakpoint()
    # compute the angle difference between of TEM with respect of FLM
    # the TEM image will be rotated. 
    # the question also remains in which direction
    diff_rot_direct = tem_a1 - flm_a1
    rot_tem = rotate(tem_img, diff_rot_direct, reshape=False)
    cv2.imwrite('./output/rot_tem_wrt_flm.png', rot_tem)
    # check for the difference in angle and it should be about 90 degrees or 270 with some tolerance say 3 degrees
    # diff = np.abs(a1 - a2)
    # tol = 3 

    # diff_90 = 'yes' if np.abs(diff - 90) > tol else 'no'
    # # rotate the image
    # angle = (90 - a2) # assuming a2 is in first quadrant
    # rot_img = rotate(flm_img, -angle, reshape=False) # need anticlockwise rot

    # # make the image binary
    # rot_img = rot_img.mean(axis=2)
    # rot_img[rot_img.astype(int) > 30] = 255 # where it is not black make it white
    # rot_img = rot_img.astype(np.uint8)
    # rot_img_pil = Image.fromarray(rot_img)
    # rot_img_pil.save('rot_flm.png')
    # out = overlay_hough_lines(flm_img, lines)
    # imsave("flm_512.png", out.astype(np.uint8))
    # visualize_detected_lines(flm_img, lines, "TEM")
    # visualize_edges_only("./images/grid_1/flm.tif", threshold_range=[0.1, 0.2, 0.3], sigma_range=[6, 7, 8])
if __name__ == "__main__":
    main()
