import mrcfile
from PIL import Image
from skimage import exposure
import numpy as np 
import cv2
from pathlib import Path
from helper import load_image, create_edge_tem, create_edge_flm, tile_flm

"""
the .st file is 4096 x 4096 numpy array so it has only the intensity information
the .tif file is a 1366 x 1366 x 3 RGB image which means some software was
used to convert from the single channel data to 3 channel RGB image
"""


# convert to single channel since sift expects single channel
# if len(flm_cropped_arr.shape) == 3:
#     flm_cropped_arr = cv2.cvtColor(flm_cropped_arr, cv2.COLOR_RGB2GRAY)
# if len(tem_arr.shape) == 3:
#     tem_arr = cv2.cvtColor(tem_arr, cv2.COLOR_RGB2GRAY)

def match_sift_keypoints(tem_img, flm_img):

    sift = cv2.SIFT.create(
        nfeatures=0, # 0 = detect all,
        nOctaveLayers=3, # layers per octave in the pyramid
        contrastThreshold=0.04, # lower = more keypoints detected
        edgeThreshold=10, # higher = mroe keypoints detected
        sigma=1.6, # blur for the first octave
    )
    keypoints_flm, descriptors_flm = sift.detectAndCompute(flm_img, None)
    keypoints_tem, descriptors_tem = sift.detectAndCompute(tem_img, None)

    # img_keypoints = cv2.drawKeypoints(flm_img, keypoints_flm, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imwrite(out_dir / 'keypoints_flm.png', img_keypoints)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_flm, descriptors_tem, k=2)

    # Lowe's ratio test to filter out the bad matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.85 * n.distance:
            good_matches.append(m)

    img_matches = cv2.drawMatches(flm_img, keypoints_flm, tem_img, keypoints_tem, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite(out_dir / 'matches.png', img_matches)

if __name__ == "__main__":
    jey_002_g3_l3_path = Path('./jey_002_g3_l3')
    tem_img_path = jey_002_g3_l3_path / "JEY002_G3_L3_1950x_t-13.tif"
    flm_cropped_refl_path = jey_002_g3_l3_path / "FLM-JEY002_G3_L3_z11_refl_cropped.tif"
    flm_refl_path = jey_002_g3_l3_path / "FLM-JEY002_G3_L3_z11_refl.tif"
    flm_stack_path = jey_002_g3_l3_path / "FLM-stack_JEY002_G3_L3.tif"

    out_dir = Path('./output')

    # tem_contour = create_edge_tem(tem_img_path, 300, 400)
    # cv2.imwrite(out_dir / 'tem_contour.png', tem_contour)

    # flm_contour = create_edge_flm(flm_cropped_refl_path_2, 30, 50)
    # cv2.imwrite(out_dir / 'flm_contour.png', flm_contour)
    tem_img = load_image(tem_img_path)

    # folder where search_regions are stored
    for img_path in (out_dir / 'search_regions').glob("*.png"):
        flm_img = load_image(img_path)

        tiles = tile_flm(flm_img, tem_img, tile_scale=2)

        tiles_dir = out_dir / 'tiles'
        tiles_dir.mkdir(exist_ok=True)
        for i, tile in enumerate(tiles):
            img = Image.fromarray(tile['crop'])
            img.save(tiles_dir / f'{img_path.stem}_{str(i).zfill(4)}.png')
    # crop_flm_center(flm_img, tem_img)

    
    # breakpoint()
    # match_sift_keypoints(tem_contour, flm_img)