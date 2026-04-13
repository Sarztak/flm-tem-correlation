# /content/drive/MyDrive/setup.py
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np 
import tempfile
import cv2
import subprocess
import math

DRIVE_BASE = Path('/content/drive/MyDrive')

SAM_REPO = DRIVE_BASE / 'segment_anything'
SAM_CHECKPOINT = SAM_REPO / 'sam_vit_h_4b8939.pt'
LIGHTGLUE_REPO = DRIVE_BASE / 'LightGlue'

SWINIR_REPO = DRIVE_BASE / 'SwinIR'
SWINIR_MODEL_ZOO = SWINIR_REPO / 'model_zoo' / 'swinir'
SWINIR_MODEL = SWINIR_MODEL_ZOO / '003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth'
SWINIR_MODEL_URL = ('https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/'
                    '003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth')

INPUT_DIR = DRIVE_BASE / 'flm_tem_correlation'


def download_repos_and_setup():
    if not SAM_REPO.exists():
        print("Cloning SAM repo to Drive (one-time setup)...")
        os.system(f'git clone https://github.com/facebookresearch/segment-anything.git {SAM_REPO}')
    else:
        print("SAM repo found on Drive, skipping clone.")
    sys.path.append(str(SAM_REPO))
    os.system(f'pip install -e {str(SAM_REPO)} -q')

    if not SAM_CHECKPOINT.exists():
        print("Downloading SAM checkpoint to Drive (one-time setup)...")
        os.system(f'wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O {SAM_CHECKPOINT}')
    else:
        print("SAM checkpoint found on Drive, skipping download.")

    if not LIGHTGLUE_REPO.exists():
        print("Cloning LightGlue to Drive (one-time setup)...")
        os.system(f'git clone --quiet https://github.com/cvg/LightGlue/ {LIGHTGLUE_REPO}')
    else:
        print("LightGlue found on Drive, skipping clone.")
    sys.path.append(str(LIGHTGLUE_REPO))
    os.system(f'pip install --progress-bar off --quiet -e {LIGHTGLUE_REPO}')

    if not SWINIR_REPO.exists():
        print("Cloning SwinIR to Drive (one-time setup)...")
        os.system(f'git clone https://github.com/JingyunLiang/SwinIR.git {SWINIR_REPO}')
    else:
        print("SwinIR repo found on Drive, skipping clone.")
    sys.path.insert(0, str(SWINIR_REPO))
    os.system('pip install timm -q')

    SWINIR_MODEL_ZOO.mkdir(parents=True, exist_ok=True)
    if not SWINIR_MODEL.exists():
        print("Downloading SwinIR pretrained model to Drive (one-time setup)...")
        os.system(f'wget -q {SWINIR_MODEL_URL} -O {SWINIR_MODEL}')
    else:
        print("SwinIR model found on Drive, skipping download.")


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def show_image(path, figsize=(5, 5)) -> None:
    """shows image and returns back the array"""
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    return image

def show_mask_over_image(masks, image, figsize=(5, 5)):
    plt.figure(figsize=figsize)
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.show()

def plot_masks(masks, n=10, cell_w=3, cell_h=3):
    # plot top n masks
    n = min(n, len(masks))

    cols = 5 # 5 masks in one row
    rows = math.ceil(n / 5)
    fig, ax = plt.subplots(rows, cols, figsize=(cols * cell_w, rows * cell_h))
    ax = ax.flatten()
    for i in range(n):
        ax[i].axis('off')
        if i >= len(masks):
            continue
        ax[i].imshow(masks[i]['segmentation'], cmap='coolwarm')
    plt.tight_layout()
    plt.show()

def plot_one_mask(masks, n):
    """utility to plot the nth mask indexing starts from 1"""
    if 0 < n < len(masks):
        plt.imshow(masks[n - 1]['segmentation'], cmap='coolwarm')

def swinir_upscale(img: np.ndarray) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        input_dir  = tmpdir / 'inputs'
        output_dir = tmpdir / 'results'
        input_dir.mkdir()
        
        cv2.imwrite(str(input_dir / 'img.png'), img)
        
        subprocess.run([
            'python', str(SWINIR_REPO / 'main_test_swinir.py'),
            '--task', 'real_sr',
            '--scale', '4',
            '--large_model',
            '--model_path', str(SWINIR_MODEL),
            '--folder_lq', str(input_dir),
            '--save_dir', str(output_dir),
        ], check=True, cwd=str(SWINIR_REPO), capture_output=False)
        
        result_path = next(output_dir.glob('*.png'))
        return cv2.imread(str(result_path))

