# flm-tem-alignment

A napari plugin for correlating fluorescence light microscopy (FLM) and transmission electron microscopy (TEM) images of the same biological specimen.

FLM gives you the big picture — which cells, which structures, where. TEM gives you the nanoscale detail. This plugin aligns the two so you can map what you see in FLM to what you see in TEM.

## How it works

The pipeline runs in three sequential steps, each exposed as a napari widget:

1. **FLM ROI Finder** — loads an FLM z-stack and a TEM image, scores each z-frame per region of interest using Laplacian sharpness to find the best focal plane, and tiles the result at TEM scale
2. **FLM & TEM Segmentation** — uses SAM2 (point-prompted) to segment the structure of interest in both the FLM tile and the TEM image
3. **Keypoint Matching** — runs SuperPoint + LightGlue on the two segmentation masks to find corresponding keypoints and estimates the affine transform that maps FLM coordinates to TEM coordinates

## Requirements

- Python 3.10
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager
- GPU recommended (CUDA) — the pipeline runs on CPU but SAM2 is slow without one

## Installation

Clone the repo and run the setup script. It installs uv (if not present), all Python dependencies, the three model repos at pinned commits, downloads model weights, and installs the napari plugin.

**Windows:**
```
git clone https://github.com/Sarztak/flm-tem-correlation.git
cd flm-tem-correlation
setup.bat
```

**Mac / Linux:**
```
git clone https://github.com/Sarztak/flm-tem-correlation.git
cd flm-tem-correlation
bash setup.sh
```

## Models installed

| Model | Purpose |
|---|---|
| [SAM2](https://github.com/facebookresearch/sam2) | Point-prompted segmentation of FLM and TEM images |
| [LightGlue + SuperPoint](https://github.com/cvg/LightGlue) | Keypoint detection and matching between segmentation masks |
| [SwinIR](https://github.com/JingyunLiang/SwinIR) | 4× super-resolution upscaling of FLM tiles before matching |

## Sample data

Demo images (specimen JEY002, grids G3_L3 and G3_L8) are hosted on OSF. Download with:

```
python download_sample_data.py           # both specimens (~2 GB total)
python download_sample_data.py g3_l3     # JEY002_G3_L3 only
python download_sample_data.py g3_l8     # JEY002_G3_L8 only
```

## Usage

Open napari, then load the three widgets in order from **Plugins → ROI Detect**:

### 1. FLM ROI Finder
- Set the path to your FLM z-stack (`.tif`) and TEM image (`.tif`)
- Set the pixel sizes in nm (FLM and TEM)
- Click **Detect Best Frame & ROIs** — the plugin scores all z-frames per ROI and displays the sharpest frame for each region
- Click points on the ROIs you want to process, then press **Enter** to confirm

### 2. FLM & TEM Segmentation
- Click **Load Segmentation Images** to display the upscaled FLM tiles
- Click on the tile you want to segment, then press **Enter**
- Place green points (foreground) and red points (background) on the FLM crop, press **Enter**
- Repeat for the TEM image

### 3. Keypoint Matching
- Adjust the match threshold if needed (default 0.02)
- Click **Match Keypoints** — the plugin runs LightGlue, estimates the affine transform, and displays the overlay of FLM warped onto TEM

## Project structure

```
flm-tem-correlation/
├── plugin/                  # napari plugin (roi-detect)
│   └── src/roi_detect/
│       ├── _widget_roi.py   # the three napari widgets
│       └── napari.yaml      # plugin manifest
├── model_setup.py           # model loading and inference helpers
├── app_helper.py            # ROI detection and tiling
├── best_frame_fix.py        # z-frame sharpness scoring
├── app2.py                  # Streamlit app (alternative UI)
├── install.py               # sets up model repos and weights
├── download_sample_data.py  # downloads demo images from OSF
├── setup.bat                # Windows one-command setup
└── setup.sh                 # Mac/Linux one-command setup
```
