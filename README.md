# SAM-CEBNet: Foundation-Guided Corner & Edge Building Network Suite

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official PyTorch implementation for **SAM-CEBNet** (*Foundation-Guided Corner and Edge Refinement Network Family (SAM-2 & SAM-3) for High-Precision Building Footprint Extraction in High-Resolution Remote Sensing Imagery*) along with a unified, benchmark-ready implementation suite of state-of-the-art remote sensing methods (2022–2026).

---

## 🌟 Key Features

* **Proposed Architecture Family**:
  * **SAM-CEBNet**: SAM-2 Hiera foundation perception encoder + High-Frequency Edge Cross-Attention + Corner-Guided BiFPN + DSBR Decoder.
  * **SAM3-CEBNet**: SAM-3 perception backbone with promptable concept segmentation (PCS) tokens and decoupled global presence gating.
* **Chronological SOTA Benchmark Suite (2022–2026)**:
  * **CBRNet** (*Chen et al., IEEE TGRS 2026*) — Corner-Guided Boundary Refinement.
  * **CGCCE-Net** (*Wang et al., Information Sciences 2026*) — Change-Guided Cross-Correlation.
  * **EdgeRefNet** (*Hima et al., IEEE TGRS 2026*) — Edge-Guided Refinement with EDB/EEM blocks.
  * **GMFIMamba** (*Xu et al., EAAI 2026*) — Group Mamba Feature Interaction (GMBFIM) + MCFEM.
  * **SemiBCD** (*Liu et al., IEEE TGRS 2026*) — Semi-supervised VLM pseudo-supervision.
  * **KGBDCNet** (*Wang et al., ISPRS JPRS 2026*) — Keyword-guided damage captioning + Space-Scan Mamba.
  * **SAM2-CD** (*Qin et al., IEEE JSTARS 2025*) — Bi-temporal change detection with SAM2.
  * **BiSAM-CD** (*Qin et al., IEEE TGRS 2025*) — Zero-shot bi-temporal change detection.
  * **Improved Mask R-CNN** (*Han et al., CIS 2022*) — Anchor-optimized building detection.
* **Geospatial & Vector Native**:
  * Direct GeoTIFF raster reading + EPSG:3857 GeoJSON building polygon rasterization (handles multi-polygons, interior courtyards, and holes).
  * Automated morphological boundary ($B_{gt}$) and Harris corner ($C_{gt}$) generation.
* **Strict Evaluation Protocol**:
  * Spatial-block quadrant splitting preventing geographic spatial leakage into unseen test zones.
  * Evaluation metrics: IoU, F1 / Dice, Precision, Recall, Overall Accuracy, Boundary F1 (B-F1), Latency (ms), and FPS.

---

## 📂 Repository Structure

```text
├── ag_models/                # Core modular deep learning library (blocks, attention, losses, layers)
├── datasets/
│   ├── __init__.py
│   └── custom_datasets.py    # GeoTIFF/GeoJSON and custom dataset loaders
├── models/
│   └── __init__.py           # Unified model catalog & factory (create_method_model)
├── sample_custom_data/       # Lightweight sample dataset for rapid verification
├── train.py                  # Primary training entrypoint
├── evaluate.py               # Evaluation & benchmark metric computation
├── predict.py                # Single-image and pair inference script
├── run_pipeline.py           # Unified end-to-end multi-method execution pipeline
├── run_fair_comparison_2026.py # Fair benchmark evaluation across all 2026 models
├── inspect_results.py        # Benchmark results viewer
├── requirements.txt          # Python dependencies
├── setup.py                  # Standard package configuration
├── pyproject.toml
├── LICENSE                   # MIT License
└── README.md
```

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/haile493/SAM-CEBNet.git
cd SAM-CEBNet

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

---

## 🚀 Quickstart: 1-Minute Verification

Verify your installation using the included mini sample dataset:

```bash
# 1. Quick train test on sample custom data (1 epoch)
python train.py --model sam_cebnet --data_dir ./sample_custom_data --epochs 1 --batch_size 2 --img_size 128

# 2. Evaluate model
python evaluate.py --model sam_cebnet --data_dir ./sample_custom_data --img_size 128

# 3. Predict building footprints on sample image
python predict.py --model sam_cebnet --image sample_custom_data/images/sample_000.png --output outputs/sample_000_pred.png
```

---

## 📊 Dataset Setup

### A. Hue City GeoTIFF + GeoJSON Dataset
Place satellite imagery tiles and vector polygon annotations in `./dataset/`:
```text
dataset/
├── image-hue_01_01/
│   ├── image-hue_01_01.tif        # High-resolution RGB GeoTIFF (EPSG:3857)
│   └── image-hue_01_01.geojson    # Vector building polygons (EPSG:3857)
└── ...
```

### B. Standard Flat Directory Layout
For custom datasets, organize files into standard directory structures:
* **Single-Image Building Extraction**:
  ```text
  my_dataset/
  ├── images/       # Aerial/satellite images (.png, .jpg, .tif)
  └── masks/        # Binary building masks (0: background, 255: building)
  ```
* **Bi-Temporal Change Detection**:
  ```text
  my_bitemporal_data/
  ├── A/            # Pre-event / T1 imagery
  ├── B/            # Post-event / T2 imagery
  └── label/        # Binary change mask (0: unchanged, 255: changed)
  ```

---

## 🏋️ Training

### 1. Train Proposed SAM-CEBNet
```bash
# Train SAM-CEBNet from scratch on Hue dataset
python train.py --model sam_cebnet --data_dir ./dataset --epochs 50 --batch_size 4 --img_size 256

# Train SAM-CEBNet with Pretrained Foundation Backbone
python train.py --model sam_cebnet --data_dir ./dataset --pretrained --epochs 50 --batch_size 4 --img_size 256

# Train SAM3-CEBNet
python train.py --model sam3_cebnet --data_dir ./dataset --pretrained --epochs 50 --batch_size 4 --img_size 256
```

### 2. Train Benchmark SOTA Models
```bash
# Train CBRNet (2026)
python train.py --model cbrnet --data_dir ./dataset --epochs 50 --batch_size 4 --img_size 256

# Train EdgeRefNet (2026)
python train.py --model edgerefnet --data_dir ./dataset --epochs 50 --batch_size 4 --img_size 256

# Train GMFIMamba (2026)
python train.py --model gmfimamba --data_dir ./dataset --epochs 50 --batch_size 4 --img_size 256

# Train SemiBCD (2026)
python train.py --model semibcd --data_dir ./dataset --epochs 50 --batch_size 4 --img_size 256

# Train CGCCE-Net (2026)
python train.py --model cgcce_net --data_dir ./dataset --epochs 50 --batch_size 4 --img_size 256
```

---

## 📈 Evaluation & Benchmark Comparison

### 1. Evaluate a Checkpoint
```bash
python evaluate.py --model sam_cebnet --checkpoint outputs/sam_cebnet/best_model.pth --data_dir ./dataset --split unseen_test
```

### 2. Run Comprehensive 2026 Fair Comparison Benchmark
Runs all 2026 architectures under identical experimental conditions (Scratch, Pretrained Backbone, WHU Transfer):
```bash
python run_fair_comparison_2026.py --data_dir ./dataset --epochs 50 --img_size 256
```
Inspect summarized comparison results:
```bash
python inspect_results.py
```

---

## 🔮 Inference & Prediction

Run building footprint extraction on a single image:
```bash
python predict.py \
    --model sam_cebnet \
    --checkpoint outputs/sam_cebnet/best_model.pth \
    --image path/to/aerial_image.tif \
    --output outputs/predictions/prediction_mask.png
```

Run bi-temporal change detection on an image pair:
```bash
python predict.py \
    --model sam2_cd \
    --image1 path/to/pre_event.png \
    --image2 path/to/post_event.png \
    --output outputs/predictions/change_mask.png
```

---

## 📜 Citation

If you find this code or dataset useful in your research, please cite:

```bibtex
@article{le2026samcebnet,
  title={SAM-CEBNet: Foundation-Guided Corner and Edge Refinement Network Family (SAM-2 \& SAM-3) for High-Precision Building Footprint Extraction in High-Resolution Remote Sensing Imagery},
  author={Thi-Mai-Anh Dinh, Nguyen-Ngoc-Nguyen To, Luong-Huy-Thong Huynh, Dai-Hiep Tran, Thi-Thu-Hien Pham and Thanh-Hai Le},
  journal={submitting},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
