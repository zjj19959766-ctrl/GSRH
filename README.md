# GSRH: Geometric-Semantic Regulated Hypergraph for Tiny Object Detection

Official implementation of **GSRH** on top of **Ultralytics YOLO (YOLOv11-style codebase)**.

## Overview
GSRH targets tiny object detection in remote sensing scenes with dense layout, cluttered background, and directional ambiguity.

This repository keeps the Ultralytics training/inference pipeline, and introduces two core ideas:
- **GPS (Geometric Prior Synthesis)**: geometry-aware directional prior enhancement.
- **SRC (Semantic Reliability Controller)**: reliability-gated semantic propagation.

## Key Innovations
### 1) GPS: Geometric Prior Synthesis
- Location: `ultralytics/nn/modules/block.py`
- Related classes: `DirectionalShift`, `SingleAnglePrior`, `GPS`
- Purpose: inject directional priors (default 0 deg and 90 deg) into feature enhancement.

### 2) SRC: Semantic Reliability Controller
- Location: `ultralytics/nn/modules/block.py`
- Related class: `SRC`
- Purpose: estimate reliability from self/neighbor statistics and adaptively gate features.

### 3) Architecture-Level Integration
- Location: `ultralytics/cfg/models/12/yolo12*.yaml`
- Core change: use `A2C2f` blocks in backbone/neck for stronger area-aware representation.

## Repository Structure
```text
.
├─ ultralytics/
│  ├─ nn/modules/block.py                # GPS/SRC and related modules
│  └─ cfg/models/12/yolo12*.yaml         # GSRH-oriented model configs
├─ docs/
├─ examples/
├─ tests/
└─ README.md
```

## Environment
Recommended:
- Python 3.10
- PyTorch >= 2.0 (CUDA matched with your machine)
- OS: Linux/Windows

Create env:
```bash
conda create -n gsrh python=3.10 -y
conda activate gsrh
```

Install PyTorch (example for CUDA 12.1):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Install dependencies:
```bash
pip install ultralytics opencv-python pyyaml scipy tqdm matplotlib psutil pillow thop
```

## Data Preparation
Use Ultralytics dataset YAML format:

```yaml
# data.yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test
names:
  0: class0
  1: class1
```

## Training
Main GSRH line (detect):
```bash
yolo task=detect mode=train \
  model=ultralytics/cfg/models/12/yolo12.yaml \
  data=/path/to/data.yaml \
  epochs=300 imgsz=640 batch=16 device=0 workers=8
```

Resume training:
```bash
yolo task=detect mode=train resume model=runs/detect/train/weights/last.pt
```

## Evaluation
```bash
yolo task=detect mode=val \
  model=runs/detect/train/weights/best.pt \
  data=/path/to/data.yaml \
  imgsz=640 batch=16 device=0
```

## Inference
Images/folder:
```bash
yolo task=detect mode=predict \
  model=runs/detect/train/weights/best.pt \
  source=/path/to/images \
  imgsz=640 conf=0.25 device=0
```

Video:
```bash
yolo task=detect mode=predict \
  model=runs/detect/train/weights/best.pt \
  source=/path/to/video.mp4 \
  imgsz=640 conf=0.25 device=0
```

## Outputs
- Training runs: `runs/detect/train*`
- Best weights: `runs/detect/train*/weights/best.pt`
- Inference results: `runs/detect/predict*`

## Notes
- `MANet` in `block.py` is experimental; reproduction is recommended with `yolo12*.yaml` main configs.
- For stable reproduction, fix random seed, data split, and report mAP50/mAP50-95/FPS together.

## Citation
If this project helps your research, please cite:

```bibtex
@article{zheng2025gsrh,
  title={Geometric-Semantic Regulated Hypergraph for Tiny Object Detection},
  author={Zheng, JinJie and Zhuang, Jingyi and Sa, Baihui and Xiang, Wenjie and Zhang, Zetao and Zhu, Jianqing},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2025}
}
```

## License
This project is built on the Ultralytics codebase. Please follow:
- This repository's license file
- Upstream Ultralytics license and usage terms

## Acknowledgements
We thank the Ultralytics team and the remote sensing tiny-object detection community for open-source contributions.
