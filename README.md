# Understanding Predictive AI  
![Python](https://img.shields.io/badge/python-3.10%2B-blue)  
![PyTorch](https://img.shields.io/badge/pytorch-2.6.0%2B-red)  
![Torchvision](https://img.shields.io/badge/torchvision-0.21.0-green)  
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)  

**Understanding Predictive AI** is a research framework for studying how deep learning models **predict object properties** on a simplified dataset of geometric objects.  
It goes beyond black-box performance metrics by analyzing **prediction accuracy across object scales and aspect ratios**, and by incorporating **explainable AI (XAI) techniques** to interpret model behavior.  

---

## 🧭 Research Motivation

AI models often act as *black boxes*. In this project, we use a **controlled dataset of squares and rectangles** (AMSL dataset) to systematically explore:  

- How predictive AI architectures (ResNet, Faster R-CNN, SSD, etc.) handle **object size prediction**.  
- How **training variations** (different sizes, number of shapes per image, dataset size) affect performance.  
- How well models generalize to unseen shapes.  
- How **XAI methods** (Grad-CAM, Captum, LRP, DiCE) can make predictions interpretable.  

This controlled environment helps us **trace model decisions**, identify biases, and understand the conditions under which predictive AI performs reliably.  

---

## 📂 Project Structure

```
.
├── scripts/
│   ├── cls/          # Training & evaluation scripts for classification/regression
│   │   ├── train.py
│   │   └── eval.py
│   └── det/          # Training & evaluation scripts for detection
│       ├── train.py
│       └── eval.py
│
├── src/
│   ├── constants/    # Shared constants (size buckets, normalization stats)
│   ├── data/         # Dataset loaders (VOC XML, classification, detection)
│   ├── metrics/      # Evaluation (MAE/MSE/MAPE, AP by size, AP by AR)
│   ├── models/       # Model builders (ResNet backbone, Faster R-CNN)
│   ├── transforms/   # Data augmentations for cls/det
│   └── utils/        # Helpers (determinism, config, logging)
│
├── configs/          # Example YAML configs (edit for dataset/model)
├── outputs/          # Training logs & checkpoints (created automatically)
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/understanding-predictive-ai.git
   cd understanding-predictive-ai
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # (Linux/Mac)
   .venv\Scripts\activate      # (Windows)

   pip install -r requirements.txt
   ```

---

## 📊 Data Setup

The framework expects a dataset in **VOC-style format**:

```
data_root/
├── train/         # Training images (.bmp)
├── val/           # Validation images (.bmp)
├── test/          # (optional) Test images
└── annotations/   # VOC XML annotation files
```

Each image must have a corresponding XML file (same stem).

---

## 🚀 Usage

### 1. Train Classifier (Size Regression)
```bash
python scripts/cls/train.py --config configs/default.yaml
```

### 2. Evaluate Classifier
```bash
python scripts/cls/eval.py --config configs/default.yaml --split val
```

---

### 3. Train Detector (Faster R-CNN)
```bash
python scripts/det/train.py --config configs/default.yaml
```

### 4. Evaluate Detector
```bash
python scripts/det/eval.py --config configs/default.yaml --split val --ar
```

---

## 📐 Buckets & Metrics

- **Size Buckets** (XS → XL) allow analyzing how well models perform on small vs large objects.  
- **Aspect Ratio Buckets** (near-square, moderate, skinny) capture how predictions vary with shape elongation.  

---

## 🧪 Explainability (XAI)

We integrate open-source XAI tools for model interpretation:
- **Layer-based**: Grad-CAM, Captum, LRP.  
- **Counterfactuals**: DiCE.  

This ensures results are not just predictive, but also interpretable.  

---

## 🧪 Reproducibility
- Random seeds controlled (`src/utils/determinism.py`)  
- AMP (automatic mixed precision) supported  
- Checkpoints:  
  - `cls_resnet.best.ckpt` / `cls_resnet.last.ckpt`  
  - `det_fasterrcnn.best.ckpt` / `det_fasterrcnn.last.ckpt`  

---

## 📈 Example Results

Classifier:
```
VAL | MAE_short=2.13 | MAE_long=4.57 | MAPE_short=0.12 | MAPE_long=0.09 | N=500
```

Detector:
```
VAL | ap50=0.72 | npos=1340
VAL ap50_by_size | XS=0.55 | S=0.68 | M=0.74 | L=0.80 | XL=0.77
VAL ap50_by_AR   | near_square=0.73 | moderate=0.71 | skinny=0.65
```

---

## 📜 License
MIT License – see [LICENSE](LICENSE) for details.
