# Understand Pred AI

Two Tasks Live In This Repo:

- **CLS** — Classification/Regression Of Short/Long Box Sides (ResNet Backbones)
- **DET** — Object Detection With Faster R‑CNN (ResNet‑50 Backbone By Default)

---

## 1) Requirements & Installation

- **Python**: **3.11.9** (See `pyproject.toml`)  
- **PyTorch / TorchVision**: Hard‑Locked Per Platform  
- Other Libs Are Pinned In `base.txt`

Choose **one** setup below.

### CPU‑Only
```bash
# From Repo Root
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install --upgrade pip
pip install -r requirements/cpu.txt
```

### GPU (CUDA 12.4 Wheels)
```bash
# From Repo Root
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install --upgrade pip
pip install -r requirements/gpu.txt
```

> **Why Two Files?**  
> `base.txt` carries framework-agnostic pins (numpy, pillow, tqdm, captum, etc.).  
> `cpu.txt` and `gpu.txt` pin exact Torch/TorchVision wheels per platform. Do **not** mix them.

---

## 2) Project Layout

```
configs/
  cls/           # sample *.yaml
  det/           # sample *.yaml

outputs/         # Training/Eval Runs

scripts/
  cls/           # train.py, test.py
  det/           # train.py, test.py

src/
  constants/     # Norms, Size Buckets
  data/          # VOC Parsing, Split Helpers
  metrics/       # cls.py, det.py, ar.py
  models/        # cls_resnet.py, det_fasterrcnn.py
  transforms/    # cls.py, det.py, det_experimental.py
  utils/         # Config Loader, Determinism, Echo
  xai/           # Captum etc.

weights/         # Optional local backbone weights
```

---

## 3) Quickstart

### DET (Faster R‑CNN)

**Train**
```bash
python -m scripts.det.train --cfg configs/det/default.yaml
python -m scripts.det.train --cfg configs/det/default_cust.yaml
```

**Test**
```bash
python -m scripts.det.test --cfg configs/det/default.yaml
python -m scripts.det.test --cfg configs/det/default_cust.yaml
```

**Config Knobs Of Interest (DET)**

```yaml
model:
  num_classes: 2
  local_backbone_weights: weights/resnet50-0676ba61.pth  # Optional local backbone
  weights: null           # keep null to avoid network downloads
  trainable_backbone_layers: 3
  anchor_sizes: [32, 64, 128, 256, 512]                 # default; override in custom
  anchor_aspect_ratios: [0.5, 1.0, 2.0]                # default; override in custom
train:
  epochs: 50
  batch_size: 64
  num_workers: 4
  lr: 0.005
  weight_decay: 0.0005
  optimizer: sgd
  scheduler:
    name: step
    step_size: 10
    gamma: 0.1
  amp: true
  grad_clip: 1.0
early_stopping:
  monitor: ap50_global
```

> **Custom Anchors** can be set per feature map in `default_cust.yaml`. Example:
```yaml
anchor_sizes: [[16],[32],[64],[128],[256]]
anchor_aspect_ratios:
  - [0.5, 1.0, 2.0]
  - [0.5, 1.0, 2.0]
  - [0.5, 1.0, 2.0]
  - [0.5, 1.0, 2.0]
  - [0.5, 1.0, 2.0]
```

---

### CLS (ResNet Classifier)

**Train**
```bash
python -m scripts.cls.train --cfg configs/cls/default.yaml
```

**Test**
```bash
python -m scripts.cls.test --cfg configs/cls/default.yaml
```

**Config Knobs Of Interest (CLS)**

```yaml
model:
  model_name: resnet18       # or resnet34/50
  out_dim: 2                 # Output dimension (2 regression targets)
  pretrained: true
  final_act: null            # "softmax", "sigmoid", or null for logits
  p_drop: 0.0                # Dropout probability
train:
  epochs: 50
  batch_size: 64
  num_workers: 4
  lr: 0.001
  weight_decay: 0.0001
  loss: mse
  amp: true
  rot_deg: 5.0               # Optional rotation augmentation
  clip_grad_norm: 1.0
eval:
  batch_size: 64
  num_workers: 4
  run_during_train: true
metrics:
  which: [mae, mse, rmse, mape, smape, r2, pearson, bucket_acc, bucket_mae, bucket_rmse]
  bucket_on: long
  bucket_edges: [0, 32, 64, 96, 128, 160, 99999]
early_stopping:
  monitor: overall_rmse
out_dir: outputs/cls/default
```

---

## 4) Reproducibility

- Set `seed` in config. Scripts call `src/utils/determinism.py`.
- Deterministic CuDNN and torch RNG behavior is enforced.

```yaml
seed: 1337
train:
  amp: true
  num_workers: 4
  batch_size: 64
```

- Gradient clipping, logging cadence, and early stopping are handled via `train` / `early_stopping` sections.

---

## 5) Metrics

- **CLS**: see `src/metrics/cls.py`  
  (MAE, MSE, RMSE, MAPE, SMAPE, R², Pearson, bucket accuracy/MAE/RMSE)
- **DET**: see `src/metrics/det.py`  
  (AP@0.5; per-image prediction/GT format documented in file)

> Bucket accuracy uses predefined size buckets in `src/constants/`.

---

## 6) Notes & Conventions

- Avoid `print()`. Use `echo` for consistent logs.
- Transforms live under `src/transforms/`. Experimental DET transforms must be imported explicitly.
- Do not mix requirements files.
- DET best vs last model checkpoints:
  - `outputs/det/train/det_fasterrcnn.best.ckpt`
  - `outputs/det/train/det_fasterrcnn.last.ckpt`

---

## 7) Troubleshooting

- **Torch version mismatch**: ensure CPU/GPU requirements are not mixed.
- **CUDA mismatch**: GPU wheels assume CUDA 12.4.
- **Weights download attempts**: keep `model.weights: null`; use `model.local_backbone_weights` for local files.

---

## 8) Re-running a Clean Env

```bash
rm -rf .venv
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r base.txt -r cpu.txt   # or: -r base.txt -r gpu.txt
pip install -e .
```

