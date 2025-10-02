# Understand Pred AI

Two tasks live in this repo:

- **CLS** — classification/regression of short/long box sides (ResNet backbones)
- **DET** — object detection with Faster R‑CNN (ResNet‑50 backbone by default)

This README is updated to reflect the **version‑locked installs** that the project actually uses: `base.txt` + `cpu.txt` **or** `gpu.txt`.


## 1) Requirements & Installation

- **Python**: pinned to **3.11.9** (see `pyproject.toml`)
- **PyTorch / TorchVision**: hard‑locked per platform
- Other libs are pinned in `base.txt`

Choose **one** of the two setups below.

### CPU‑only
```bash
# from repo root
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install --upgrade pip

pip install -r requirements/cpu.txt
pip install -e .
```

### GPU (CUDA 12.4 wheels)
```bash
# from repo root
python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install --upgrade pip

pip install -r requirements/gpu.txt
pip install -e .
```

> **Why two files?**  
> `base.txt` carries framework‑agnostic pins (numpy, pillow, tqdm, captum, etc.).  
> `cpu.txt` and `gpu.txt` pin the exact Torch/TorchVision wheels for each target. You should **not** mix them.


## 2) Project Layout

```
src/
  constants/     # norms, size buckets
  utils/         # config loader, determinism, echo
  data/          # VOC parsing, split helpers
  transforms/    # cls_transforms.py, det_transforms.py
  models/        # cls_resnet.py, det_fasterrcnn.py
  metrics/       # cls.py, det.py, ar.py
  xai/           # (Captum etc.)
scripts/
  cls/           # train.py, eval.py
  det/           # train.py, eval.py
configs/
  cls/           # sample *.yaml
  det/           # sample *.yaml
weights/         # (optional) local backbone weights, e.g., resnet50-0676ba61.pth
outputs/         # training/eval runs
```


## 3) Quickstart

### DET (Faster R‑CNN)

**Train**
```bash
python -m scripts.det.train --config configs/det/test.yaml
```

**Eval**
```bash
python -m scripts.det.eval --config configs/det/test.yaml
```

**Config knobs of interest (DET)**

```yaml
model:
  num_classes: 2
  # When present, we will use local backbone weights (no web download):
  local_backbone_weights: weights/resnet50-0676ba61.pth
  # Optional (keep None to avoid network):
  weights: null
  trainable_backbone_layers: 3
  anchor_sizes: [32, 64, 128, 256, 512]
  anchor_aspect_ratios: [0.5, 1.0, 2.0]
```

> The training/eval scripts now **pass through** `model.local_backbone_weights` to the model builder.  
> If the key is omitted, behavior is unchanged (no local loading).


### CLS (ResNet classifier)

**Train**
```bash
python -m scripts.cls.train --config configs/cls/test.yaml
```

**Eval**
```bash
python -m scripts.cls.eval --config configs/cls/test.yaml
```

**Config knobs of interest (CLS)**

```yaml
model:
  backbone: resnet18      # or resnet34/50 etc., as supported in cls_resnet.py
  out_dim: 2              # (backward‑compatible: `num_classes` also works)
  pretrained: true
  final_act: null         # "softmax", "sigmoid", or null for logits
  p_drop: 0.0
```

> The CLS scripts use a **single explicit constructor** (`build_resnet_classifier`) and accept
> both `out_dim` and `num_classes` for backwards compatibility.


## 4) Reproducibility

Set seeds and deterministic behavior in configs; the utilities in `src/utils/determinism.py` are called by the scripts.

```yaml
seed: 1337
train:
  amp: true         # Automatic Mixed Precision (where available)
  num_workers: 4
  batch_size: 4
```

Log cadence, gradient clipping, and early stopping are all set in the corresponding `train` / `early_stopping` sections per task.


## 5) Metrics

**CLS**: see `src/metrics/cls.py` (MAE/MSE/RMSE/MAPE/SMAPE/R2/Pearson/bucket accuracy).  
**DET**: see `src/metrics/det.py` (AP@0.5; per‑image prediction/GT format documented in the file).


## 6) Notes & Conventions

- Avoid `print()`. Use the shared `echo` helper for consistent logging.
- Transforms live under `src/transforms/`; keep experimental changes in a separate module and import them explicitly in configs.
- Do not add a monolithic `requirements.txt`. Use the split files: `base.txt` + `cpu.txt` **or** `gpu.txt`.
- Python pin is enforced via `pyproject.toml` (3.11.9).


## 7) Troubleshooting

- **Pip resolves different Torch versions**  
  Ensure you used the correct pair: `base.txt` + `cpu.txt` **or** `gpu.txt` (not both).

- **CUDA mismatch**  
  The GPU wheels assume **CUDA 12.4**. If your system driver/toolkit differs, prefer the CPU setup or adapt the wheels accordingly.

- **Weights download attempts**  
  Keep `model.weights: null` to avoid network and use `model.local_backbone_weights` for local files instead.


## 8) Re-running a clean env

```bash
rm -rf .venv
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r base.txt -r cpu.txt   # or: -r base.txt -r gpu.txt
pip install -e .
```

---

Happy training! If anything feels off or you need a ready‑to‑apply patch for configs/scripts, open an issue or ping the maintainer.
