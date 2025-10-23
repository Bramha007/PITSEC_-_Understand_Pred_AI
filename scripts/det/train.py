# scripts/det/train.py

# Trains A Faster R-CNN Object Detector On A VOC-Style Dataset With Configurable YAML Settings
# Supports Deterministic Training, AMP, Gradient Clipping, And Customizable Optimizers Or Schedulers
# Logs Progress And Saves Checkpoints With mAP Evaluation Each Epoch

from __future__ import annotations

# Force Deterministic CuBLAS Workspace For Reproducibility
import os
if "CUBLAS_WORKSPACE_CONFIG" not in os.environ:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Suppress Non-Deterministic Roi_Align Warnings
import warnings
warnings.filterwarnings(
    "ignore",
    "roi_align_backward_kernel does not have a deterministic implementation"
)

# Standard Library
import os
import time
import argparse
import json
from pathlib import Path
from contextlib import nullcontext
from collections import defaultdict, deque
from typing import Dict, List

# Third-Party
import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm
import yaml

# Local Modules
from src.utils.echo import echo_line
from src.utils.determinism import set_seed, worker_init_fn, make_generator
from src.data.voc import paired_image_xml_list
from src.data.det import SquaresDetectionDataset, collate_fn
from src.transforms.det import Compose, ToTensor, ClampBoxes, RandomHorizontalFlip
from src.models.det_fasterrcnn import build_fasterrcnn
from src.metrics.det import evaluate_ap_by_size


# Smoothed Metric Buffer For Averaging Losses
class Smoothed:
    def __init__(self, window: int = 200):
        self.buf = deque(maxlen=window)
        self.total = 0.0
        self.count = 0

    def update(self, x: float):
        x = float(x)
        self.buf.append(x)
        self.total += x
        self.count += 1

    @property
    def avg(self) -> float:
        return 0.0 if not self.buf else sum(self.buf) / len(self.buf)

    @property
    def global_avg(self) -> float:
        return 0.0 if self.count == 0 else self.total / self.count


# Load YAML Configuration File
def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Build DataLoaders For Training And Validation
def _build_loaders(cfg: dict, gen: torch.Generator):
    data = cfg.get("data", {}) or {}
    root = Path(data.get("root", "."))
    limit = data.get("limit_per_split", None)
    img_tr, img_ev, ann = root / "train", root / "val", root / "annotations"

    # Pair Images With Annotations
    tr_pairs = paired_image_xml_list(img_tr, ann, limit=limit)
    ev_pairs = paired_image_xml_list(img_ev, ann, limit=limit)

    # Define Transforms
    tfm_tr = Compose([ToTensor(), RandomHorizontalFlip(0.5), ClampBoxes()])
    tfm_ev = Compose([ToTensor(), ClampBoxes()])

    # Build Datasets
    ds_tr = SquaresDetectionDataset(tr_pairs, transforms=tfm_tr)
    ds_ev = SquaresDetectionDataset(ev_pairs, transforms=tfm_ev)

    # Training And Evaluation Configurations
    tr_cfg, ev_cfg = cfg.get("train", {}) or {}, cfg.get("eval", {}) or {}

    # Build PyTorch DataLoaders
    dl_tr = DataLoader(
        ds_tr,
        batch_size=int(tr_cfg.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(tr_cfg.get("num_workers", 2)),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=gen,
    )
    dl_ev = DataLoader(
        ds_ev,
        batch_size=int(ev_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(ev_cfg.get("num_workers", 2)),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=gen,
    )
    return ds_tr, ds_ev, dl_tr, dl_ev


# Build Faster R-CNN Model
def _build_model(cfg: dict, device: torch.device):
    mcfg = cfg.get("model", {}) or {}
    model = build_fasterrcnn(
        num_classes=int(mcfg.get("num_classes", 2)),
        anchor_sizes=mcfg.get("anchor_sizes", None),
        anchor_aspect_ratios=mcfg.get("anchor_aspect_ratios", None),
        weights=mcfg.get("weights", None),
        trainable_backbone_layers=mcfg.get("trainable_backbone_layers", None),
        local_backbone_weights=mcfg.get("local_backbone_weights", None),
    )
    model.to(device)
    return model


# Build Optimizer And Scheduler
def _build_optim_sched(model: torch.nn.Module, cfg: dict):
    tr_cfg = cfg.get("train", {}) or {}
    lr = float(tr_cfg.get("lr", 1e-4))
    wd = float(tr_cfg.get("weight_decay", 1e-4))
    opt_name = (tr_cfg.get("optimizer", "adamw") or "adamw").lower()

    # Select Optimizer
    optimizer = (
        torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
        if opt_name == "sgd"
        else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    )

    # Optional Scheduler
    sc_cfg = tr_cfg.get("scheduler", {}) or {}
    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sc_cfg.get("step_size", 10)),
            gamma=float(sc_cfg.get("gamma", 0.1)),
        )
        if (sc_cfg.get("name") or "").lower() == "step"
        else None
    )
    return optimizer, scheduler


# Get Current Learning Rate
def _lr(optimizer: torch.optim.Optimizer) -> float:
    for g in optimizer.param_groups:
        if "lr" in g:
            return float(g["lr"])
    return 0.0


# Automatic Mixed Precision Context Manager
def _amp_ctx(enabled: bool, device: torch.device):
    return autocast(device_type="cuda", dtype=torch.float16) if enabled and device.type == "cuda" else nullcontext()


# Train One Epoch
def train_one_epoch(model, optimizer, scaler, dl, device, log_every, epoch, epochs, max_norm: float | None):
    model.train()
    meters = defaultdict(lambda: Smoothed(window=200))
    bar = tqdm(dl, total=len(dl), desc=f"E{epoch:03d}/{epochs:03d}", leave=True)

    for i, (imgs, targets) in enumerate(bar, start=1):
        # Move Images And Targets To Device
        imgs = [im.to(device, non_blocking=True) for im in imgs]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        # Forward Pass With Optional AMP
        with _amp_ctx(enabled=(scaler is not None), device=device):
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

        # Backward Pass And Optimization
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            if max_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        # Update Smoothed Losses
        meters["loss"].update(float(loss.detach().item()))
        for k, v in loss_dict.items():
            meters[k].update(float(v.detach().item()))

        # Logging
        if (i % max(1, log_every)) == 0 or i == len(dl):
            echo_line(
                "DET_TRAIN",
                {
                    "epoch": f"{epoch}/{epochs}",
                    "iter": f"{i}/{len(dl)}",
                    "lr": _lr(optimizer),
                    "loss": meters["loss"].avg,
                    **{k: meters[k].avg for k in sorted(loss_dict.keys())},
                },
                order=["epoch", "iter", "lr", "loss"],
            )

    return meters["loss"].global_avg


# Evaluate Model On Validation Set
@torch.inference_mode()
def evaluate(model, dl, device, out_dir: Path, tag="val") -> Dict[str, float]:
    metrics, _, _ = evaluate_ap_by_size(model, dl, device, out_dir=str(out_dir), tag=tag)

    # Logging
    echo_line(
        "DET_VAL",
        {
            "split": tag,
            "ap50": float(metrics.get("ap50_global", 0.0)),
            "ap": float(metrics.get("ap_global", 0.0)),
            "ar": float(metrics.get("ar_global", 0.0)),
        },
        order=["split", "ap50", "ap", "ar"],
    )

    # Save JSON Metrics
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


# Prepare Output Directory And Save Config
def _prepare_out_dir(cfg: dict) -> Path:
    out_dir = Path(cfg.get("out_dir", "outputs/det/train"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_dir


# Main Training Loop
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    args = parser.parse_args()

    # Load Config And Set Seed
    cfg = _load_cfg(args.cfg)
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    gen = make_generator(seed)

    # Device Selection
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    echo_line("INFO", {"deterministic": True, "device": str(device), "seed": seed})

    # Prepare Output Directory
    out_dir = _prepare_out_dir(cfg)

    # Build Model, Optimizer, And Scheduler
    model = _build_model(cfg, device)
    optimizer, scheduler = _build_optim_sched(model, cfg)

    # Build DataLoaders
    ds_tr, ds_ev, dl_tr, dl_ev = _build_loaders(cfg, gen)

    # AMP And Gradient Clipping
    tr_cfg = cfg.get("train", {}) or {}
    use_amp = bool(tr_cfg.get("amp", True)) and (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp) if use_amp else None
    max_norm = float(tr_cfg.get("grad_clip", 0.0)) or None

    # Training Loop
    epochs = int(tr_cfg.get("epochs", 10))
    log_every = int(tr_cfg.get("log_every", 50))
    best_val = float("-inf")
    best_file = out_dir / "det_fasterrcnn.best.ckpt"
    last_file = out_dir / "det_fasterrcnn.last.ckpt"
    monitor = (cfg.get("early_stopping", {}) or {}).get("monitor", "ap50_global")

    for ep in range(1, epochs + 1):
        train_loss = train_one_epoch(model, optimizer, scaler, dl_tr, device, log_every, ep, epochs, max_norm)
        val_stats = evaluate(model, dl_ev, device, out_dir)

        # Save Best Model
        cur_val = float(val_stats.get(monitor, -float("inf")))
        if cur_val > best_val:
            best_val = cur_val
            torch.save(model.state_dict(), best_file)
            echo_line("INFO", {"epoch": ep, "best_val": best_val, "saved": str(best_file)})

        # Save Last Model
        torch.save(model.state_dict(), last_file)
        echo_line("INFO", {"epoch": ep, "saved_last_model": str(last_file)})

        # Step Scheduler
        if scheduler:
            scheduler.step()

    echo_line("INFO", {"best_val": best_val, "msg": "Training Completed", "out_dir": str(out_dir)})


# Entry Point
if __name__ == "__main__":
    main()
