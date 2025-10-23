# scripts/det/test.py

# Evaluates A Faster R-CNN Object Detector On A VOC-Style Dataset With Configurable YAML Settings
# Supports Deterministic Evaluation And Configurable Weights Loading
# Logs Detection Metrics (AP, AR) And Saves Results To Output Directory

from __future__ import annotations

# Standard Library
import os
import argparse
import json
from pathlib import Path

# Third-Party
import torch
from torch.utils.data import DataLoader
import yaml

# Local Modules
from src.utils.echo import echo_line
from src.utils.determinism import set_seed, worker_init_fn, make_generator
from src.data.voc import paired_image_xml_list
from src.data.det import SquaresDetectionDataset, collate_fn
from src.transforms.det import Compose, ToTensor, ClampBoxes
from src.models.det_fasterrcnn import build_fasterrcnn
from src.metrics.det import evaluate_ap_by_size


# Load YAML Configuration File
def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Build DataLoader For Testing
def _build_loader(cfg: dict, gen: torch.Generator):
    data = cfg.get("data", {}) or {}
    root = Path(data.get("root", "."))
    limit = data.get("limit_per_split", None)
    img_te, ann = root / "test", root / "annotations"

    # Pair Images With Annotations
    te_pairs = paired_image_xml_list(img_te, ann, limit=limit)

    # Define Transform
    tfm_te = Compose([ToTensor(), ClampBoxes()])

    # Build Dataset
    ds_te = SquaresDetectionDataset(te_pairs, transforms=tfm_te)

    # Test Configurations
    te_cfg = cfg.get("test", {}) or {}

    # Build PyTorch DataLoader
    dl_te = DataLoader(
        ds_te,
        batch_size=int(te_cfg.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(te_cfg.get("num_workers", 2)),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=gen,
    )
    return ds_te, dl_te


# Build Faster R-CNN Model
def _build_model(cfg: dict, device: torch.device):
    mcfg = cfg.get("model", {}) or {}
    model = build_fasterrcnn(
        num_classes=int(mcfg.get("num_classes", 2)),
        anchor_sizes=mcfg.get("anchor_sizes", None),
        anchor_aspect_ratios=mcfg.get("anchor_aspect_ratios", None),
        weights=None,
        trainable_backbone_layers=None,
        local_backbone_weights=mcfg.get("local_backbone_weights", None),
    )
    model.to(device)
    return model


# Load Model Weights
def _load_weights(model: torch.nn.Module, cfg: dict, device: torch.device):
    tcfg = cfg.get("test", {}) or {}
    wpath = tcfg.get("weights", None)
    if not wpath:
        echo_line("WARN", {"msg": "No Weights Path Provided"})
        return model

    wpath = Path(wpath)
    if not wpath.exists():
        echo_line("WARN", {"msg": f"Weight File Not Found: {wpath}"})
        return model

    state = torch.load(wpath, map_location=device)
    model.load_state_dict(state)
    echo_line("INFO", {"loaded_weights": str(wpath)})
    return model


# Evaluate Model On Test Set
@torch.inference_mode()
def evaluate(model, dl, device, out_dir: Path, tag="test"):
    metrics, _, _ = evaluate_ap_by_size(model, dl, device, out_dir=str(out_dir), tag=tag)

    # Logging
    echo_line(
        "DET_TEST",
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
    out_dir = Path(cfg.get("out_dir", "outputs/det/test"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_dir


# Main Evaluation Loop
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

    # Build Model And Load Weights
    model = _build_model(cfg, device)
    model = _load_weights(model, cfg, device)
    model.eval()

    # Build Test DataLoader
    _, dl_te = _build_loader(cfg, gen)

    # Evaluate Model
    evaluate(model, dl_te, device, out_dir)


# Entry Point
if __name__ == "__main__":
    main()
