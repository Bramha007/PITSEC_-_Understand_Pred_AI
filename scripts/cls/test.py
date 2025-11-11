# scripts/cls/test.py

# Tests The Classification/Regression Model On Squares Dataset
# Supports Deterministic Evaluation, AMP, And Configurable YAML Settings
# Logs Metrics And Saves JSON/CSV Outputs

from __future__ import annotations

# Standard Library
import os, csv, json
from pathlib import Path
from typing import Dict, Any
from contextlib import nullcontext
import argparse

# Third-Party
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import autocast
import yaml

# Local Modules
from src.utils.echo import echo_line
from src.utils.determinism import set_seed, worker_init_fn, make_generator
from src.data.voc import paired_image_xml_list
from src.data.cls import SquaresCLSData
from src.transforms.cls import CLSTransform
from src.metrics.cls import compute_regression_metrics_from_cfg


# Load YAML Configuration File
def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Prepare Output Directory And Save Config
def _prepare_out_dir(cfg: dict) -> Path:
    out_dir = Path(cfg.get("out_dir", "outputs/cls/test"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_dir


# Build Model According To Config
def _build_model(cfg: dict, device: torch.device):
    from src.models import cls_resnet
    mcfg = cfg.get("model") or {}
    out_dim = int(mcfg.get("out_dim", mcfg.get("num_classes", 2)))
    model_name = str(mcfg.get("model_name", mcfg.get("backbone", "resnet18")))
    pretrained = bool(mcfg.get("pretrained", True))
    final_act = mcfg.get("final_act", None)
    p_drop = float(mcfg.get("p_drop", 0.0))

    local_weights = (cfg.get("model") or {}).get("local_backbone_weights")
    if local_weights is None:
        raise ValueError("Offline testing requires 'local_backbone_weights' in model config.")
    model = cls_resnet.build_resnet_classifier(
        model_name=model_name,
        out_dim=out_dim,
        final_act=final_act,
        p_drop=p_drop,
        local_backbone_weights=local_weights
    )
    model.to(device)
    return model


# Generate Bucket Names From Configuration
def _bucket_names_from_cfg(cfg: dict):
    edges = (cfg.get("metrics") or {}).get("bucket_edges") or [0, 32, 64, 96, 128, 160, 99999]
    return [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges)-1)] + [f"{int(edges[-1])}+"]


# Resolve Weights File From Config
def _resolve_weights(cfg: dict) -> Path:
    base = Path(cfg.get("out_dir", "outputs/cls/test"))

    # Use config-specified weights if present
    cfg_weights = (cfg.get("test") or {}).get("weights")
    if cfg_weights:
        p = Path(cfg_weights)
        if not p.is_absolute():
            p = base / cfg_weights
        if p.exists():
            return p

    # Default priority: global best -> last model -> latest best_i
    global_best = base / "best_global.pt"
    if global_best.exists():
        return global_best

    last_model = base / "last_model.pt"
    if last_model.exists():
        return last_model

    best_files = sorted(base.glob("best_*.pt"))
    if best_files:
        return best_files[-1]

    raise FileNotFoundError(f"No weights found at {cfg_weights or base}")


# Automatic Mixed Precision Context
def _amp_ctx(enabled: bool, device: torch.device):
    return autocast(device_type="cuda", dtype=torch.float16) if enabled and device.type == "cuda" else nullcontext()


# Main Testing Function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", required=True, help="YAML Config File Path")
    args = parser.parse_args()

    # Load Configuration And Set Seed
    cfg = _load_cfg(args.cfg)
    seed = int(cfg.get("seed", 1337))
    set_seed(seed)
    gen = make_generator(seed)

    # Device Selection And Determinism
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    if cfg.get("deterministic", False) and device.type == "cuda":
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    # Prepare Output Directory
    out_dir = _prepare_out_dir(cfg)

    # Build Dataset And DataLoader
    root = Path((cfg.get("data") or {}).get("root", "."))
    pairs = paired_image_xml_list(root / "test", root / "annotations",
                                  limit=(cfg.get("data") or {}).get("limit_per_split"))
    img_size = int(cfg.get("eval", {}).get("img_size", cfg.get("train", {}).get("img_size", 224)))
    use_pad = bool(cfg.get("eval", {}).get("use_padding_canvas", cfg.get("train", {}).get("use_padding_canvas", True)))

    # Create Dataset
    ds = SquaresCLSData(pairs, canvas=img_size, train=False, use_padding_canvas=use_pad,
                        transforms=CLSTransform(size=img_size, train=False))

    # Build DataLoader
    dl = DataLoader(
        ds,
        batch_size=int(cfg.get("eval", {}).get("batch_size", 64)),
        shuffle=False,
        num_workers=int(cfg.get("eval", {}).get("num_workers", 4)),
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=gen
    )

    # Build Model And Load Weights
    model = _build_model(cfg, device)
    weights = _resolve_weights(cfg)
    echo_line("CLS_TEST_LOAD", {"weights": str(weights)})
    state = torch.load(weights, map_location=device)
    sd = state.get("model", state) if isinstance(state, dict) else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        echo_line("CLS_TEST_SD_MISSING", {"n": len(missing), "keys": missing[:8] + (["..."] if len(missing) > 8 else [])})
    if unexpected:
        echo_line("CLS_TEST_SD_UNEXPECTED", {"n": len(unexpected), "keys": unexpected[:8] + (["..."] if len(unexpected) > 8 else [])})

    # Forward Pass And Collect Predictions
    model.eval()
    y_true, y_pred = [], []
    amp_enabled = ((cfg.get("train", {}) or {}).get("amp", False) and device.type == "cuda")
    with torch.no_grad(), _amp_ctx(amp_enabled, device):
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y_true.append(y.detach().cpu().numpy())
            y_pred.append(model(x).detach().cpu().numpy())

    # Compute Metrics
    y_true = np.concatenate(y_true, 0) if y_true else np.zeros((0, 2), np.float32)
    y_pred = np.concatenate(y_pred, 0) if y_pred else np.zeros((0, 2), np.float32)
    stats: Dict[str, Any] = compute_regression_metrics_from_cfg(cfg, y_true, y_pred)

    # Log Metrics
    echo_line(
        "CLS_TEST",
        {"split": "test",
         "mae": float(stats.get("overall_mae", float("nan"))),
         "rmse": float(stats.get("overall_rmse", float("nan")))},
        order=["split", "mae", "rmse"]
    )

    # Save Metrics JSON
    (out_dir / "test_metrics.json").write_text(json.dumps(stats, indent=2))

    # Save Predictions CSV
    with (out_dir / "test_preds.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["y_short", "y_long", "p_short", "p_long"])
        for (yt0, yt1), (yp0, yp1) in zip(y_true, y_pred):
            w.writerow([yt0, yt1, yp0, yp1])

    # Save One-Row Metrics CSV With Bucketed Values
    bnames = _bucket_names_from_cfg(cfg)
    header = ["overall_mae", "overall_rmse", "bucket_acc_global"] + \
            [f"bucket_mae_{bn}" for bn in bnames] + [f"bucket_rmse_{bn}" for bn in bnames]
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        row = [f"{stats.get(k, float('nan')):.6f}" for k in header]
        w.writerow(row)


# Entry Point
if __name__ == "__main__":
    main()
