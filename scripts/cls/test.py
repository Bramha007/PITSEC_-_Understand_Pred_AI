# scripts/cls/test.py
# Run Classification/Regression Model On Squares Dataset (Test Split).
# Deterministic Eval With Bucketed Metrics And Predictions Saved.

from __future__ import annotations
import argparse, csv, json, os
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np, torch, yaml
from torch.utils.data import DataLoader

from src.utils.echo import echo_line
from src.utils.determinism import set_seed, worker_init_fn, make_generator
from src.data.voc import paired_image_xml_list
from src.data.cls import SquaresCLSData
from src.transforms.cls import CLSTransform
from src.metrics.cls import compute_regression_metrics_from_cfg


# Load YAML Config
def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Build ResNet-Based Regressor (2 Outputs By Default)
def _build_model(cfg: dict, device: torch.device):
    from src.models import cls_resnet
    m = cfg.get("model") or {}
    model = cls_resnet.build_resnet_classifier(
        model_name=str(m.get("model_name", m.get("backbone", "resnet18"))),
        out_dim=int(m.get("out_dim", m.get("num_classes", 2))),
        pretrained=bool(m.get("pretrained", True)),
        final_act=m.get("final_act", None),
        p_drop=float(m.get("p_drop", 0.0)),
    )
    model.to(device)
    return model


# Bucket Labels For Size-Based Metrics
def _bucket_names_from_cfg(cfg: dict):
    edges = (cfg.get("metrics") or {}).get("bucket_edges") or [0, 32, 64, 96, 128, 160, 99999]
    return [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges)-1)] + [f"{int(edges[-1])}+"]


# Pick Weights: CLI > out_dir/cls.best.ckpt > out_dir/cls.last.ckpt
def _resolve_weights(cfg: dict, cli_weights: Optional[str]) -> Path:
    if cli_weights:
        p = Path(cli_weights)
        if p.exists():
            return p
    base = Path(cfg.get("out_dir", "outputs/cls/test"))
    for name in ("cls.best.ckpt", "cls.last.ckpt"):
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No weights found at {cli_weights or base}")


def main():
    # Args
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    # Config + Determinism
    cfg = _load_cfg(args.config)
    seed = int(cfg.get("seed", 1337))
    os.environ["DATA_WORKER_SEED"] = str(seed)
    set_seed(seed)
    gen = make_generator(seed)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Output Dir
    out_dir = Path(args.out_dir or cfg.get("out_dir", "outputs/cls/test"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Dataset (Fixed To Test Split)
    root = Path((cfg.get("data") or {}).get("root", "."))
    pairs = paired_image_xml_list(root / "test", root / "annotations",
                                  limit=(cfg.get("data") or {}).get("limit_per_split"))
    canvas = int(cfg.get("eval", {}).get("img_size", cfg.get("train", {}).get("img_size", 224)))
    pad = bool(cfg.get("eval", {}).get("use_padding_canvas", cfg.get("train", {}).get("use_padding_canvas", True)))
    ds = SquaresCLSData(pairs, canvas=canvas, train=False, use_padding_canvas=pad,
                        transforms=CLSTransform(size=canvas, train=False))
    dl = DataLoader(ds,
                    batch_size=int(cfg.get("eval", {}).get("batch_size", 64)),
                    shuffle=False,
                    num_workers=int(cfg.get("eval", {}).get("num_workers", 4)),
                    pin_memory=True,
                    worker_init_fn=worker_init_fn,
                    generator=gen)

    # Model + Load Weights
    model = _build_model(cfg, device)
    weights = _resolve_weights(cfg, args.weights)
    echo_line("CLS_TEST_LOAD", {"weights": str(weights)})
    state = torch.load(weights, map_location="cpu")
    sd = state["model"] if isinstance(state, dict) and isinstance(state.get("model"), dict) else state
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        echo_line("CLS_TEST_SD_MISSING", {"n": len(missing), "keys": missing[:8] + (["..."] if len(missing) > 8 else [])})
    if unexpected:
        echo_line("CLS_TEST_SD_UNEXPECTED", {"n": len(unexpected), "keys": unexpected[:8] + (["..."] if len(unexpected) > 8 else [])})

    # Forward Pass
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            y_true.append(y.detach().cpu().numpy())
            y_pred.append(model(x).detach().cpu().numpy())
    y_true = np.concatenate(y_true, 0) if y_true else np.zeros((0, 2), np.float32)
    y_pred = np.concatenate(y_pred, 0) if y_pred else np.zeros((0, 2), np.float32)

    # Metrics
    stats: Dict[str, Any] = compute_regression_metrics_from_cfg(cfg, y_true, y_pred)
    echo_line("CLS_TEST", {"split": "test",
                           "mae": float(stats.get("overall_mae", float("nan"))),
                           "rmse": float(stats.get("overall_rmse", float("nan")))},
              order=["split", "mae", "rmse"])

    # Save JSON + Preds CSV
    (out_dir / "test_metrics.json").write_text(json.dumps(stats, indent=2))
    with (out_dir / "test_preds.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["y_short", "y_long", "p_short", "p_long"])
        for (yt0, yt1), (yp0, yp1) in zip(y_true, y_pred):
            w.writerow([yt0, yt1, yp0, yp1])

    # Save One-Row metrics.csv For Quick Diff
    bnames = _bucket_names_from_cfg(cfg)
    header = ["overall_mae", "overall_rmse", "bucket_acc_global"] + \
             [f"bucket_mae_{bn}" for bn in bnames] + [f"bucket_rmse_{bn}" for bn in bnames]
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        row = [f"{stats.get(k, float('nan')):.6f}" for k in header]
        w.writerow(row)


if __name__ == "__main__":
    main()
