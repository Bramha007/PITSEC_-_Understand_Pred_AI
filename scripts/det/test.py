# scripts/det/test.py
# Run Faster R-CNN On Squares Dataset (Test Split).
# Deterministic Eval With AP/AR Metrics And CSV/JSON Exports.

from __future__ import annotations
import argparse, csv, json, os
from pathlib import Path
from typing import Dict, List, Optional

import torch, yaml
from torch.utils.data import DataLoader

from src.utils.echo import echo_line
from src.utils.determinism import set_seed, worker_init_fn, make_generator
from src.data.voc import paired_image_xml_list
from src.data.det import SquaresDetectionDataset, collate_fn
from src.transforms.det import Compose, ToTensor, ClampBoxes
from src.models.det_fasterrcnn import build_fasterrcnn
from src.metrics.det import evaluate_ap_by_size


# Load YAML Config
def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Build Dataloader For Test Split
def _build_test_loader(cfg: dict, gen: torch.Generator):
    data = cfg.get("data", {}) or {}
    root = Path(data.get("root", "."))  # expects test/, annotations/
    limit = data.get("limit_per_split", None)

    img_dir = root / "test"
    ann_dir = root / "annotations"
    pairs = paired_image_xml_list(img_dir, ann_dir, limit=limit)

    tfm = Compose([ToTensor(), ClampBoxes()])
    ds = SquaresDetectionDataset(pairs, transforms=tfm)

    ev = cfg.get("eval", {}) or {}
    dl = DataLoader(
        ds,
        batch_size=int(ev.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(ev.get("num_workers", 2)),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,  # per-worker seed fixup
        generator=gen,                  # deterministic sampling order
    )
    return ds, dl


# Build Model (Mirrors Train)
def _build_model(cfg: dict, device: torch.device):
    m = cfg.get("model", {}) or {}
    model = build_fasterrcnn(
        num_classes=int(m.get("num_classes", 2)),
        anchor_sizes=m.get("anchor_sizes", None),
        anchor_aspect_ratios=m.get("anchor_aspect_ratios", None),
        weights=m.get("weights", None),
        trainable_backbone_layers=m.get("trainable_backbone_layers", None),
        local_backbone_weights=m.get("local_backbone_weights", None),
    )
    model.to(device)
    return model


# Match Train-Side Metrics Header
def _header_from_metrics(m: Dict[str, float]) -> List[str]:
    header = ["ap50_global", "ap_global", "ar_global"]
    for prefix in ["ap50_", "ap_", "ar_"]:
        keys = sorted([k for k in m.keys() if k.startswith(prefix) and not k.endswith("_global")])
        header.extend(keys)
    return header


# Resolve Weights: CLI > out_dir/det_fasterrcnn.best.ckpt > out_dir/det_fasterrcnn.last.ckpt
def _resolve_weights(cfg: dict, cli_weights: Optional[str]) -> Optional[Path]:
    if cli_weights:
        p = Path(cli_weights)
        if p.exists():
            return p
    base = Path(cfg.get("out_dir", "outputs/det/test"))
    for name in ("det_fasterrcnn.best.ckpt", "det_fasterrcnn.last.ckpt"):
        p = base / name
        if p.exists():
            return p
    return None  # allow running with randomly initialized weights if nothing found


def main():
    # Args
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--weights", type=str, default=None)   # .ckpt or raw state dict .pth/.pt
    ap.add_argument("--out_dir", type=str, default=None)   # optional override
    args = ap.parse_args()

    # Config + Determinism
    cfg = _load_cfg(args.config)
    seed = int(cfg.get("seed", 42))
    os.environ["DATA_WORKER_SEED"] = str(seed)
    set_seed(seed)
    gen = make_generator(seed)
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Output Dir
    out_dir = Path(args.out_dir or cfg.get("out_dir", "outputs/det/test"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # Data (Implicit Test Split)
    _, dl = _build_test_loader(cfg, gen)

    # Model
    model = _build_model(cfg, device)

    # Weights (Best/Last/Direct SD); optional to allow dry runs
    weights_path = _resolve_weights(cfg, args.weights)
    if weights_path is not None:
        echo_line("DET_TEST_LOAD", {"weights": str(weights_path)})
        state = torch.load(weights_path, map_location="cpu")
        sd = state["model"] if isinstance(state, dict) and isinstance(state.get("model"), dict) else state
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            echo_line("DET_TEST_SD_MISSING", {"n": len(missing), "keys": missing[:8] + (["..."] if len(missing) > 8 else [])})
        if unexpected:
            echo_line("DET_TEST_SD_UNEXPECTED", {"n": len(unexpected), "keys": unexpected[:8] + (["..."] if len(unexpected) > 8 else [])})
    else:
        echo_line("DET_TEST_LOAD", {"weights": "NONE (USING RANDOM INIT)"} )

    # Eval
    tag = "test"
    metrics, _, _ = evaluate_ap_by_size(model, dl, device, out_dir=str(out_dir), tag=tag)
    echo_line("DET_TEST",
              {"split": tag,
               "ap50": float(metrics.get("ap50_global", 0.0)),
               "ap": float(metrics.get("ap_global", 0.0)),
               "ar": float(metrics.get("ar_global", 0.0))},
              order=["split", "ap50", "ap", "ar"])

    # Save JSON + One-Row CSV
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    header = _header_from_metrics(metrics)
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow([f"{float(metrics.get(k, float('nan'))):.6f}" for k in header])


if __name__ == "__main__":
    main()
