# scripts/cls/explain.py

# Generates XAI Visualizations For Classification Models Using Configurable YAML Settings
# Supports Grad-CAM, Integrated Gradients, Occlusion, Deterministic Evaluation, And AMP
# Saves Heatmaps And Overlayed Images To Disk

from __future__ import annotations

# Standard Library
import os
import argparse
from pathlib import Path
from contextlib import nullcontext
from typing import Optional, List

# Third-Party
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast
import yaml

# Local Modules
from src.utils.echo import echo_line
from src.utils.determinism import set_seed, worker_init_fn, make_generator
from src.data.voc import paired_image_xml_list
from src.data.cls import SquaresCLSData
from src.transforms.cls import CLSTransform
from src.models import cls_resnet
from src.xai import gradcam, integrated_gradients, occlusion
from src.utils.vis import save_heatmap_overlay


# Load YAML Configuration File
def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# Prepare Output Directory And Save Config
def _prepare_out_dir(cfg: dict) -> Path:
    out_dir = Path((cfg.get("explain") or {}).get("out_dir", "outputs/cls/xai"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_dir


# Build Model According To Config
def _build_model(cfg: dict, device: torch.device):
    mcfg = cfg.get("model") or {}
    out_dim = int(mcfg.get("out_dim", 2))
    model_name = str(mcfg.get("model_name", "resnet18"))
    final_act = mcfg.get("final_act", None)
    p_drop = float(mcfg.get("p_drop", 0.0))
    local_weights = str(mcfg.get("local_backbone_weights", ""))

    model = cls_resnet.build_resnet_classifier(
        model_name=model_name,
        out_dim=out_dim,
        final_act=final_act,
        p_drop=p_drop,
        local_backbone_weights=local_weights
    )
    model.to(device)
    return model


# Resolve Weights File From Config Or Default
def _resolve_weights(cfg: dict) -> Path:
    base = Path(cfg.get("out_dir", "outputs/cls/test"))

    # Use Config-Specified Weights If Present
    cfg_weights = (cfg.get("explain") or {}).get("weights")
    if cfg_weights:
        p = Path(cfg_weights)
        if not p.is_absolute():
            p = base / cfg_weights
        if p.exists():
            return p

    # Default Priority: global best -> last model -> latest best_i
    for name in ("best_global.pt", "last_model.pt"):
        p = base / name
        if p.exists():
            return p

    best_files = sorted(base.glob("best_*.pt"))
    if best_files:
        return best_files[-1]

    raise FileNotFoundError(f"No weights found at {cfg_weights or base}")


# Automatic Mixed Precision Context
def _amp_ctx(enabled: bool, device: torch.device):
    return autocast(device_type="cuda", dtype=torch.float16) if enabled and device.type == "cuda" else nullcontext()


# Main XAI Function
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/cls/test.yaml", help="YAML Config File Path")
    args = parser.parse_args()

    # Load Configuration And Set Seed
    cfg = _load_cfg(args.cfg)
    seed = int(cfg.get("seed", 1337))
    set_seed(seed)
    gen = make_generator(seed)

    # Device Selection
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # Prepare Output Directory
    out_dir = _prepare_out_dir(cfg)

    # Build Dataset And DataLoader
    data_root = Path((cfg.get("data") or {}).get("root", "."))
    pairs = paired_image_xml_list(data_root / "test", data_root / "annotations",
                                  limit=(cfg.get("data") or {}).get("limit_per_split"))
    img_size = int(cfg.get("eval", {}).get("img_size", cfg.get("train", {}).get("img_size", 224)))

    ds = SquaresCLSData(pairs, canvas=img_size, train=False,
                        transforms=CLSTransform(size=img_size, train=False))
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn, generator=gen)

    # Build Model And Load Weights
    model = _build_model(cfg, device)
    weights = _resolve_weights(cfg)
    echo_line("CLS_XAI_LOAD", {"weights": str(weights)})
    state = torch.load(weights, map_location=device)
    sd = state.get("model", state) if isinstance(state, dict) else state
    model.load_state_dict(sd, strict=False)
    model.eval()

    # XAI Methods From Config
    methods: List[str] = (cfg.get("explain") or {}).get("methods", ["gradcam", "ig", "occlusion"])
    amp_enabled = ((cfg.get("train", {}) or {}).get("amp", False) and device.type == "cuda")

    # Forward Pass And Save Heatmaps
    for idx, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        if method == "gradcam":
            heatmap = gradcam.apply_gradcam(model, x)
        elif method == "ig":
            heatmap = integrated_gradients.apply_ig(model, x)
        elif method == "occlusion":
            heatmap = occlusion.apply_occlusion(model, x)
        else:
            raise ValueError(f"Unknown method: {method}")

        save_heatmap_overlay(x[0], heatmap[0], out_dir / f"{method.lower()}_{idx}.png")

# Entry Point
if __name__ == "__main__":
    main()
