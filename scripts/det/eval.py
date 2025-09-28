# scripts/det/eval.py
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from src.utils.echo import echo_line
from src.utils.determinism import set_seed
from src.data.voc import paired_image_xml_list
from src.data.det import SquaresDetectionDataset, collate_fn
from src.transforms.det import Compose, ToTensor, ClampBoxes
from src.metrics.det import evaluate_ap_by_size as eval_detector
from src.metrics.ar import evaluate_ap_by_ar
from src.models.det_fasterrcnn import build_fasterrcnn, build_fasterrcnn_custom


def implicit_det_paths(root: Path, split: str):
    root = Path(root)
    img_dir = root / split
    ann_dir = root / "annotations"
    return img_dir, ann_dir


def _to_int(x, default):
    try:
        return int(x)
    except Exception:
        return int(default)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--split", type=str, default="val")
    # ---- AR toggle ----
    g = p.add_mutually_exclusive_group()
    g.add_argument("--ar", dest="do_ar", action="store_true", help="Run AR-bucket eval (default)")
    g.add_argument("--no-ar", dest="do_ar", action="store_false", help="Skip AR-bucket eval for speed")
    p.set_defaults(do_ar=True)

    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg.get("seed", 42))
    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    root = Path(cfg["data"]["root"])
    img_dir, ann_dir = implicit_det_paths(root, args.split)

    pairs = paired_image_xml_list(img_dir, ann_dir, limit=cfg.get("data", {}).get("limit_per_split"))
    ds = SquaresDetectionDataset(pairs, transforms=Compose([ToTensor(), ClampBoxes()]))
    dl = DataLoader(
        ds,
        batch_size=_to_int(cfg.get("eval", {}).get("batch_size", 1), 1),
        shuffle=False,
        num_workers=_to_int(cfg.get("eval", {}).get("num_workers", 2), 2),
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Model
    if cfg["model"].get("custom", False):
        model = build_fasterrcnn_custom(cfg["model"])
    else:
        model = build_fasterrcnn(num_classes=cfg["model"]["num_classes"])
    model.to(device)

    # Checkpoint
    default_ckpt_dir = Path(cfg.get("out_dir", Path("outputs") / "det" / Path(args.config).stem))
    ckpt_path = Path(args.ckpt) if args.ckpt else (default_ckpt_dir / "det_fasterrcnn.best.ckpt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Info banner
    echo_line("SETUP", {
        "device": str(device),
        "cuda": torch.cuda.is_available(),
        "split": args.split,
        "batch": cfg.get("eval", {}).get("batch_size", 1),
        "img_dir": str(img_dir),
        "ann_dir": str(ann_dir),
        "ckpt": str(ckpt_path),
        "ar_eval": args.do_ar,
    }, order=["device","cuda","split","batch","img_dir","ann_dir","ckpt","ar_eval"])

    # Size-bucket AP evaluation
    metrics, size_json, _ = eval_detector(model, dl, device=device, out_dir=str(default_ckpt_dir), tag=args.split)
    ap = float(metrics.get("ap50_global", 0.0))
    npos = int(metrics.get("npos_total", 0))
    echo_line("VAL", {"ap50": ap, "npos": npos}, order=["ap50","npos"])

    sizes = metrics.get("ap50_by_size", {})
    if sizes:
        keys = sorted(sizes.keys(), key=lambda k: (len(k), k))
        kv = {k: float(sizes[k]) for k in keys}
        echo_line("VAL ap50_by_size", kv, order=keys)

    # Aspect-ratio bucket AP evaluation (optional)
    if args.do_ar:
        # Saves: ar_metrics_{split}.json and ar_metrics_{split}.png
        ar_res = evaluate_ap_by_ar(model, dl, device=device, out_dir=str(default_ckpt_dir), tag=args.split)
        ar_ap = ar_res.get("ap50_by_ar", {})
        if ar_ap:
            order = ["near_square", "moderate", "skinny"]
            kv = {k: float(ar_ap.get(k, 0.0)) for k in order}
            echo_line("VAL ap50_by_AR", kv, order=order)

    print("Saved size metrics JSON:", size_json)
    if args.do_ar:
        print("Saved AR metrics JSON/PNG under:", str(default_ckpt_dir))


if __name__ == "__main__":
    main()
