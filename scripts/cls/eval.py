# scripts/cls/eval.py
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from src.utils.echo import echo_line
from src.utils.determinism import set_seed

from src.data.voc import paired_image_xml_list
from src.data.cls import SquaresClassificationDatasetStream
from src.transforms.cls import CLSTransform
from src.models.cls_resnet import build_resnet_backbone as build_resnet_classifier
from src.metrics.cls import evaluate_size_regression


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--split", type=str, default="val")
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg.get("seed", 42))

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    root = Path(cfg["data"]["root"])
    img_dir = root / args.split
    ann_dir = root / "annotations"

    pairs = paired_image_xml_list(img_dir, ann_dir, limit=cfg.get("data", {}).get("limit_per_split"))

    ds = SquaresClassificationDatasetStream(
        pairs=pairs,
        transforms=CLSTransform(size=int(cfg.get("model", {}).get("input_size", 224)), train=False),
    )
    dl = DataLoader(
        ds,
        batch_size=cfg.get("eval", {}).get("batch_size", 128),
        shuffle=False,
        num_workers=cfg.get("eval", {}).get("num_workers", 2),
        pin_memory=True,
    )

    # Build model (filter args)
    mcfg = cfg.get("model", {})
    model_args = {k: mcfg[k] for k in ("model_name", "out_dim", "pretrained", "final_act", "p_drop") if k in mcfg}
    model = build_resnet_classifier(**model_args).to(device)

    default_ckpt_dir = Path("outputs") / "cls" / Path(args.config).stem
    ckpt_dir = Path(cfg.get("out_dir", default_ckpt_dir))
    ckpt_path = args.ckpt or ckpt_dir / "cls_resnet.best.ckpt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    echo_line(
        "SETUP",
        {
            "device": str(device),
            "cuda": torch.cuda.is_available(),
            "split": args.split,
            "batch": cfg.get("eval", {}).get("batch_size", 128),
            "img_dir": str(img_dir),
            "ann_dir": str(ann_dir),
            "ckpt": str(ckpt_path),
        },
        order=["device", "cuda", "split", "batch", "img_dir", "ann_dir", "ckpt"],
    )

    res = evaluate_size_regression(model, dl, device=device, out_dir=str(ckpt_dir), tag=args.split)
    echo_line(
        tag="VAL",
        kv_pairs=dict(
            MAE_short=res.get("MAE_short", 0.0),
            MAE_long=res.get("MAE_long", 0.0),
            MSE_short=res.get("MSE_short", 0.0),
            MSE_long=res.get("MSE_long", 0.0),
            MAPE_short=res.get("MAPE_short", 0.0),
            MAPE_long=res.get("MAPE_long", 0.0),
            N=res.get("N", 0),
        ),
        order=["MAE_short", "MAE_long", "MSE_short", "MSE_long", "MAPE_short", "MAPE_long", "N"],
    )


if __name__ == "__main__":
    main()
