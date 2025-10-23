# scripts/check_data.py

# Quick Sanity Check For Detection And Classification Datasets
# Verifies Sample Shapes, Labels, And One-Batch DataLoader Behavior

import argparse
import random
import sys
import torch
from torch.utils.data import DataLoader

from src.data.det import SquaresDetectionDataset, collate_fn
from src.data.cls import SquaresCLSData
from src.data.voc import paired_image_xml_list
from src.transforms import det as Tdet


# Check Detection Samples And One-Batch Dataloader
def check_det(images_dir: str, ann_dir: str, num: int, batch_size: int):
    pairs = paired_image_xml_list(images_dir, ann_dir)
    if not pairs:
        print("[DET] No (image, xml) pairs found. Check your paths & filenames.")
        return  # Don't Exit; Allow CLS Check To Run

    ds = SquaresDetectionDataset(pairs, transforms=Tdet.Compose([Tdet.ToTensor()]))

    print(f"[DET] samples: {len(ds)}")
    k = min(num, len(ds))
    for i in random.sample(range(len(ds)), k):
        img, tgt = ds[i]
        n_boxes = int(tgt["boxes"].shape[0]) if "boxes" in tgt else 0
        print(f"  sample {i}: img={tuple(img.shape)} boxes={n_boxes}")

    # One-Batch Dataloader Smoke Test (Catches Collate/Stacking Issues)
    try:
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        imgs, tgts = next(iter(dl))
        print(f"[DET] batch: n={len(imgs)} first_img={tuple(imgs[0].shape)}")
    except Exception as e:
        print("[DET] DataLoader error (collate/stacking). Traceback follows:")
        raise


# Check Classification Samples And One-Batch Dataloader
def check_cls(images_dir: str, ann_dir: str, num: int, batch_size: int, canvas: int = 224):
    pairs = paired_image_xml_list(images_dir, ann_dir)
    if not pairs:
        print("[CLS] No (image, xml) pairs found. Check your paths & filenames.")
        return  # Don't Exit; Allow DET Check To Report Too

    ds = SquaresCLSData(
        pairs, canvas=canvas, train=False, use_padding_canvas=True
    )

    print(f"[CLS] samples: {len(ds)}")
    k = min(num, len(ds))
    for i in random.sample(range(len(ds)), k):
        img, label = ds[i]
        if isinstance(label, torch.Tensor):
            if label.numel() == 1:
                lbl_str = str(int(label.item()))
            else:
                # Show Two Regression Values With 1 Decimal (Short, Long)
                vals = [float(x) for x in label.flatten().tolist()]
                lbl_str = f"reg{tuple(round(v, 1) for v in vals)}"
        else:
            lbl_str = str(label)
        print(f"  sample {i}: img={tuple(img.shape)} label={lbl_str}")

    # One-Batch Dataloader Smoke Test
    try:
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        x, y = next(iter(dl))
        print(f"[CLS] batch: x={tuple(x.shape)} y={tuple(y.shape)}")
    except Exception as e:
        print("[CLS] DataLoader error (stacking). Traceback follows:")
        raise


# Main
def main():
    ap = argparse.ArgumentParser(description="Minimal Dataset Sanity Check")
    ap.add_argument("--task", choices=["det", "cls"], default=None, help="Which Pipeline To Check. If Not Set, Runs Both (det Then cls).")
    ap.add_argument("--images", default="data/sized_squares_unfilled/train", help="Image Dir To Inspect")
    ap.add_argument("--ann", default="data/sized_squares_unfilled/annotations", help="VOC XML Dir To Inspect")
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--canvas", type=int, default=224, help="Classification Canvas (Only For --task cls)")
    args = ap.parse_args()

    # Run Both If No Task Is Specified
    ran_any = False
    if args.task is None or args.task == "det":
        check_det(args.images, args.ann, num=args.num, batch_size=args.batch_size)
        ran_any = True

    if args.task is None or args.task == "cls":
        check_cls(args.images, args.ann, num=args.num, batch_size=args.batch_size, canvas=args.canvas)
        ran_any = True

    if not ran_any:
        print("[ERR] Nothing to run. Use --task det or --task cls.")
        sys.exit(1)


if __name__ == "__main__":
    main()
