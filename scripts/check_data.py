# scripts/check_data.py
import argparse
import random
import sys
import torch
from torch.utils.data import DataLoader

from src.data.det import SquaresDetectionDataset, collate_fn
from src.data.cls import SquaresClassificationDatasetStreamStream
from src.data.voc import paired_image_xml_list
from src.transforms import det as Tdet


def check_det(images_dir: str, ann_dir: str, num: int, batch_size: int):
    pairs = paired_image_xml_list(images_dir, ann_dir)
    if not pairs:
        print("[DET] No (image, xml) pairs found. Check your paths & filenames.")
        return  # don't exit; allow cls check to run

    ds = SquaresDetectionDataset(pairs, transforms=Tdet.Compose([Tdet.ToTensor()]))

    print(f"[DET] samples: {len(ds)}")
    k = min(num, len(ds))
    for i in random.sample(range(len(ds)), k):
        img, tgt = ds[i]
        n_boxes = int(tgt["boxes"].shape[0]) if "boxes" in tgt else 0
        print(f"  sample {i}: img={tuple(img.shape)} boxes={n_boxes}")

    # One-batch DataLoader smoke test (catches collate/stacking issues)
    try:
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        imgs, tgts = next(iter(dl))
        print(f"[DET] batch: n={len(imgs)} first_img={tuple(imgs[0].shape)}")
    except Exception as e:
        print("[DET] DataLoader error (collate/stacking). Traceback follows:")
        raise


def check_cls(images_dir: str, ann_dir: str, num: int, batch_size: int, canvas: int = 224):
    pairs = paired_image_xml_list(images_dir, ann_dir)
    if not pairs:
        print("[CLS] No (image, xml) pairs found. Check your paths & filenames.")
        return  # don't exit; allow det check to report too

    ds = SquaresClassificationDatasetStreamStream(
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
                # show two regression values with 1 decimal (short, long)
                vals = [float(x) for x in label.flatten().tolist()]
                lbl_str = f"reg{tuple(round(v, 1) for v in vals)}"
        else:
            lbl_str = str(label)
        print(f"  sample {i}: img={tuple(img.shape)} label={lbl_str}")

    # One-batch DataLoader smoke test
    try:
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        x, y = next(iter(dl))
        print(f"[CLS] batch: x={tuple(x.shape)} y={tuple(y.shape)}")
    except Exception as e:
        print("[CLS] DataLoader error (stacking). Traceback follows:")
        raise


def main():
    ap = argparse.ArgumentParser(description="Minimal dataset sanity check")
    ap.add_argument("--task", choices=["det", "cls"], default=None, help="Which pipeline to check. If not set, runs both (det then cls).")
    ap.add_argument("--images", default="data/sized_squares_unfilled/train", help="Image dir to inspect")
    ap.add_argument("--ann", default="data/sized_squares_unfilled/annotations", help="VOC XML dir to inspect")
    ap.add_argument("--num", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--canvas", type=int, default=224, help="Classification canvas (only for --task cls)")
    args = ap.parse_args()

    # Run both if no task is specified
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
