import os, json, logging, datetime, argparse, torch, numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from src.data.voc import paired_image_xml_list
from src.data.split import subsample_pairs
from src.constants.norms import IMAGENET_MEAN, IMAGENET_STD

from src.models.cls_resnet import build_resnet_classifier
from src.xai.pipelines.cls_captum import run_cls_captum_pipeline
from src.data.cls import SquaresClassificationDatasetStreamStream
from src.xai.core.overlays import overlay_cam

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="outputs/xai_cls_captum")
    ap.add_argument("--fraction", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=64)
    args = ap.parse_args()

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out)/run_id; out_dir.mkdir(parents=True, exist_ok=True)

    pairs_all = paired_image_xml_list(args.images, args.ann)
    pairs = subsample_pairs(pairs_all, fraction=args.fraction, seed=args.seed)
    ds = SquaresClassificationDatasetStreamStream(pairs, canvas=224, train=False, use_padding_canvas=True)

    device = torch.device("cpu")
    model = build_resnet_classifier(num_outputs=5).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)); model.eval()

    import numpy as np
    for i in range(min(12, len(ds))):
        x, y = ds[i]
        with torch.no_grad():
            logits = model(x.unsqueeze(0).to(device))
            pred = int(logits.argmax(1).item())
        res = run_cls_captum_pipeline(model, x.unsqueeze(0).to(device), target=pred, steps=args.steps)
        for k, att in res.items():
            vis = overlay_cam(x, att, alpha=0.5)
            stem = f"{k}_{i+1:03d}.png"
            plt.imsave(out_dir/stem, vis)
            print("saved:", out_dir/stem)

if __name__ == "__main__": main()
