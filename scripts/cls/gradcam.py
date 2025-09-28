import os, json, logging, datetime, argparse, torch, numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from src.data.voc import paired_image_xml_list
from src.data.split import subsample_pairs
from src.constants.norms import IMAGENET_MEAN, IMAGENET_STD

from src.models.cls_resnet import build_resnet_classifier
from src.xai.pipelines.cls_gradcam import run_cls_gradcam_pipeline
from src.data.cls import SquaresClassificationDatasetStreamStream

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="outputs/xai_cls")
    ap.add_argument("--fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--canvas", type=int, default=224)
    args = ap.parse_args()

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out)/run_id; out_dir.mkdir(parents=True, exist_ok=True)

    pairs_all = paired_image_xml_list(args.images, args.ann)
    pairs = subsample_pairs(pairs_all, fraction=args.fraction, seed=args.seed)
    ds = SquaresClassificationDatasetStreamStream(pairs, canvas=args.canvas, train=False, use_padding_canvas=True)

    device = torch.device("cpu")
    model = build_resnet_classifier(num_outputs=5).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)); model.eval()

    for i in range(min(20, len(ds))):
        x, y = ds[i]
        x1 = x.unsqueeze(0).to(device)
        out = run_cls_gradcam_pipeline(model, x1, target_class=None)
        cam = out["cam"]
        over = (cam.detach().cpu().numpy())
        from src.xai.core.overlays import overlay_cam, denorm
        vis = overlay_cam(x, cam, alpha=0.45)
        stem = f"gradcam_{i+1:03d}.png"
        plt.imsave(out_dir/stem, vis)
        print("saved:", out_dir/stem)

if __name__ == "__main__": main()
