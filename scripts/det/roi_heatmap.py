import os, json, logging, datetime, argparse, torch, numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from src.data.voc import paired_image_xml_list
from src.data.split import subsample_pairs
from src.constants.norms import IMAGENET_MEAN, IMAGENET_STD

from src.models.det_fasterrcnn import build_fasterrcnn
from src.xai.pipelines.det_roi_cam import roi_gradcam_on_box
from src.xai.core.overlays import overlay_cam, draw_box, denorm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headers-dir", required=True, help="Directory with *_xai_header.json from sandbox")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="outputs/xai_det")
    args = ap.parse_args()

    device = torch.device("cpu")
    model = build_fasterrcnn(num_classes=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)); model.eval()

    headers = sorted(Path(args.headers-dir if False else args.headers_dir).glob("*_xai_header.json"))  # safe accessor
    for hpath in headers:
        h = json.load(open(hpath, "r"))
        roi = h.get("xai_targets",{}).get("roi")
        if not roi or "box" not in roi: 
            print("skip (no roi):", hpath.name); 
            continue
        img_path = h["image"]
        stem = Path(img_path).stem
        out_png = Path(args.out)/Path(hpath).parent.name/f"{stem}_roi_heatmap.png"
        out_png.parent.mkdir(parents=True, exist_ok=True)

        from torchvision.transforms.functional import to_tensor, normalize
        img_pil = Image.open(img_path).convert("RGB")
        img_t = to_tensor(img_pil); img_t = normalize(img_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        res = roi_gradcam_on_box(model, img_t, roi["box"])
        cam = res["pooled_cam"]
        import torch.nn.functional as F
        x1,y1,x2,y2 = roi["box"]
        box_w, box_h = max(1,int(round(x2-x1))), max(1,int(round(y2-y1)))
        cam_up = F.interpolate(cam.unsqueeze(0).unsqueeze(0), size=(box_h, box_w), mode="bilinear", align_corners=False)[0,0]

        import torch
        H,W = img_pil.size[1], img_pil.size[0]
        hm = torch.zeros((H,W), dtype=torch.float32)
        x1i,y1i = int(round(x1)), int(round(y1))
        x2i,y2i = min(W, x1i+box_w), min(H, y1i+box_h)
        hm[y1i:y2i, x1i:x2i] = cam_up[:(y2i-y1i), :(x2i-x1i)]
        hm = hm - hm.min(); hm = hm / (hm.max()+1e-6)

        over = overlay_cam(denorm(img_t), hm, alpha=0.45)
        plt.figure(figsize=(4.6,4.6)); plt.imshow(over); plt.axis("off"); draw_box(plt.gca(), roi["box"], color="white")
        plt.title("ROI Grad-CAM (cls=1)", fontsize=9)
        plt.tight_layout(); plt.savefig(out_png, dpi=160, bbox_inches="tight"); plt.close()
        print("saved:", out_png)

if __name__ == "__main__": main()
