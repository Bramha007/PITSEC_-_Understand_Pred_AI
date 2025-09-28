import os, json, logging, datetime, argparse, torch, numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from src.data.voc import paired_image_xml_list
from src.data.split import subsample_pairs
from src.constants.norms import IMAGENET_MEAN, IMAGENET_STD

from src.models.det_fasterrcnn import build_fasterrcnn
from src.xai.core.captum_wrappers import captum_ig
from src.xai.core.roi_api import map_box_to_transformed
from src.xai.core.overlays import overlay_cam, draw_box, denorm

def make_forward_fn(model, W, H, box_xyxy):
    device = next(model.parameters()).device
    x1,y1,x2,y2 = box_xyxy
    def forward_fn(x_in):
        B,C,H_in,W_in = x_in.shape
        imgs_list = [x_in[b].to(device) for b in range(B)]
        images_list, _ = model.transform(imgs_list, None)
        x = images_list.tensors
        image_sizes = images_list.image_sizes
        feats = model.backbone(x)
        boxes_tr = []
        for b in range(B):
            Ht,Wt = image_sizes[b]
            boxes_tr.append(torch.tensor([map_box_to_transformed(box_xyxy, (W,H), (Wt,Ht))], dtype=torch.float32, device=device))
        pooled = model.roi_heads.box_roi_pool(feats, boxes_tr, image_sizes)
        rep = model.roi_heads.box_head(pooled)
        class_logits, _ = model.roi_heads.box_predictor(rep)
        return class_logits[:, 1]
    return forward_fn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--headers-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="outputs/xai_det")
    ap.add_argument("--steps", type=int, default=64)
    args = ap.parse_args()

    device = torch.device("cpu")
    model = build_fasterrcnn(num_classes=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)); model.eval()

    headers = sorted(Path(args.headers_dir).glob("*_xai_header.json"))
    for hpath in headers:
        h = json.load(open(hpath, "r"))
        roi = h.get("xai_targets",{}).get("roi")
        if not roi or "box" not in roi: 
            print("skip (no roi):", hpath.name); continue
        img_path = h["image"]; stem = Path(img_path).stem
        run_id = Path(args.headers_dir).name
        out_png = Path(args.out)/run_id/f"{stem}_captum_roi_ig.png"
        out_png.parent.mkdir(parents=True, exist_ok=True)

        from torchvision.transforms.functional import to_tensor, normalize
        img_pil = Image.open(img_path).convert("RGB")
        img_t = to_tensor(img_pil); img_t = normalize(img_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        x = img_t.unsqueeze(0).to(device); x.requires_grad_(True)

        W,H = img_pil.size
        fwd = make_forward_fn(model, W, H, roi["box"])
        att = captum_ig(fwd, x, target=None, steps=args.steps)
        att = att.abs().squeeze(0).sum(0)
        att = (att - att.min())/(att.max()-att.min()+1e-6)

        vis = overlay_cam(img_t, att, alpha=0.5)
        plt.figure(figsize=(4.6,4.6)); plt.imshow(vis); plt.axis("off"); draw_box(plt.gca(), roi["box"], color="white")
        plt.title("Captum IG toward ROI cls=1", fontsize=9)
        plt.tight_layout(); plt.savefig(out_png, dpi=160, bbox_inches="tight"); plt.close()
        print("saved:", out_png)

if __name__ == "__main__": main()
