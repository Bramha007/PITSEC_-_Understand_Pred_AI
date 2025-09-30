# scripts/cls/explain.py

import argparse, json, math
from pathlib import Path
from datetime import datetime
import torch
import torchvision.transforms.functional as TF
from torchvision.io import read_image

from src.xai.core.gradcam import gradcam_on_layer
from src.xai.core.attributions import roi_integrated_gradients
from src.xai.core.overlays import overlay_cam
from src.xai.core.roi_ops import (
    select_top_roi,
    rpn_cell_near,
    paste_heatmap_into_bbox,
    jitter_box, scale_box, occlude_box
)

# Model
def load_detector(ckpt, device):
    # TODO: adapt to your detector builder
    from src.models.detector_frcnn import build_model
    m = build_model().to(device)
    state = torch.load(ckpt, map_location=device)
    m.load_state_dict(state["model"] if "model" in state else state)
    m.eval()
    return m

def image_iter(images_dir, limit=0):
    paths = [p for p in Path(images_dir).rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png"}]
    if limit > 0: paths = paths[:limit]
    for p in paths: yield p

def make_header(model, img_path, device, score_thr=0.5):
    img = read_image(str(img_path)).float()/255.0
    H, W = img.shape[-2:]
    xb = TF.resize(img, [H, W]).unsqueeze(0).to(device)

    with torch.no_grad():
        det = model(xb)[0]  # Faster R-CNN style: dict with boxes,scores,labels

    if len(det["boxes"]) == 0: 
        return None

    roi_idx, roi_box = select_top_roi(det, score_thr=score_thr)  # (x1,y1,x2,y2), int idx
    header = {
        "image": str(img_path),
        "size": [int(H), int(W)],
        "roi_idx": int(roi_idx),
        "roi_box": [float(x) for x in roi_box],
        "label": int(det["labels"][roi_idx]),
        "score": float(det["scores"][roi_idx]),
    }
    return header, img  # also return tensor image for overlays

# RPN & ROI Grad-CAM
def run_rpn_gradcam(model, header, img, device, rpn_layer="backbone.body.layer4", save_to=None):
    # 1) pick RPN cell near ROI center
    cell = rpn_cell_near(header["roi_box"], fmap_stride=16)  # TODO: stride per backbone
    # 2) Grad-CAM targeting RPN objectness at that cell (your gradcam_on_layer must support this)
    cam = gradcam_on_layer(model, img.unsqueeze(0).to(device),
                           target=("rpn_objectness", cell),
                           layer=rpn_layer)  # (H,W) normalized
    vis = overlay_cam(img, cam, alpha=0.45)
    if save_to: TF.to_pil_image(vis).save(save_to)
    return cam

def run_roi_gradcam(model, header, img, device, roi_layer="roi_heads.box_head.fc7", save_to=None):
    cam_small = gradcam_on_layer(model, img.unsqueeze(0).to(device),
                                 target=("roi_class_logit", header["roi_idx"]),
                                 layer=roi_layer)  # returns heatmap in ROI space or resized
    cam_full = paste_heatmap_into_bbox(cam_small, header["roi_box"], img.shape[-2:])
    vis = overlay_cam(img, cam_full, alpha=0.45)
    if save_to: TF.to_pil_image(vis).save(save_to)
    return cam_full

# Captum IG on ROI
def run_roi_ig(model, header, img, device, steps, save_to=None):
    ig_small = roi_integrated_gradients(model, img.unsqueeze(0).to(device),
                                        roi_idx=header["roi_idx"],
                                        target_label=header["label"],
                                        steps=steps)  # ROI-space heatmap
    ig_full = paste_heatmap_into_bbox(ig_small, header["roi_box"], img.shape[-2:])
    vis = overlay_cam(img, ig_full, alpha=0.5)
    if save_to: TF.to_pil_image(vis).save(save_to)
    return ig_full

# Counterfactual probes
def run_counterfactuals(model, header, img, device, save_dir: Path):
    records = {}
    # Jitter
    jb = jitter_box(header["roi_box"], img.shape[-2:], px=6)
    # Scale
    sb = scale_box(header["roi_box"], img.shape[-2:], scale=1.2)
    # Occlusion
    ob = occlude_box(img, header["roi_box"], value=0.0)  # Returns occluded image tensor

    with torch.no_grad():
        # Re-evaluate logits / scores for target ROI/label under CFs (simple: max class logit in new box area)
        base = model(img.unsqueeze(0).to(device))[0]
        base_logit = float(base["scores"][header["roi_idx"]])

        # Jittered
        jimg = img.clone()
        # (optionally draw white box or crop/paste—simplified here)
        jlogit = base_logit  # TODO: recompute properly if you re-run model on modified input
        # Scaled
        slogit = base_logit  # TODO: recompute properly
        # Occluded
        ologit = float(model(ob.unsqueeze(0).to(device))[0]["scores"].max().item())

    records["jitter"] = {"box": [float(x) for x in jb], "logit": jlogit}
    records["scale"]  = {"box": [float(x) for x in sb], "logit": slogit}
    records["occlusion"] = {"logit": ologit}
    (save_dir / "counterfactuals.json").write_text(json.dumps(records, indent=2))
    return records

# Pack index
def pack_index(out_dir: Path, items):
    idx = {"n": len(items), "items": items}
    (out_dir / "INDEX.json").write_text(json.dumps(idx, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ann", required=False)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="outputs/xai/det")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--steps-ig", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = load_detector(args.ckpt, device)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for img_path in image_iter(args.images, limit=args.limit):
        result = make_header(model, img_path, device)
        if result is None:
            continue
        header, img = result
        stem = Path(header["image"]).stem

        # RPN CAM
        rpn_path = out_dir / f"{stem}_rpn_cam.png"
        run_rpn_gradcam(model, header, img, device, save_to=rpn_path)

        # ROI CAM
        roi_path = out_dir / f"{stem}_roi_cam.png"
        run_roi_gradcam(model, header, img, device, save_to=roi_path)

        # ROI IG
        ig_path = out_dir / f"{stem}_roi_ig.png"
        run_roi_ig(model, header, img, device, steps=args.steps_ig, save_to=ig_path)

        # Counterfactuals
        cf_dir = out_dir / f"{stem}_cf"; cf_dir.mkdir(exist_ok=True)
        run_counterfactuals(model, header, img, device, cf_dir)

        # Collect index entry
        items.append({
            "image": header["image"],
            "roi_box": header["roi_box"],
            "label": header["label"],
            "score": header["score"],
            "rpn_cam": str(rpn_path),
            "roi_cam": str(roi_path),
            "roi_ig":  str(ig_path),
            "cf_dir":  str(cf_dir),
        })

    pack_index(out_dir, items)
    print("Detection XAI artifacts →", out_dir)

if __name__ == "__main__":
    main()
