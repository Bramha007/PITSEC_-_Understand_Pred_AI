import os, json, glob
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from src.models.det_fasterrcnn import build_fasterrcnn

OUT_DIR   = "outputs/xai_det"
CKPT_PATH = "outputs/fasterrcnn_squares_cpu.pt"
ALPHA     = 0.45

def to_tensor_norm(pil):
    import torchvision.transforms as T
    return T.Compose([T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])(pil)

def read_header_jsons(out_dir):
    import glob, os
    return sorted(glob.glob(os.path.join(out_dir, "*_xai_header.json")))

def occlude_inside_roi(pil, box, frac=0.3, where="top"):
    x1,y1,x2,y2 = map(int, box)
    pil2 = pil.copy()
    draw = ImageDraw.Draw(pil2)
    w, h = x2-x1, y2-y1
    if where == "top":
        draw.rectangle([x1, y1, x2, y1 + int(frac*h)], fill=(255,255,255))
    elif where == "bottom":
        draw.rectangle([x1, y2 - int(frac*h), x2, y2], fill=(255,255,255))
    elif where == "left":
        draw.rectangle([x1, y1, x1 + int(frac*w), y2], fill=(255,255,255))
    else:
        draw.rectangle([x2 - int(frac*w), y1, x2, y2], fill=(255,255,255))
    return pil2

def jitter_box(box, dx=0, dy=0):
    x1,y1,x2,y2 = box
    return [x1+dx, y1+dy, x2+dx, y2+dy]

def scale_box(box, scale=1.0):
    x1,y1,x2,y2 = box
    cx, cy = 0.5*(x1+x2), 0.5*(y1+y2)
    w, h   = (x2-x1)*scale, (y2-y1)*scale
    return [cx - 0.5*w, cy - 0.5*h, cx + 0.5*w, cy + 0.5*h]

def clip_box_to_image(box, W, H):
    x1,y1,x2,y2 = box
    x1 = float(np.clip(x1, 0, W-1)); y1 = float(np.clip(y1, 0, H-1))
    x2 = float(np.clip(x2, 0, W));   y2 = float(np.clip(y2, 0, H))
    if x2 <= x1 + 1: x2 = min(W, x1 + 1)
    if y2 <= y1 + 1: y2 = min(H, y1 + 1)
    return [x1,y1,x2,y2]

def box_logit(model, img_t, pil_size, box):
    device = next(model.parameters()).device
    W, H = pil_size
    images_list, _ = model.transform([img_t.to(device)], None)
    x = images_list.tensors
    Ht, Wt = images_list.image_sizes[0]
    features_od = model.backbone(x)
    x1,y1,x2,y2 = box
    sx, sy = Wt/float(W), Ht/float(H)
    box_tr = torch.tensor([[x1*sx, y1*sy, x2*sx, y2*sy]], dtype=torch.float32, device=device)
    pooled = model.roi_heads.box_roi_pool(features_od, [box_tr], images_list.image_sizes)
    rep = model.roi_heads.box_head(pooled)
    class_logits, _ = model.roi_heads.box_predictor(rep)
    return float(class_logits[0,1].item())

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cpu")
    model = build_fasterrcnn(num_classes=2).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    headers = read_header_jsons(OUT_DIR)
    if not headers:
        print(f"No *_xai_header.json found in {OUT_DIR}.")
        raise SystemExit

    summary_rows = []

    for hpath in headers:
        with open(hpath, "r") as f:
            header = json.load(f)

        img_path = header["image"]
        roi_tgt  = header.get("xai_targets", {}).get("roi", None)
        if roi_tgt is None or "box" not in roi_tgt:
            print(f"[{Path(img_path).stem}] No ROI target; skip.")
            continue
        box0 = roi_tgt["box"]
        stem = Path(img_path).stem

        pil = Image.open(img_path).convert("RGB")
        img_t = to_tensor_norm(pil)
        W,H = pil.size
        box0 = clip_box_to_image(box0, W, H)
        base_logit = box_logit(model, img_t, (W,H), box0)

        occ_fracs = [0.2, 0.4, 0.6]
        occ_where = ["top","bottom","left","right"]
        occ_results = []
        for where in occ_where:
            xs, ys = [], []
            for f in occ_fracs:
                pil_occ = occlude_inside_roi(pil, box0, frac=f, where=where)
                from torchvision.transforms.functional import to_tensor, normalize
                import torch as _t
                img_occ = to_tensor(pil_occ)
                img_occ = normalize(img_occ, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                logit_occ = box_logit(model, img_occ, (W,H), box0)
                occ_results.append({"where": where, "frac": f, "logit": logit_occ})
                xs.append(f); ys.append(logit_occ)
            plt.figure(figsize=(3.6,2.8))
            plt.plot(xs, ys, marker="o")
            plt.axhline(base_logit, linestyle="--", linewidth=1)
            plt.xlabel(f"occluded fraction ({where})"); plt.ylabel("cls logit (square)")
            plt.title(f"{stem}: occlusion-{where}")
            plt.tight_layout()
            plt.savefig(os.path.join(OUT_DIR, f"{stem}_occl_{where}.png"), dpi=140, bbox_inches="tight")
            plt.close()

        jitters = [(-4,0),(4,0),(0,-4),(0,4)]
        jit_results = []
        xs, ys = [], []
        for dx,dy in jitters:
            box_j = clip_box_to_image(jitter_box(box0, dx, dy), W, H)
            logit_j = box_logit(model, img_t, (W,H), box_j)
            jit_results.append({"dx": dx, "dy": dy, "logit": logit_j})
            xs.append(f"{dx},{dy}"); ys.append(logit_j)
        plt.figure(figsize=(3.6,2.8))
        plt.bar(range(len(xs)), ys)
        plt.axhline(base_logit, linestyle="--", linewidth=1)
        plt.xticks(range(len(xs)), xs)
        plt.ylabel("cls logit (square)"); plt.title(f"{stem}: jitter")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{stem}_jitter.png"), dpi=140, bbox_inches="tight")
        plt.close()

        scales = [0.8, 0.9, 1.1, 1.2]
        scl_results = []
        xs, ys = [], []
        for s in scales:
            box_s = clip_box_to_image(scale_box(box0, s), W, H)
            logit_s = box_logit(model, img_t, (W,H), box_s)
            scl_results.append({"scale": s, "logit": logit_s})
            xs.append(s); ys.append(logit_s)
        plt.figure(figsize=(3.6,2.8))
        plt.plot(xs, ys, marker="o")
        plt.axhline(base_logit, linestyle="--", linewidth=1)
        plt.xlabel("box scale"); plt.ylabel("cls logit (square)")
        plt.title(f"{stem}: scale")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{stem}_scale.png"), dpi=140, bbox_inches="tight")
        plt.close()

        rec = {"image": img_path, "roi_box": box0, "baseline_logit": base_logit, "occlusion": occ_results, "jitter": jit_results, "scale": scl_results}
        with open(os.path.join(OUT_DIR, f"{stem}_cf_summary.json"), "w") as f:
            json.dump(rec, f, indent=2)

        summary_rows.append({"image": Path(img_path).name, "baseline": base_logit,
                             "worst_occl": min(occ_results, key=lambda r: r["logit"]),
                             "worst_jitter": min(jit_results, key=lambda r: r["logit"]),
                             "best_scale": max(scl_results, key=lambda r: r["logit"]) })
        print(f"Saved counterfactual plots/summary for {stem}")

    with open(os.path.join(OUT_DIR, "cf_run_summary.json"), "w") as f:
        json.dump(summary_rows, f, indent=2)

    print("Done: counterfactual probes saved in", OUT_DIR)

if __name__ == "__main__":
    main()
