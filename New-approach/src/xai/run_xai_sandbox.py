# src/xai/make_xai_sandbox.py
import os, json, datetime
from pathlib import Path

import torch
from PIL import Image

from src.models.fasterrcnn_cpu import build_fasterrcnn_cpu
from src.dataio.voc_parser import paired_image_xml_list
from src.dataio.split_utils import subsample_pairs
from src.utils.viz import show_prediction  # draws boxes on a de-normalized image

# ----------------- CONFIG (edit paths/fractions here) -----------------
# DATA_ROOT    = r"E:\WPT-Project\Data\sized_squares_filled"
DATA_ROOT    = r"E:\WPT-Project\Data\sized_rectangles_filled"
IMG_DIR_TEST = fr"{DATA_ROOT}\test"
XML_DIR_ALL  = fr"{DATA_ROOT}\annotations"

OUTPUT_DIR   = "outputs/xai_det"
CKPT_PATH    = "outputs/fasterrcnn_squares_cpu.pt"

F_TEST       = 0.05     # fraction of test to consider
SEED         = 14
K_IMAGES     = 5    # how many images to prepare for XAI
SCORE_THR    = 0.50     # confidence threshold for final detections
# ----------------------------------------------------------------------


def _ensure_out():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def _denorm_for_viz(img_tensor):
    """
    img_tensor: [3,H,W] normalized with ImageNet stats.
    show_prediction already de-normalizes internally in your repo,
    but we keep this helper in case you want to draw manually later.
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (img_tensor * std + mean).clamp(0, 1)

def _load_image_as_tensor(img_path):
    # Minimal test-time transforms: ToTensor + Normalize
    import torchvision.transforms as T
    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    img = Image.open(img_path).convert("RGB")
    return tfm(img), img.size  # (tensor [3,H,W], (W,H))

def main():
    _ensure_out()
    device = torch.device("cpu")

    # -------- Model: rebuild + load weights --------
    model = build_fasterrcnn_cpu(num_classes=2).to(device)
    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # -------- Select test subset (fraction + first K) --------
    all_pairs   = paired_image_xml_list(IMG_DIR_TEST, XML_DIR_ALL)
    frac_pairs  = subsample_pairs(all_pairs, fraction=F_TEST, seed=SEED)
    chosen_pairs = frac_pairs[:K_IMAGES] if len(frac_pairs) >= K_IMAGES else frac_pairs

    # Save indices manifest
    with open(os.path.join(OUTPUT_DIR, "indices.json"), "w") as f:
        json.dump(
            {
                "images": [p[0] for p in chosen_pairs],
                "fraction": F_TEST,
                "seed": SEED,
                "k_images": len(chosen_pairs),
            },
            f,
            indent=2,
        )

    # -------- Run summary scaffold --------
    run_summary = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": CKPT_PATH,
        "fraction": F_TEST,
        "seed": SEED,
        "score_thr": SCORE_THR,
        "n_available_test_pairs": len(all_pairs),
        "n_fractioned": len(frac_pairs),
        "n_processed": len(chosen_pairs),
        "processed_images": [],
    }

    # -------- Process each chosen image --------
    for idx, (img_path, xml_path) in enumerate(chosen_pairs, start=1):
        img_stem = Path(img_path).stem
        out_png  = os.path.join(OUTPUT_DIR, f"{img_stem}_pred.png")
        out_json = os.path.join(OUTPUT_DIR, f"{img_stem}_xai_header.json")

        # 1) Load & transform
        img_tensor, (W, H) = _load_image_as_tensor(img_path)
        img_list = [img_tensor.to(device)]

        # 2) Forward pass (no gradients)
        with torch.no_grad():
            preds = model(img_list)

        pred = preds[0]
        boxes  = pred.get("boxes",  torch.empty(0,4)).cpu()
        scores = pred.get("scores", torch.empty(0)).cpu()
        labels = pred.get("labels", torch.empty(0, dtype=torch.long)).cpu()

        # 3) Select top detections
        keep = []
        if scores.numel() > 0:
            # filter by threshold
            for i, s in enumerate(scores):
                if float(s) >= SCORE_THR:
                    keep.append(i)
            # if none above thr, keep single best
            if len(keep) == 0:
                keep = [int(scores.argmax().item())]
        # # Cap to top 3
        if len(keep) > 3:
            keep = sorted(keep, key=lambda i: float(scores[i]), reverse=True)[:3]

        kept_dets = []
        for i in keep:
            b = boxes[i].tolist()
            s = float(scores[i].item())
            kept_dets.append({"box": [float(x) for x in b], "score": s})

        # 4) Choose XAI targets
        # ROI target: the highest-score kept detection (or None)
        roi_target = None
        if len(kept_dets) > 0:
            roi_target = max(kept_dets, key=lambda d: d["score"])

        # RPN target (A1.1): we defer the actual RPN pick to A2.
        # For now, record a placeholder that mirrors the ROI target's box
        # so you have coordinates to work with. In A2 you’ll backprop the
        # objectness score at the matched location.
        rpn_target = None
        if roi_target is not None:
            rpn_target = {"box": roi_target["box"], "note": "placeholder; refine in A2 from RPN objectness peak"}

        # 5) Save a quick detection overlay (context)
        # (show_prediction expects img tensor on CPU and a pred dict with 'boxes'/'scores')
        # We'll repackage a minimal pred dict from kept detections.
        mini_pred = {
            "boxes": torch.tensor([d["box"] for d in kept_dets], dtype=torch.float32),
            "scores": torch.tensor([d["score"] for d in kept_dets], dtype=torch.float32),
            "labels": torch.ones(len(kept_dets), dtype=torch.long),  # single foreground class
        }
        # Draw & save
        show_prediction(img_tensor.cpu(), mini_pred, gt=None, score_thr=SCORE_THR, save_path=out_png)

        # 6) Write the per-image XAI header JSON
        header = {
            "image": img_path,
            "xml": xml_path,
            "img_size": [W, H],
            "checkpoint": CKPT_PATH,
            "score_thr": SCORE_THR,
            "detections": kept_dets,                   # list of {box, score}
            "xai_targets": {
                "rpn": rpn_target,                     # placeholder; refine in A2
                "roi": roi_target                      # highest-score kept det
            }
        }
        with open(out_json, "w") as f:
            json.dump(header, f, indent=2)

        # 7) Update run summary
        run_summary["processed_images"].append({
            "basename": Path(img_path).name,
            "pred_overlay": out_png,
            "xai_header": out_json,
            "n_kept_detections": len(kept_dets)
        })

        print(f"[{idx}/{len(chosen_pairs)}] Saved {Path(out_png).name} and XAI header.")

    # -------- Save run summary --------
    with open(os.path.join(OUTPUT_DIR, "run_summary.json"), "w") as f:
        json.dump(run_summary, f, indent=2)

    print("\nA1.1 sandbox ready.")
    print(f" - Run summary : {os.path.join(OUTPUT_DIR, 'run_summary.json')}")
    print(f" - Overlays    : {OUTPUT_DIR}\\*_pred.png")
    print(f" - Headers     : {OUTPUT_DIR}\\*_xai_header.json")


if __name__ == "__main__":
    main()
