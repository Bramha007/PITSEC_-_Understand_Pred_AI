import os, json, logging, datetime, argparse, torch, numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

from src.data.voc import paired_image_xml_list
from src.data.split import subsample_pairs
from src.constants.norms import IMAGENET_MEAN, IMAGENET_STD

from src.models.det_fasterrcnn import build_fasterrcnn
from src.viz.det import show_prediction
import torchvision.transforms as T

def build_outdir(base_out: str) -> str:
    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(base_out, run_id)
    os.makedirs(out, exist_ok=True)
    return out

def load_t(img_path): 
    tfm = T.Compose([T.ToTensor(), T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return tfm(Image.open(img_path).convert("RGB"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="outputs/xai_det")
    ap.add_argument("--fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=14)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--score-thr", type=float, default=0.5)
    args = ap.parse_args()

    out_dir = build_outdir(args.out)
    device = torch.device("cpu")
    model = build_fasterrcnn(num_classes=2).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device)); model.eval()

    all_pairs = paired_image_xml_list(args.images, args.ann)
    pairs = subsample_pairs(all_pairs, fraction=args.fraction, seed=args.seed)[:args.k]

    indices = {"images": [p[0] for p in pairs], "fraction": args.fraction, "seed": args.seed, "k_images": len(pairs)}
    with open(os.path.join(out_dir, "indices.json"), "w") as f: json.dump(indices, f, indent=2)

    run_summary = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": args.ckpt,
        "score_thr": args.score_thr,
        "n_available_test_pairs": len(all_pairs),
        "n_processed": len(pairs),
        "processed_images": [],
    }

    for i,(img_path, xml_path) in enumerate(pairs, start=1):
        stem = Path(img_path).stem
        out_png  = os.path.join(out_dir, f"{stem}_pred.png")
        out_json = os.path.join(out_dir, f"{stem}_xai_header.json")

        img_t = load_t(img_path)
        with torch.no_grad():
            pred = model([img_t.to(device)])[0]

        boxes = pred.get("boxes", torch.empty(0,4)).cpu()
        scores= pred.get("scores", torch.empty(0)).cpu()

        keep = [i for i,s in enumerate(scores) if float(s)>=args.score_thr]
        if len(keep)==0 and scores.numel()>0: keep=[int(scores.argmax().item())]
        if len(keep)>3:
            keep = sorted(keep, key=lambda i: float(scores[i]), reverse=True)[:3]
        kept = [{"box":[float(x) for x in boxes[i].tolist()], "score": float(scores[i].item())} for i in keep]
        roi = max(kept, key=lambda d: d["score"]) if kept else None
        rpn = {"box": roi["box"], "note": "placeholder"} if roi else None

        mini = {
            "boxes": boxes if len(keep)==0 else torch.tensor([d["box"] for d in kept], dtype=torch.float32),
            "scores": scores if len(keep)==0 else torch.tensor([d["score"] for d in kept], dtype=torch.float32),
            "labels": torch.ones(len(kept), dtype=torch.long),
        }
        show_prediction(img_t.cpu(), mini, gt=None, score_thr=args.score_thr, save_path=out_png)

        header = {"schema":"xai_header/v1", "image": img_path, "xml": xml_path, "checkpoint": args.ckpt,
                  "score_thr": args.score_thr, "detections": kept, "xai_targets": {"rpn": rpn, "roi": roi}}
        with open(out_json, "w") as f: json.dump(header, f, indent=2)

        run_summary["processed_images"].append({"basename": Path(img_path).name, "pred_overlay": out_png, "xai_header": out_json})
        print(f"[{i}/{len(pairs)}] {Path(out_png).name} + header saved." )

    with open(os.path.join(out_dir, "run_summary.json"), "w") as f: json.dump(run_summary, f, indent=2)
    print("Sandbox ready in:", out_dir)

if __name__ == "__main__": main()
