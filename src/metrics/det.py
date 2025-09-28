# src/metrics/det.py
import json, os
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.ops import box_iou

# Size buckets: [0,12]=~8px, (12,24]=~16, (24,48]=~32, (48,96]=~64, (96,inf)=~128
from src.constants.sizes import DET_EDGES as BUCKET_EDGES, DET_LABELS as BUCKET_LABELS

def side_from_boxes(boxes: torch.Tensor) -> torch.Tensor:
    # boxes: [N,4] xyxy
    wh = boxes[:, 2:4] - boxes[:, 0:2]
    return torch.max(wh[:, 0], wh[:, 1])

def bucket_index(side: float) -> int:
    for i in range(len(BUCKET_EDGES)-1):
        if BUCKET_EDGES[i] <= side <= BUCKET_EDGES[i+1]:
            return i
    return len(BUCKET_EDGES)-2

@dataclass
class PRData:
    scores: List[float]
    tp: List[int]
    fp: List[int]
    npos: int

def _accumulate_pr(all_scores: List[float], all_tp: List[int], all_fp: List[int], npos: int):
    if len(all_scores) == 0 or npos == 0:
        return np.zeros(1), np.zeros(1), 0.0
    order = np.argsort(-np.array(all_scores))
    tp = np.array(all_tp)[order]
    fp = np.array(all_fp)[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / float(npos)
    precision = tp_cum / np.maximum(1, tp_cum + fp_cum)
    # 11-point VOC AP
    ap = 0.0
    for r in np.linspace(0, 1, 11):
        p = np.max(precision[recall >= r]) if np.any(recall >= r) else 0.0
        ap += p / 11.0
    return precision, recall, ap

@torch.no_grad()
def collect_pr_data(model, loader, device="cpu") -> Tuple[PRData, Dict[int, PRData]]:
    model.eval()
    # Global
    scores, tp, fp = [], [], []
    npos_global = 0
    # Per-bucket
    bucket_scores = {i: [] for i in range(len(BUCKET_LABELS))}
    bucket_tp     = {i: [] for i in range(len(BUCKET_LABELS))}
    bucket_fp     = {i: [] for i in range(len(BUCKET_LABELS))}
    bucket_npos   = {i: 0   for i in range(len(BUCKET_LABELS))}

    for imgs, targets in loader:
        imgs = [img.to(device) for img in imgs]
        gts  = [t["boxes"].to(device) for t in targets]
        preds = model(imgs)

        for pred, gt in zip(preds, gts):
            # Count positives (global + per-bucket)
            npos_global += gt.size(0)
            for s in side_from_boxes(gt).cpu().tolist():
                bucket_npos[bucket_index(s)] += 1

            if pred["boxes"].numel() == 0:
                continue
            pb = pred["boxes"]; ps = pred["scores"]
            order = torch.argsort(ps, descending=True)
            pb = pb[order]; ps = ps[order]
            ious = box_iou(pb, gt) if gt.numel() else torch.zeros((pb.size(0), 0), device=pb.device)

            matched_gt = set()
            for i in range(pb.size(0)):
                s = float(ps[i].item())
                # global bookkeeping
                if gt.numel():
                    j = torch.argmax(ious[i]).item()
                    hit = (ious[i, j] >= 0.5) and (j not in matched_gt)
                else:
                    hit = False
                # mark GT as matched if this is a TP to prevent double counting
                if hit:
                    matched_gt.add(j)
                scores.append(s)
                tp.append(1 if hit else 0)
                fp.append(0 if hit else 1)

                # per-bucket bookkeeping (bucket by the matched GT if hit; otherwise bucket by nearest GT)
                if gt.numel():
                    ref_j = j
                    b_idx = bucket_index(float(side_from_boxes(gt[ref_j:ref_j+1])[0].item()))
                else:
                    # no GT -> put in largest bucket arbitrarily to not break accounting
                    b_idx = len(BUCKET_LABELS)-1
                bucket_scores[b_idx].append(s)
                bucket_tp[b_idx].append(1 if hit else 0)
                bucket_fp[b_idx].append(0 if hit else 1)

    global_pr = PRData(scores, tp, fp, npos_global)
    per_bucket = {i: PRData(bucket_scores[i], bucket_tp[i], bucket_fp[i], bucket_npos[i]) for i in bucket_scores}
    return global_pr, per_bucket

def evaluate_ap_by_size(model, loader, device="cpu", out_dir="outputs", tag="val"):
    os.makedirs(out_dir, exist_ok=True)
    global_pr, per_bucket = collect_pr_data(model, loader, device=device)

    prec, rec, ap_global = _accumulate_pr(global_pr.scores, global_pr.tp, global_pr.fp, global_pr.npos)
    ap_buckets = {}
    for i in range(len(BUCKET_LABELS)):
        pbd = per_bucket[i]
        _, _, ap_i = _accumulate_pr(pbd.scores, pbd.tp, pbd.fp, pbd.npos)
        ap_buckets[BUCKET_LABELS[i]] = float(ap_i)

    # Compute FP/FN per bucket
    fpfn = _compute_fp_fn(per_bucket)

    # Save JSON
    js = {
        "tag": tag,
        "ap50_global": float(ap_global),
        "ap50_by_size": ap_buckets,
        "npos_total": int(global_pr.npos),
        "npos_by_size": {BUCKET_LABELS[i]: int(per_bucket[i].npos) for i in range(len(BUCKET_LABELS))},
        "fp_fn_by_size": fpfn,
    }
    out_json = os.path.join(out_dir, f"det_metrics_{tag}.json")
    with open(out_json, "w") as f:
        json.dump(js, f, indent=2)
    # Bar charts
    out_png = os.path.join(out_dir, f"det_ap_by_size_{tag}.png")
    _plot_bars(ap_buckets, out_png, title=f"AP@0.5 by size ({tag})")

    out_fpfn_png = os.path.join(out_dir, f"det_fpfn_by_size_{tag}.png")
    _plot_fp_fn_bars(fpfn, out_fpfn_png, title=f"FP/FN by size ({tag})")

    # Optional global PR curve
    _plot_pr(prec, rec, os.path.join(out_dir, f"det_pr_curve_{tag}.png"))

    return js, out_json, out_png

def _plot_bars(ap_dict: Dict[str, float], out_path: str, title: str):
    xs = list(ap_dict.keys())
    ys = [ap_dict[k] for k in xs]
    plt.figure(figsize=(6,4))
    plt.bar(xs, ys)
    plt.ylim(0,1.0); plt.ylabel("AP@0.5"); plt.title(title)
    for i,y in enumerate(ys):
        plt.text(i, y+0.02, f"{y:.2f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close()

def _plot_pr(precision, recall, out_path: str):
    plt.figure(figsize=(5,4))
    plt.plot(recall, precision)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR curve (global)")
    plt.xlim(0,1); plt.ylim(0,1)
    plt.tight_layout(); plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close()


def _compute_fp_fn(per_bucket) -> dict:
    res = {}
    for i in range(len(BUCKET_LABELS)):
        pbd = per_bucket[i]
        tp_sum = int(sum(pbd.tp))
        fp_sum = int(sum(pbd.fp))
        fn_sum = int(max(0, pbd.npos - tp_sum))
        res[BUCKET_LABELS[i]] = {"tp": tp_sum, "fp": fp_sum, "fn": fn_sum, "npos": int(pbd.npos)}
    return res

def _plot_fp_fn_bars(fpfn: dict, out_path: str, title: str):
    import matplotlib.pyplot as plt
    labels = list(fpfn.keys())
    fps = [fpfn[k]["fp"] for k in labels]
    fns = [fpfn[k]["fn"] for k in labels]
    xs = range(len(labels))
    width = 0.4
    plt.figure(figsize=(7,4))
    plt.bar([x - width/2 for x in xs], fps, width=width, label="FP")
    plt.bar([x + width/2 for x in xs], fns, width=width, label="FN")
    plt.xticks(list(xs), labels)
    plt.ylabel("Count"); plt.title(title)
    plt.legend()
    plt.tight_layout(); plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close()
