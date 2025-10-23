# src/metrics/det.py

# Detection Metrics: Global AP@0.5, AR, And Size-Bucket AP

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


# Compute IoU For Two Boxes In XYXY Format
def compute_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-12
    return float(inter / union)


# 11-Point AP From TP/FP Lists
def _ap11_from_tp_fp(tp: np.ndarray, fp: np.ndarray, total_gts: int) -> float:
    if total_gts <= 0 or tp.size == 0:
        return 0.0
    # Sanity: Lengths Must Match
    if tp.shape != fp.shape:
        # Pad Shorter One With Zeros (Safety Net; Should Not Trigger With Correct Logic)
        n = max(tp.size, fp.size)
        if tp.size < n:
            tp = np.pad(tp, (0, n - tp.size))
        if fp.size < n:
            fp = np.pad(fp, (0, n - fp.size))

    tp_c = np.cumsum(tp)
    fp_c = np.cumsum(fp)
    recall = tp_c / float(total_gts)
    precision = tp_c / np.maximum(tp_c + fp_c, 1e-12)

    ap = 0.0
    for r in np.linspace(0, 1, 11):
        mask = (recall >= r)
        p_at_r = precision[mask].max() if mask.any() else 0.0
        ap += p_at_r
    return float(ap / 11.0)


# Global 11-Point AP@IoU=0.5
def ap_at_05(preds: List[Dict], gts: List[Dict], iou_thr: float = 0.5) -> float:
    # Flatten Predictions Into (Score, ImgId, Box)
    records = []
    total_gts = 0
    for i, (p, g) in enumerate(zip(preds, gts)):
        pb = np.asarray(p.get("boxes", np.zeros((0, 4), np.float32)), dtype=float)
        ps = np.asarray(p.get("scores", np.zeros((0,), np.float32)), dtype=float)
        gb = np.asarray(g.get("boxes", np.zeros((0, 4), np.float32)), dtype=float)
        total_gts += len(gb)
        for j in range(len(ps)):
            records.append((float(ps[j]), i, pb[j]))
    if not records:
        return 0.0
    records.sort(key=lambda x: x[0], reverse=True)

    # Greedy One-To-One Matching
    matched = set()
    tp_list, fp_list = [], []
    all_gb = [np.asarray(g.get("boxes", np.zeros((0, 4), np.float32)), dtype=float) for g in gts]
    for _, img_id, box in records:
        gb = all_gb[img_id]
        best_iou, best_j = 0.0, -1
        for j in range(len(gb)):
            if (img_id, j) in matched:
                continue
            iou = compute_iou(box, gb[j])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= iou_thr and best_j >= 0:
            matched.add((img_id, best_j))
            tp_list.append(1.0); fp_list.append(0.0)
        else:
            tp_list.append(0.0); fp_list.append(1.0)

    return _ap11_from_tp_fp(np.array(tp_list, float), np.array(fp_list, float), total_gts)


# Size Bucket Thresholds (Pixels^2, COCO-Like)
_AREA_SMALL_MAX = 32 * 32
_AREA_MED_MAX   = 96 * 96


# Compute Box Area In XYXY
def _box_area_xyxy(box: np.ndarray) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


# Build Subset Index Lists For Each Size Bin
def _subset_indices_by_size(gt_boxes: List[np.ndarray]) -> Dict[str, List[Tuple[int, int]]]:
    bins = {"small": [], "medium": [], "large": []}
    for i, gb in enumerate(gt_boxes):
        for j in range(len(gb)):
            a = _box_area_xyxy(gb[j])
            if a <= _AREA_SMALL_MAX:
                bins["small"].append((i, j))
            elif a <= _AREA_MED_MAX:
                bins["medium"].append((i, j))
            else:
                bins["large"].append((i, j))
    return bins


# AP@0.5 For A GT Subset (By Size)
def _ap50_for_subset(preds_np: List[Dict[str, np.ndarray]],
                     gts_np: List[np.ndarray],
                     subset: List[Tuple[int, int]]) -> float:
    if len(subset) == 0:
        return 0.0
    subset_set = set(subset)

    # Flatten Predictions
    records = []
    total_gts = len(subset)
    for img_id, p in enumerate(preds_np):
        pb = p["boxes"]; ps = p["scores"]
        for j in range(len(ps)):
            records.append((float(ps[j]), img_id, pb[j]))
    if not records:
        return 0.0
    records.sort(key=lambda x: x[0], reverse=True)

    # Greedy Matching Only Against Subset GTs
    matched = set()
    tp, fp = [], []
    for _, img_id, box in records:
        gb = gts_np[img_id]
        best_iou, best_j = 0.0, -1
        for j in range(len(gb)):
            if (img_id, j) not in subset_set:
                continue
            if (img_id, j) in matched:
                continue
            iou = compute_iou(box, gb[j])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= 0.5 and best_j >= 0:
            matched.add((img_id, best_j)); tp.append(1.0); fp.append(0.0)
        else:
            tp.append(0.0); fp.append(1.0)   # Keep TP/FP Aligned

    # Sanity Check
    assert len(tp) == len(fp), f"TP/FP Length Mismatch: {len(tp)} vs {len(fp)}"

    return _ap11_from_tp_fp(np.array(tp, float), np.array(fp, float), total_gts)


# Public API: Evaluate Model And Return (Metrics, Preds, GTs)
def evaluate_ap_by_size(model, data_loader, device, out_dir: str | None = None, tag: str = "val"
                        ) -> Tuple[Dict, List[Dict], List[Dict]]:
    # Returns (Metrics Dict, Predictions, Ground Truths)
    # Metrics Include ap50_global, ap_global (=ap50), ar_global, ap50_small/medium/large, And Counts
    import torch  # Local Import To Keep Module Lightweight When Not Used

    # Switch To Eval Mode
    model.eval()

    # Collect Predictions/GTs On CPU
    preds_cpu: List[Dict] = []
    gts_cpu: List[Dict] = []

    with torch.no_grad():
        for images, targets in data_loader:
            ims = [im.to(device) for im in images]
            outs = model(ims)
            for o, t in zip(outs, targets):
                boxes = o.get("boxes", torch.empty(0, 4))
                scores = o.get("scores", torch.empty(0))
                preds_cpu.append({
                    "boxes": boxes.detach().cpu().numpy().astype(float),
                    "scores": scores.detach().cpu().numpy().astype(float),
                })
                gb = t.get("boxes", torch.empty(0, 4))
                gts_cpu.append({"boxes": gb.detach().cpu().numpy().astype(float)})

    # Prepare NumPy Views
    preds_np = [{"boxes": p["boxes"], "scores": p["scores"]} for p in preds_cpu]
    gts_np   = [g["boxes"] for g in gts_cpu]

    # Global AP@0.5
    ap50_global = ap_at_05(
        preds=[{"boxes": p["boxes"], "scores": p["scores"]} for p in preds_cpu],
        gts=[{"boxes": g} for g in gts_np],
        iou_thr=0.5
    )

    # Simple AR Estimate (Best Recall Over Score Sweep)
    total_gts = int(sum(len(g) for g in gts_np))
    records = []
    for img_id, p in enumerate(preds_np):
        for j in range(len(p["scores"])):
            records.append((float(p["scores"][j]), img_id, p["boxes"][j]))
    records.sort(key=lambda x: x[0], reverse=True)

    matched = set()
    tp_marks = []
    for _, img_id, box in records:
        gb = gts_np[img_id]
        best_iou, best_j = 0.0, -1
        for j in range(len(gb)):
            if (img_id, j) in matched:
                continue
            iou = compute_iou(box, gb[j])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= 0.5 and best_j >= 0:
            matched.add((img_id, best_j)); tp_marks.append(1.0)
        else:
            tp_marks.append(0.0)
    ar_global = float(np.max(np.cumsum(np.array(tp_marks, float)) / max(1, total_gts))) if (total_gts > 0 and tp_marks) else 0.0

    # Size-Bucket APs
    bins = _subset_indices_by_size(gts_np)
    ap50_small  = _ap50_for_subset(preds_np, gts_np, bins["small"])
    ap50_medium = _ap50_for_subset(preds_np, gts_np, bins["medium"])
    ap50_large  = _ap50_for_subset(preds_np, gts_np, bins["large"])

    # Metrics Dictionary
    metrics = {
        "ap50_global": float(ap50_global),
        "ap_global": float(ap50_global),   # Same As ap50 (Single IoU)
        "ar_global": float(ar_global),
        "ap50_small": float(ap50_small),
        "ap50_medium": float(ap50_medium),
        "ap50_large": float(ap50_large),
        "num_images": len(gts_np),
        "num_gts": total_gts,
        "num_small": len(bins["small"]),
        "num_medium": len(bins["medium"]),
        "num_large": len(bins["large"]),
    }

    # JSON-Friendly Copies
    preds_list = [{"boxes": p["boxes"].tolist(), "scores": p["scores"].tolist()} for p in preds_np]
    gts_list   = [{"boxes": g.tolist()} for g in gts_np]
    return metrics, preds_list, gts_list
