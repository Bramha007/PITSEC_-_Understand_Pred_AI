# src/metrics/ar.py
# Aspect-Ratio bucketed AP@0.5 evaluation for object detection (class-agnostic by default).
# Works with standard torchvision-style models and dataloaders: images -> model -> {boxes, scores, labels}
# Saves JSON + bar chart to out_dir.
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence
import os, json
import numpy as np

try:
    # Lazy import for charting (only if available in the runtime)
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except Exception:
    _HAS_PLT = False

@dataclass
class ARBucket:
    name: str
    lo: float   # inclusive
    hi: float   # exclusive

DEFAULT_AR_BUCKETS: Tuple[ARBucket, ...] = (
    ARBucket("near_square", 1.0, 1.3334),  # ≤ 4:3
    ARBucket("moderate",    1.3334, 2.0),  # 4:3–2:1
    ARBucket("skinny",      2.0,  1e9),    # ≥ 2:1
)

def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a: [Na,4], b: [Nb,4], format xyxy
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ixmin = np.maximum(a[:, None, 0], b[None, :, 0])
    iymin = np.maximum(a[:, None, 1], b[None, :, 1])
    ixmax = np.minimum(a[:, None, 2], b[None, :, 2])
    iymax = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.maximum(ixmax - ixmin, 0.0)
    ih = np.maximum(iymax - iymin, 0.0)
    inter = iw * ih
    a_area = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    b_area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = a_area[:, None] + b_area[None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    return iou.astype(np.float32)

def _compute_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    # VOC-style interpolation (area under precision-recall envelope)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i-1] = np.maximum(mpre[i-1], mpre[i])
    # Points where recall changes
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[i+1] - mrec[i]) * mpre[i+1]))
    return ap

def evaluate_ap_by_ar(
    model,
    data_loader,
    device,
    iou_thresh: float = 0.5,
    buckets: Sequence[ARBucket] = DEFAULT_AR_BUCKETS,
    out_dir: str | None = None,
    tag: str = "val",
    class_agnostic: bool = True,
) -> Dict:
    """
    Evaluate AP@0.5 per AR bucket.
    - Bucket assignment is based on GT AR = long/short.
    - Only images containing at least one GT in the bucket are included for that bucket.
    - Predictions matched to GTs outside the bucket are ignored for that bucket's PR computation.
    - Unmatched predictions on included images count as FPs for that bucket.
    Returns a dict with fields:
      {'ap50_by_ar': {bucket_name: value}, 'counts': {...}, 'detail': {...}}
    """
    model.eval()
    all_imgs = []
    with_no_grad = getattr(__import__('torch'), 'no_grad')
    with with_no_grad():
        for images, targets in data_loader:
            ims = [im.to(device) for im in images]
            preds = model(ims)
            # Move to CPU numpy for evaluation
            pred_cpu = []
            for p in preds:
                boxes = p.get('boxes', None)
                scores = p.get('scores', None)
                labels = p.get('labels', None)
                boxes = boxes.detach().cpu().numpy() if boxes is not None else np.zeros((0,4), np.float32)
                scores = scores.detach().cpu().numpy() if scores is not None else np.zeros((0,), np.float32)
                labels = labels.detach().cpu().numpy() if labels is not None else np.zeros((0,), np.int64)
                pred_cpu.append((boxes, scores, labels))
            tgt_cpu = []
            for t in targets:
                boxes = t.get('boxes', None)
                labels = t.get('labels', None)
                boxes = boxes.detach().cpu().numpy() if boxes is not None else np.zeros((0,4), np.float32)
                labels = labels.detach().cpu().numpy() if labels is not None else np.zeros((0,), np.int64)
                tgt_cpu.append((boxes, labels))
            all_imgs.append((pred_cpu, tgt_cpu))

    # Flatten across batches
    preds_per_image = []
    tgts_per_image  = []
    for (pred_batch, tgt_batch) in all_imgs:
        for p, t in zip(pred_batch, tgt_batch):
            preds_per_image.append(p)
            tgts_per_image.append(t)

    # Prepare bucket membership per image
    gt_ar_per_image: List[np.ndarray] = []
    img_has_bucket: List[List[bool]] = []
    for (gt_boxes, _) in tgts_per_image:
        if gt_boxes.size == 0:
            ars = np.zeros((0,), dtype=np.float32)
        else:
            w = gt_boxes[:,2] - gt_boxes[:,0]
            h = gt_boxes[:,3] - gt_boxes[:,1]
            ar = np.where(h > 0, w / np.maximum(h, 1e-9), 0.0)
            ar = np.maximum(ar, 1.0/np.maximum(ar, 1e-9))
            ars = ar.astype(np.float32)
        gt_ar_per_image.append(ars)
        flags = []
        for b in buckets:
            in_b = np.logical_and(ars >= b.lo, ars < b.hi).any()
            flags.append(bool(in_b))
        img_has_bucket.append(flags)

    results = {"ap50_by_ar": {}, "counts": {}, "detail": {}}

    for bi, b in enumerate(buckets):
        # Gather images that have at least one GT in this AR bucket
        eligible = [i for i, flags in enumerate(img_has_bucket) if flags[bi]]
        if len(eligible) == 0:
            results["ap50_by_ar"][b.name] = 0.0
            results["counts"][b.name] = {"num_images": 0, "num_gts": 0}
            continue

        pred_records = []  # (score, is_tp)
        num_gts_total = 0

        # For each image, per-class matching (or class-agnostic)
        for i in eligible:
            pred_boxes, pred_scores, pred_labels = preds_per_image[i]
            gt_boxes, gt_labels = tgts_per_image[i]

            # Select GTs in this AR bucket
            if gt_boxes.size == 0:
                continue
            w = gt_boxes[:,2] - gt_boxes[:,0]
            h = gt_boxes[:,3] - gt_boxes[:,1]
            ar = np.maximum(w/h, h/np.maximum(w,1e-9))
            in_bucket = np.logical_and(ar >= b.lo, ar < b.hi)
            gt_boxes_b = gt_boxes[in_bucket]
            gt_labels_b = gt_labels[in_bucket]
            num_gts_total += int(gt_boxes_b.shape[0])
            if gt_boxes_b.shape[0] == 0:
                # No GTs in bucket for this image; all predictions on this image are counted as FP
                for s in pred_scores.tolist():
                    pred_records.append((float(s), 0))
                continue

            # Optionally class-agnostic
            if class_agnostic:
                # Greedy match by IoU to any gt in bucket
                ious = _iou_matrix(pred_boxes, gt_boxes_b)  # [Np, Ng]
                matched_gt = set()
                order = np.argsort(-pred_scores)  # high to low
                for p_idx in order:
                    if pred_boxes.shape[0] == 0:
                        break
                    # match to best-IOU GT not yet matched
                    best_gt = -1
                    best_iou = 0.0
                    for g_idx in range(gt_boxes_b.shape[0]):
                        if g_idx in matched_gt:
                            continue
                        iou = ious[p_idx, g_idx]
                        if iou > best_iou:
                            best_iou = iou
                            best_gt = g_idx
                    if best_gt >= 0 and best_iou >= iou_thresh:
                        matched_gt.add(best_gt)
                        pred_records.append((float(pred_scores[p_idx]), 1))
                    else:
                        pred_records.append((float(pred_scores[p_idx]), 0))
            else:
                # Per-class matching
                classes = np.unique(np.concatenate([pred_labels, gt_labels_b]))
                for c in classes:
                    pb = pred_boxes[pred_labels == c]
                    ps = pred_scores[pred_labels == c]
                    gb = gt_boxes_b[gt_labels_b == c]
                    if gb.shape[0] == 0:
                        for s in ps.tolist():
                            pred_records.append((float(s), 0))
                        continue
                    ious = _iou_matrix(pb, gb)
                    matched_gt = set()
                    order = np.argsort(-ps)
                    for p_idx in order:
                        best_gt = -1
                        best_iou = 0.0
                        for g_idx in range(gb.shape[0]):
                            if g_idx in matched_gt:
                                continue
                            iou = ious[p_idx, g_idx]
                            if iou > best_iou:
                                best_iou = iou
                                best_gt = g_idx
                        if best_gt >= 0 and best_iou >= iou_thresh:
                            matched_gt.add(best_gt)
                            pred_records.append((float(ps[p_idx]), 1))
                        else:
                            pred_records.append((float(ps[p_idx]), 0))

        if num_gts_total == 0:
            results["ap50_by_ar"][b.name] = 0.0
            results["counts"][b.name] = {"num_images": len(eligible), "num_gts": 0}
            continue

        # Build PR
        scores = np.array([r[0] for r in pred_records], dtype=np.float32)
        tps    = np.array([r[1] for r in pred_records], dtype=np.int32)
        if scores.size == 0:
            results["ap50_by_ar"][b.name] = 0.0
            results["counts"][b.name] = {"num_images": len(eligible), "num_gts": int(num_gts_total)}
            continue
        order = np.argsort(-scores)
        tps = tps[order]
        fps = 1 - tps
        tp_cum = np.cumsum(tps)
        fp_cum = np.cumsum(fps)
        rec = tp_cum / max(num_gts_total, 1)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)
        ap = _compute_ap(rec, prec)
        results["ap50_by_ar"][b.name] = float(ap)
        results["counts"][b.name] = {"num_images": len(eligible), "num_gts": int(num_gts_total)}
        results["detail"][b.name] = {
            "recall_points": rec.tolist(),
            "precision_points": prec.tolist(),
        }

    # Save artifacts
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_json = os.path.join(out_dir, f"ar_metrics_{tag}.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        if _HAS_PLT:
            names = [b.name for b in buckets]
            vals = [results["ap50_by_ar"].get(n, 0.0) for n in names]
            plt.figure(figsize=(6,4))
            xs = range(len(names))
            plt.bar(xs, vals)
            plt.xticks(xs, names, rotation=15)
            plt.ylim([0.0, 1.0])
            plt.ylabel("AP@0.5")
            plt.title(f"AR-bucket AP ({tag})")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"ar_metrics_{tag}.png"), dpi=160)
            plt.close()

    return results
