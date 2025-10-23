# src/metrics/cls.py

# Computes Classification And Regression Metrics With Optional Bucketed Evaluation

from __future__ import annotations

# Standard Library
from typing import Dict, List, Sequence, Iterable, Optional

# Third-Party
import numpy as np

# Local
from src.constants.sizes import CLS_EDGES


# Bucket Utilities
def _bucket_names(edges: Sequence[float]) -> List[str]:
    names = []
    for i in range(len(edges) - 1):
        names.append(f"{int(edges[i])}-{int(edges[i + 1])}")
    names.append(f"{int(edges[-1])}+")
    return names


def _bin_indices(values: np.ndarray, edges: Sequence[float]) -> np.ndarray:
    # Right-Open Policy [Left, Right)
    return np.digitize(values, edges, right=False)


def top1_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float((y_true == y_pred).mean())


def bucket_accuracy(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    img_sizes: Sequence[float],
    edges: Sequence[float] | None = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sizes = np.asarray(img_sizes, dtype=float)
    edges = edges or CLS_EDGES

    bins = _bin_indices(sizes, edges)
    names = _bucket_names(edges)

    out: Dict[str, float] = {}
    out["acc_global"] = top1_accuracy(y_true, y_pred)

    for b in range(len(edges)):
        sel = bins == b
        if not np.any(sel):
            out[f"acc_{names[b]}"] = float("nan")
            out[f"count_{names[b]}"] = 0
            continue
        acc = float((y_true[sel] == y_pred[sel]).mean())
        out[f"acc_{names[b]}"] = acc
        out[f"count_{names[b]}"] = int(sel.sum())

    return out


def evaluate_cls_with_buckets(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    img_sizes: Sequence[float],
    edges: Sequence[float] | None = None,
) -> Dict[str, float]:
    return bucket_accuracy(y_true, y_pred, img_sizes, edges or CLS_EDGES)


def compute_cls_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    img_sizes: Sequence[float],
    edges: Sequence[float] | None = None,
) -> Dict[str, float]:
    return evaluate_cls_with_buckets(y_true, y_pred, img_sizes, edges or CLS_EDGES)


# Regression Metrics For (Short, Long)
def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else float("nan")


def _regression_metrics_1d(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(np.float64)
    y_pred = np.asarray(y_pred).astype(np.float64)
    diff = y_pred - y_true

    if y_true.size == 0:
        keys = ["mae", "mse", "rmse", "mape", "smape", "r2", "pearson"]
        return {k: float("nan") for k in keys}

    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(max(mse, 0.0)))

    eps = 1e-8
    mape = float(np.mean(np.abs(diff) / (np.abs(y_true) + eps))) * 100.0
    smape = float(np.mean(2.0 * np.abs(diff) / (np.abs(y_true) + np.abs(y_pred) + eps))) * 100.0

    ss_res = float(np.sum(diff ** 2))
    mu = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - mu) ** 2))
    ratio = _safe_div(ss_res, ss_tot)
    r2 = 1.0 - ratio if np.isfinite(ratio) else float("nan")

    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        pearson = float("nan")
    else:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])

    return {"mae": mae, "mse": mse, "rmse": rmse, "mape": mape, "smape": smape, "r2": r2, "pearson": pearson}


def _bucket_regression_stats(
    y_true_1d: np.ndarray,
    y_pred_1d: np.ndarray,
    edges: Sequence[float],
) -> Dict[str, float]:
    edges = list(edges)
    names = _bucket_names(edges)
    t_bins = np.digitize(y_true_1d, edges, right=False)
    p_bins = np.digitize(y_pred_1d, edges, right=False)

    out: Dict[str, float] = {}
    out["bucket_acc_global"] = float(np.mean(t_bins == p_bins)) if len(t_bins) else 0.0

    for b in range(len(edges)):
        sel = t_bins == b
        cnt = int(np.sum(sel))
        out[f"bucket_count_{names[b]}"] = cnt
        if cnt == 0:
            out[f"bucket_acc_{names[b]}"] = float("nan")
            out[f"bucket_mae_{names[b]}"] = float("nan")
            out[f"bucket_rmse_{names[b]}"] = float("nan")
            continue
        out[f"bucket_acc_{names[b]}"] = float(np.mean(p_bins[sel] == t_bins[sel]))
        diff = (y_pred_1d[sel] - y_true_1d[sel]).astype(np.float64)
        out[f"bucket_mae_{names[b]}"] = float(np.mean(np.abs(diff)))
        out[f"bucket_rmse_{names[b]}"] = float(np.sqrt(np.mean(diff ** 2)))

    return out


def evaluate_regression_with_buckets(
    y_true_2d: Sequence[Sequence[float]],
    y_pred_2d: Sequence[Sequence[float]],
    bucket_on: str = "long",
    edges: Optional[Sequence[float]] = None,
    which: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true_2d, dtype=np.float64)
    y_pred = np.asarray(y_pred_2d, dtype=np.float64)
    if y_true.size == 0:
        y_true = np.zeros((0, 2), dtype=np.float64)
        y_pred = np.zeros((0, 2), dtype=np.float64)

    want = set((which or ["mae", "mse", "rmse"]).copy())
    want_bucket = any(w.lower().startswith("bucket") for w in want)

    out: Dict[str, float] = {}

    # Per-Dimension Metrics
    for dim, name in enumerate(["short", "long"]):
        md = _regression_metrics_1d(y_true[:, dim], y_pred[:, dim])
        for k, v in md.items():
            out[f"{name}_{k}"] = v

    # Overall (Macro Average)
    for k in ["mae", "mse", "rmse", "mape", "smape", "r2", "pearson"]:
        out[f"overall_{k}"] = float(np.nanmean([out[f"short_{k}"], out[f"long_{k}"]]))

    # Bucketed Metrics On Selected Dimension
    if want_bucket:
        dim = 0 if str(bucket_on).lower().startswith("s") else 1
        e = list(edges) if edges is not None else list(CLS_EDGES)
        bkt = _bucket_regression_stats(y_true[:, dim], y_pred[:, dim], e)
        out.update(bkt)

    return out


def compute_regression_metrics(
    y_true_2d: Sequence[Sequence[float]],
    y_pred_2d: Sequence[Sequence[float]],
    which: Optional[Iterable[str]] = None,
    bucket_on: str = "long",
    bucket_edges: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    return evaluate_regression_with_buckets(
        y_true_2d=y_true_2d,
        y_pred_2d=y_pred_2d,
        bucket_on=bucket_on,
        edges=bucket_edges if bucket_edges is not None else CLS_EDGES,
        which=which,
    )


def compute_regression_metrics_from_cfg(
    cfg: dict,
    y_true_2d: Sequence[Sequence[float]],
    y_pred_2d: Sequence[Sequence[float]],
) -> Dict[str, float]:
    m = cfg.get("metrics") or {}
    which = m.get("which") or [
        "mae", "mse", "rmse", "mape", "smape", "r2", "pearson",
        "bucket_acc", "bucket_mae", "bucket_rmse"
    ]
    bucket_on = m.get("bucket_on") or "long"
    edges = m.get("bucket_edges") or CLS_EDGES
    return compute_regression_metrics(
        y_true_2d=y_true_2d,
        y_pred_2d=y_pred_2d,
        which=which,
        bucket_on=bucket_on,
        bucket_edges=edges,
    )
