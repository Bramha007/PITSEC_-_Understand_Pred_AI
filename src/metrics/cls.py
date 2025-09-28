# src/metrics/cls.py
import json, os
from typing import Tuple
import numpy as np
import torch


def summarize_regression(
    y_true, y_pred, out_dir: str = "outputs", tag: str = "val",
    names: Tuple[str, str] = ("short", "long")
):
    """
    y_true, y_pred: array-like [N,2] in the SAME space (pixels or log-pixels).
    Returns dict of MAE/MSE/MAPE per dimension and writes JSON to out_dir.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    assert y_true.shape == y_pred.shape and y_true.ndim == 2 and y_true.shape[1] == 2

    err = y_pred - y_true
    mae = np.mean(np.abs(err), axis=0)
    mse = np.mean(err**2, axis=0)
    mape = np.mean(np.abs(err) / (np.abs(y_true) + 1e-6), axis=0)

    result = {
        f"MAE_{names[0]}": float(mae[0]),
        f"MAE_{names[1]}": float(mae[1]),
        f"MSE_{names[0]}": float(mse[0]),
        f"MSE_{names[1]}": float(mse[1]),
        f"MAPE_{names[0]}": float(mape[0]),
        f"MAPE_{names[1]}": float(mape[1]),
        "N": int(y_true.shape[0]),
    }

    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, f"classifier_regression_metrics_{tag}.json")
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)

    return result, out_json


@torch.no_grad()
def evaluate_size_regression(
    model,
    data_loader,
    device="cpu",
    out_dir: str = "outputs",
    tag: str = "val",
    names: Tuple[str, str] = ("short", "long"),
):
    """
    Run the model over a dataloader of (img, y) pairs and compute MAE/MSE/MAPE
    for (short,long) sides. Saves a JSON summary and returns the metrics dict.
    """
    model.eval()

    ys_true = []
    ys_pred = []

    for batch in data_loader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        y_hat = model(x)

        ys_true.append(y.detach().cpu().numpy())
        ys_pred.append(y_hat.detach().cpu().numpy())

    if len(ys_true) == 0:
        res = {
            f"MAE_{names[0]}": 0.0,
            f"MAE_{names[1]}": 0.0,
            f"MSE_{names[0]}": 0.0,
            f"MSE_{names[1]}": 0.0,
            f"MAPE_{names[0]}": 0.0,
            f"MAPE_{names[1]}": 0.0,
            "N": 0,
        }
        # Write an empty file for consistency
        os.makedirs(out_dir, exist_ok=True)
        out_json = os.path.join(out_dir, f"classifier_regression_metrics_{tag}.json")
        with open(out_json, "w") as f:
            json.dump(res, f, indent=2)
        return res

    y_true = np.concatenate(ys_true, axis=0)
    y_pred = np.concatenate(ys_pred, axis=0)

    res, _ = summarize_regression(y_true, y_pred, out_dir=out_dir, tag=tag, names=names)
    return res
