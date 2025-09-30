# scripts/cls/train.py
# Train A Classification/Regression Model On Squares Dataset
# Supports AMP, Validation, Checkpoints, And Rich Metrics

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
from tqdm.auto import tqdm
from torch.amp import GradScaler, autocast

from src.utils.echo import echo_line
from src.utils.determinism import set_seed, worker_init_fn, make_generator
from src.data.voc import paired_image_xml_list
from src.data.cls import SquaresCLSData
from src.transforms.cls import CLSTransform
from src.metrics.cls import compute_regression_metrics_from_cfg  # Centralized Metrics


# Config / FS
def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _prepare_out_dir(cfg: dict) -> Path:
    out_dir = Path(cfg.get("out_dir", "outputs/cls/train"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_dir


# Model
def _build_model(cfg: dict, device: torch.device):
    from src.models import cls_resnet

    mcfg = cfg.get("model") or {}
    out_dim = int(mcfg.get("out_dim", mcfg.get("num_classes", 2)))
    model_name = str(mcfg.get("model_name", mcfg.get("backbone", "resnet18")))
    pretrained = bool(mcfg.get("pretrained", True))
    final_act = mcfg.get("final_act", None)
    p_drop = float(mcfg.get("p_drop", 0.0))

    model = cls_resnet.build_resnet_classifier(
        model_name=model_name,
        out_dim=out_dim,
        pretrained=pretrained,
        final_act=final_act,
        p_drop=p_drop,
    )
    model.to(device)
    return model


# Data
def _build_loaders(cfg: dict, gen: torch.Generator):
    data = cfg.get("data", {}) or {}
    root = Path(data.get("root", "."))
    limit = data.get("limit_per_split")

    img_tr = root / "train"
    img_ev = root / "val"          # implicit validation split
    ann_dir = root / "annotations"

    tr = cfg.get("train", {}) or {}
    ev = cfg.get("eval", {}) or {}
    img_size = int((cfg.get("model", {}) or {}).get("input_size", tr.get("img_size", 224)))
    use_padding_canvas = bool(tr.get("use_padding_canvas", True))

    ds_tr = SquaresCLSData(
        paired_image_xml_list(img_tr, ann_dir, limit=limit),
        canvas=img_size,
        train=True,
        use_padding_canvas=use_padding_canvas,
        transforms=CLSTransform(size=img_size, train=True),
    )
    ds_ev = SquaresCLSData(
        paired_image_xml_list(img_ev, ann_dir, limit=limit),
        canvas=img_size,
        train=False,
        use_padding_canvas=use_padding_canvas,
        transforms=CLSTransform(size=img_size, train=False),
    )

    dl_tr = DataLoader(
        ds_tr,
        batch_size=int(tr.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(tr.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=gen,
    )
    dl_ev = DataLoader(
        ds_ev,
        batch_size=int(ev.get("batch_size", 64)),
        shuffle=False,
        num_workers=int(ev.get("num_workers", 4)),
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=gen,
    )
    return ds_tr, ds_ev, dl_tr, dl_ev


# Optimizer / Scheduler / Loss
def _build_optim_sched(model: torch.nn.Module, cfg: dict):
    tr = cfg.get("train", {}) or {}
    lr = float(tr.get("lr", 1e-3))
    wd = float(tr.get("weight_decay", 0.0))
    opt_name = (tr.get("optimizer", "adamw") or "adamw").lower()

    if opt_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    sc = tr.get("scheduler", {}) or {}
    if (sc.get("name") or "").lower() == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sc.get("step_size", 10)),
            gamma=float(sc.get("gamma", 0.1)),
        )
    else:
        scheduler = None
    return optimizer, scheduler


def _build_loss(cfg: dict):
    name = ((cfg.get("train", {}) or {}).get("loss", "mse") or "mse").lower()
    if name in {"mse", "l2"}:
        return torch.nn.MSELoss(), "mse"
    if name in {"l1", "mae"}:
        return torch.nn.L1Loss(), "l1"
    if name in {"smoothl1", "huber"}:
        beta = float(((cfg.get("train", {}) or {}).get("smoothl1_beta", 1.0)))
        return torch.nn.SmoothL1Loss(beta=beta), "smoothl1"
    return torch.nn.MSELoss(), "mse"


def _lr(optimizer: torch.optim.Optimizer) -> float:
    for g in optimizer.param_groups:
        if "lr" in g:
            return float(g["lr"])
    return 0.0


def _amp_ctx(enabled: bool, device: torch.device):
    if enabled and device.type == "cuda":
        try:
            return autocast(device_type="cuda", dtype=torch.float16)
        except TypeError:
            return autocast(enabled=True)
    return nullcontext()


# Train / Val
def train_one_epoch(model, optimizer, scaler, dl, device, loss_fn, log_every, epoch, epochs, max_norm: float | None):
    model.train()
    total, n = 0.0, 0
    bar = tqdm(dl, total=len(dl), desc=f"E{epoch:03d}/{epochs:03d}", leave=True)

    for i, (x, y) in enumerate(bar, 1):
        x = x.to(device, non_blocking=True)  # [N,3,H,W]
        y = y.to(device, non_blocking=True)  # [N,2]

        optimizer.zero_grad(set_to_none=True)
        with _amp_ctx(enabled=(scaler is not None), device=device):
            pred = model(x)                 # [N,2]
            loss = loss_fn(pred, y)

        if scaler is not None:
            scaler.scale(loss).backward()
            if max_norm is not None and max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if max_norm is not None and max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()

        total += float(loss.item()); n += 1
        if (i % max(1, log_every)) == 0 or (i == len(dl)):
            echo_line(
                "CLS_TRAIN",
                {"epoch": f"{epoch}/{epochs}", "iter": f"{i}/{len(dl)}", "lr": _lr(optimizer), "loss": total / n},
                order=["epoch", "iter", "lr", "loss"],
            )
    return total / max(1, n)


@torch.no_grad()
def evaluate(model, dl, device, out_dir: Path, cfg: dict, tag: str = "val") -> Dict[str, Any]:
    model.eval()
    y_true, y_pred = [], []

    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)        # [N,2]
        p = model(x)                               # [N,2]
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(p.detach().cpu().numpy())

    y_true = np.concatenate(y_true, axis=0) if y_true else np.zeros((0, 2), dtype=np.float32)
    y_pred = np.concatenate(y_pred, axis=0) if y_pred else np.zeros((0, 2), dtype=np.float32)

    stats = compute_regression_metrics_from_cfg(cfg, y_true, y_pred)

    echo_line("CLS_VAL", {"split": tag, "mae": float(stats.get("overall_mae", float("nan"))),
                           "rmse": float(stats.get("overall_rmse", float("nan")))},
              order=["split", "mae", "rmse"])

    # Persist Artifacts
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(stats, indent=2))
    with (out_dir / f"{tag}_preds.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["y_short", "y_long", "p_short", "p_long"])
        for (yt0, yt1), (yp0, yp1) in zip(y_true, y_pred):
            w.writerow([float(yt0), float(yt1), float(yp0), float(yp1)])

    return stats


# Monitor / Early-Best
def _pick_monitor_value(name: str, stats: dict, train_loss: float):
    key = (name or "").lower()
    if key in {"val_loss", "loss"}:
        return float(train_loss), "min"

    aliases = {
        "mae": "overall_mae",
        "mse": "overall_mse",
        "rmse": "overall_rmse",
        "mape": "overall_mape",
        "smape": "overall_smape",
        "r2": "overall_r2",
        "pearson": "overall_pearson",
        "acc": "bucket_acc_global",
        "bucket_acc": "bucket_acc_global",
    }
    canon = aliases.get(key, key)
    val = stats.get(canon, None)
    if val is None:
        return None, "min"

    direction = "max" if any(tok in canon for tok in ["r2", "pearson", "acc"]) else "min"
    return float(val), direction


def _bucket_names_from_cfg(cfg: dict) -> List[str]:
    m = (cfg.get("metrics") or {})
    edges = m.get("bucket_edges") or [0, 32, 64, 96, 128, 160, 99999]
    names = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges)-1)] + [f"{int(edges[-1])}+"]
    return names


# Main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    seed = int(cfg.get("seed", 1337))

    # Determinism: env for workers + global seeds + DataLoader generator
    os.environ["DATA_WORKER_SEED"] = str(seed)
    set_seed(seed)
    gen = make_generator(seed)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = _prepare_out_dir(cfg)

    ds_tr, ds_ev, dl_tr, dl_ev = _build_loaders(cfg, gen)

    model = _build_model(cfg, device)
    optimizer, scheduler = _build_optim_sched(model, cfg)
    loss_fn, loss_name = _build_loss(cfg)

    tr = cfg.get("train", {}) or {}
    ev = cfg.get("eval", {}) or {}
    epochs = int(tr.get("epochs", 50))
    log_every = int(tr.get("log_every", 100))
    grad_clip = tr.get("grad_clip", None)
    grad_clip = float(grad_clip) if grad_clip is not None else None

    use_amp = bool(tr.get("amp", True)) and (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp) if use_amp else None

    monitor_name = ((cfg.get("early_stopping") or {}).get("monitor") or "overall_rmse")

    # Logs
    with (out_dir / "train.log.jsonl").open("a", encoding="utf-8"):
        pass

    # Metrics.csv Header (Includes Per-Bucket MAE/RMSE Columns)
    bucket_names = _bucket_names_from_cfg(cfg)
    header = ["epoch", "lr", "train_loss", "overall_mae", "overall_rmse", "bucket_acc_global"]
    for bn in bucket_names:
        header.append(f"bucket_mae_{bn}")
        header.append(f"bucket_rmse_{bn}")
    header.append("time_sec")

    with (out_dir / "metrics.csv").open("w", newline="") as f:
        csv.writer(f).writerow(header)

    echo_line("CLS_SETUP", {
        "device": str(device),
        "cuda": torch.cuda.is_available(),
        "seed": seed,
        "epochs": epochs,
        "train_bs": int(tr.get("batch_size", 64)),
        "val_bs": int(ev.get("batch_size", 64)),
        "lr": f"{_lr(optimizer):.3e}",
        "loss": loss_name,
        "out_dir": str(out_dir),
    })

    best_val = float("inf")
    best_direction = None
    t0 = time.time()

    for ep in range(1, epochs + 1):
        echo_line("CLS_EPOCH", {"phase": "train", "epoch": f"{ep}/{epochs}"}, order=["phase", "epoch"])
        train_loss = train_one_epoch(model, optimizer, scaler, dl_tr, device, loss_fn, log_every, ep, epochs, grad_clip)
        echo_line("CLS_TRAIN_SUM", {"epoch": ep, "loss": train_loss, "lr": _lr(optimizer)}, order=["epoch", "loss", "lr"])

        if scheduler is not None:
            scheduler.step()

        val_stats: Dict[str, Any] = {}
        if bool(ev.get("run_during_train", True)):
            echo_line("CLS_EPOCH", {"phase": "val", "epoch": f"{ep}/{epochs}"}, order=["phase", "epoch"])
            val_stats = evaluate(model, dl_ev, device, out_dir=out_dir, cfg=cfg, tag="val")

        # Write Logs
        rec = {"epoch": ep, "lr": _lr(optimizer), "loss": float(train_loss), "val": val_stats, "time_sec": round(time.time() - t0, 3)}
        with (out_dir / "train.log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        # Metrics.csv Row
        row = [
            ep,
            f"{_lr(optimizer):.8f}",
            f"{train_loss:.6f}",
            f"{val_stats.get('overall_mae', float('nan')):.6f}" if val_stats else "",
            f"{val_stats.get('overall_rmse', float('nan')):.6f}" if val_stats else "",
            f"{val_stats.get('bucket_acc_global', float('nan')):.6f}" if val_stats else "",
        ]
        for bn in bucket_names:
            row.append(f"{val_stats.get(f'bucket_mae_{bn}', float('nan')):.6f}" if val_stats else "")
            row.append(f"{val_stats.get(f'bucket_rmse_{bn}', float('nan')):.6f}" if val_stats else "")
        row.append(f"{rec['time_sec']:.3f}")

        with (out_dir / "metrics.csv").open("a", newline="") as f:
            csv.writer(f).writerow(row)

        # Checkpoints
        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": ep}
        if scaler is not None:
            ckpt["scaler"] = scaler.state_dict()
        torch.save(ckpt, out_dir / "cls.last.ckpt")

        mon_val, direction = _pick_monitor_value(monitor_name, val_stats, train_loss)
        if mon_val is not None:
            if best_direction is None:
                best_direction = direction
                best_val = float("-inf") if direction == "max" else float("inf")
            better = (mon_val > best_val) if best_direction == "max" else (mon_val < best_val)
            if better:
                best_val = mon_val
                torch.save(ckpt, out_dir / "cls.best.ckpt")

    echo_line("CLS_DONE", {"best": best_val if np.isfinite(best_val) else None})


if __name__ == "__main__":
    main()
