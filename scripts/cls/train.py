# scripts/cls/train.py

# Train A Classification/Regression Model On Squares Dataset

from __future__ import annotations

# Standard Library
import os
import argparse, csv, json
from pathlib import Path
from contextlib import nullcontext
from typing import Dict, Any

# Third-Party
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm.auto import tqdm
import yaml

# Local Modules
from src.utils.echo import echo_line
from src.utils.determinism import set_seed, worker_init_fn, make_generator
from src.data.voc import paired_image_xml_list
from src.data.cls import SquaresCLSData
from src.transforms.cls import CLSTransform
from src.metrics.cls import compute_regression_metrics_from_cfg

# Load YAML Config
def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

# Prepare Output Directory
def _prepare_out_dir(cfg: dict) -> Path:
    out_dir = Path(cfg.get("out_dir", "outputs/cls/train"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out_dir

# Build Model
def _build_model(cfg: dict, device: torch.device):
    from src.models import cls_resnet
    mcfg = cfg.get("model") or {}
    out_dim = int(mcfg.get("out_dim", mcfg.get("num_classes", 2)))
    model_name = str(mcfg.get("model_name", mcfg.get("backbone", "resnet18")))
    pretrained = bool(mcfg.get("pretrained", True))
    final_act = mcfg.get("final_act", None)
    p_drop = float(mcfg.get("p_drop", 0.0))
    model = cls_resnet.build_resnet_classifier(
        model_name=model_name, out_dim=out_dim,
        pretrained=pretrained, final_act=final_act, p_drop=p_drop
    )
    model.to(device)
    return model

# Build DataLoaders
def _build_loaders(cfg: dict, gen: torch.Generator):
    data = cfg.get("data", {}) or {}
    root = Path(data.get("root", "."))
    limit = data.get("limit_per_split")
    img_tr, img_ev, ann_dir = root / "train", root / "val", root / "annotations"
    tr_cfg, ev_cfg = cfg.get("train", {}) or {}, cfg.get("eval", {}) or {}
    img_size = int((cfg.get("model") or {}).get("input_size", tr_cfg.get("img_size", 224)))
    use_pad = bool(tr_cfg.get("use_padding_canvas", True))

    ds_tr = SquaresCLSData(
        paired_image_xml_list(img_tr, ann_dir, limit=limit),
        canvas=img_size, train=True, use_padding_canvas=use_pad,
        transforms=CLSTransform(size=img_size, train=True)
    )
    ds_ev = SquaresCLSData(
        paired_image_xml_list(img_ev, ann_dir, limit=limit),
        canvas=img_size, train=False, use_padding_canvas=use_pad,
        transforms=CLSTransform(size=img_size, train=False)
    )

    dl_tr = DataLoader(ds_tr, batch_size=int(tr_cfg.get("batch_size", 64)), shuffle=True,
                       num_workers=int(tr_cfg.get("num_workers", 4)), pin_memory=True,
                       drop_last=False, worker_init_fn=worker_init_fn, generator=gen)
    dl_ev = DataLoader(ds_ev, batch_size=int(ev_cfg.get("batch_size", 64)), shuffle=False,
                       num_workers=int(ev_cfg.get("num_workers", 4)), pin_memory=True,
                       drop_last=False, worker_init_fn=worker_init_fn, generator=gen)
    return ds_tr, ds_ev, dl_tr, dl_ev

# Build Optimizer & Scheduler
def _build_optim_sched(model: torch.nn.Module, cfg: dict):
    tr = cfg.get("train", {}) or {}
    lr, wd = float(tr.get("lr", 1e-3)), float(tr.get("weight_decay", 0.0))
    opt_name = (tr.get("optimizer", "adamw") or "adamw").lower()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True) \
                if opt_name=="sgd" else torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sc_cfg = tr.get("scheduler", {}) or {}
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(sc_cfg.get("step_size",10)),
                                                gamma=float(sc_cfg.get("gamma",0.1))) \
                if (sc_cfg.get("name") or "").lower() == "step" else None
    return optimizer, scheduler

# Build Loss Function
def _build_loss(cfg: dict):
    name = ((cfg.get("train", {}) or {}).get("loss", "mse") or "mse").lower()
    if name in {"mse","l2"}: return torch.nn.MSELoss(), "mse"
    if name in {"l1","mae"}: return torch.nn.L1Loss(), "l1"
    if name in {"smoothl1","huber"}:
        beta = float(((cfg.get("train", {}) or {}).get("smoothl1_beta",1.0)))
        return torch.nn.SmoothL1Loss(beta=beta), "smoothl1"
    return torch.nn.MSELoss(), "mse"

# Get Current LR
def _lr(optimizer: torch.optim.Optimizer) -> float:
    for g in optimizer.param_groups:
        if "lr" in g: return float(g["lr"])
    return 0.0

# AMP Context
def _amp_ctx(enabled: bool, device: torch.device):
    return autocast(device_type="cuda", dtype=torch.float16) if enabled and device.type=="cuda" else nullcontext()

# Train One Epoch
def train_one_epoch(model, optimizer, scaler, dl, device, loss_fn, log_every, epoch, epochs, max_norm: float | None):
    model.train(); total, n = 0.0, 0
    bar = tqdm(dl, total=len(dl), desc=f"E{epoch:03d}/{epochs:03d}", leave=True)
    for i,(x,y) in enumerate(bar,1):
        x,y = x.to(device,non_blocking=True), y.to(device,non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _amp_ctx(enabled=(scaler is not None), device=device):
            pred = model(x); loss = loss_fn(pred,y)
        if scaler:
            scaler.scale(loss).backward()
            if max_norm: scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            if max_norm: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
        total += float(loss.item()); n+=1
        if (i % max(1,log_every))==0 or i==len(dl):
            echo_line("CLS_TRAIN", {"epoch":f"{epoch}/{epochs}", "iter":f"{i}/{len(dl)}", "lr":_lr(optimizer), "loss":total/n},
                      order=["epoch","iter","lr","loss"])
    return total/max(1,n)

# Evaluate Model
@torch.inference_mode()
def evaluate(model, dl, device, out_dir: Path, cfg: dict, tag="val") -> Dict[str,Any]:
    model.eval(); y_true,y_pred=[],[]
    for x,y in dl: x,y=x.to(device),y.to(device); p=model(x); y_true.append(y.cpu().numpy()); y_pred.append(p.cpu().numpy())
    y_true = np.concatenate(y_true,0) if y_true else np.zeros((0,2),np.float32)
    y_pred = np.concatenate(y_pred,0) if y_pred else np.zeros((0,2),np.float32)
    stats = compute_regression_metrics_from_cfg(cfg, y_true, y_pred)
    echo_line("CLS_VAL", {"split":tag,"mae":float(stats.get("overall_mae",float("nan"))),
                          "rmse":float(stats.get("overall_rmse",float("nan")))} ,
              order=["split","mae","rmse"])
    (out_dir/f"{tag}_metrics.json").write_text(json.dumps(stats,indent=2))
    with (out_dir/f"{tag}_preds.csv").open("w",newline="") as f:
        w=csv.writer(f); w.writerow(["y_short","y_long","p_short","p_long"])
        for (yt0,yt1),(yp0,yp1) in zip(y_true,y_pred): w.writerow([float(yt0),float(yt1),float(yp0),float(yp1)])
    return stats

# Pick Monitor Value
def _pick_monitor_value(name: str, stats: dict, train_loss: float):
    key = (name or "").lower()
    if key in {"val_loss","loss"}: return float(train_loss),"min"
    aliases = {"mae":"overall_mae","mse":"overall_mse","rmse":"overall_rmse","mape":"overall_mape",
               "smape":"overall_smape","r2":"overall_r2","pearson":"overall_pearson",
               "acc":"bucket_acc_global","bucket_acc":"bucket_acc_global"}
    canon = aliases.get(key,key)
    val = stats.get(canon,None)
    if val is None: return None,"min"
    direction = "max" if any(tok in canon for tok in ["r2","pearson","acc"]) else "min"
    return float(val),direction

# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/cls/test.yaml", help="YAML config file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism")
    args = parser.parse_args()

    cfg = _load_cfg(args.cfg)
    out_dir = _prepare_out_dir(cfg)

    # Set Determinism From Config
    det_flag = bool((cfg.get("train", {}) or {}).get("deterministic", True))
    if det_flag:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    # Set Random Seed
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed); gen = make_generator(seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    echo_line("INFO", {"device": str(device), "seed": seed, "deterministic": det_flag})

    # Model / Optimizer / Scheduler / Loss
    model = _build_model(cfg, device)
    optimizer, scheduler = _build_optim_sched(model, cfg)
    loss_fn, loss_name = _build_loss(cfg)

    # DataLoaders
    ds_tr, ds_ev, dl_tr, dl_ev = _build_loaders(cfg, gen)

    # AMP / Grad Norm
    scaler = GradScaler() if ((cfg.get("train", {}) or {}).get("amp", False) and device.type=="cuda") else None
    max_norm = float((cfg.get("train", {}) or {}).get("clip_grad_norm", 0.0)) or None

    # Training Loop
    epochs = int((cfg.get("train", {}) or {}).get("epochs", 10))
    log_every = int((cfg.get("train", {}) or {}).get("log_every", 10))
    best_val = float("inf"); best_file = out_dir / "best_model.pt"
    last_file = out_dir / "last_model.pt"  # NEW: track last model

    for epoch in range(1, epochs+1):
        train_loss = train_one_epoch(model, optimizer, scaler, dl_tr, device, loss_fn, log_every, epoch, epochs, max_norm)
        val_stats = evaluate(model, dl_ev, device, out_dir, cfg)
        monitor_val, direction = _pick_monitor_value(cfg.get("monitor","val_loss"), val_stats, train_loss)

        # Save Best Model
        if monitor_val is not None and ((direction=="min" and monitor_val < best_val) or (direction=="max" and monitor_val > best_val)):
            best_val = monitor_val
            torch.save(model.state_dict(), best_file)
            echo_line("INFO", {"epoch": epoch, "best_val": best_val, "saved": str(best_file)})

        # Save last Epoch Model
        torch.save(model.state_dict(), last_file)
        echo_line("INFO", {"epoch": epoch, "saved_last_model": str(last_file)})

        # Scheduler Step
        if scheduler: scheduler.step()

    echo_line("INFO", {"msg": "Training Completed", "best_val": best_val, "out_dir": str(out_dir)})

if __name__ == "__main__":
    main()
