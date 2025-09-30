# scripts/det/train.py
# Train a Faster R-CNN detection model on the Squares dataset
# Deterministic training; supports AMP, validation, checkpoints, and AP/AR metrics

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
import yaml
from tqdm.auto import tqdm
from torch.amp import GradScaler, autocast

from src.utils.echo import echo_line
from src.utils.determinism import set_seed, worker_init_fn, make_generator
from src.data.voc import paired_image_xml_list
from src.data.det import SquaresDetectionDataset, collate_fn
from src.transforms.det import Compose, ToTensor, ClampBoxes, RandomHorizontalFlip
# optional extras: from src.transforms.det_experimental import AnisotropicScale, RandomResize
from src.models.det_fasterrcnn import build_fasterrcnn
from src.metrics.det import evaluate_ap_by_size


class Smoothed:
    def __init__(self, window: int = 200):
        self.buf = deque(maxlen=window)
        self.total = 0.0
        self.count = 0

    def update(self, x: float):
        x = float(x)
        self.buf.append(x)
        self.total += x
        self.count += 1

    @property
    def avg(self) -> float:
        return 0.0 if not self.buf else sum(self.buf) / len(self.buf)

    @property
    def global_avg(self) -> float:
        return 0.0 if self.count == 0 else self.total / self.count


def _load_cfg(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_loaders(cfg: dict, gen: torch.Generator):
    data = cfg.get("data", {}) or {}
    root = Path(data.get("root", "."))           # expects train/, val/, annotations/
    limit = data.get("limit_per_split", None)

    img_tr = root / "train"
    img_ev = root / "val"
    ann    = root / "annotations"

    tr_pairs = paired_image_xml_list(img_tr, ann, limit=limit)
    ev_pairs = paired_image_xml_list(img_ev, ann, limit=limit)

    tfm_tr = Compose([ToTensor(), RandomHorizontalFlip(0.5), ClampBoxes()])
    tfm_ev = Compose([ToTensor(), ClampBoxes()])

    ds_tr = SquaresDetectionDataset(tr_pairs, transforms=tfm_tr)
    ds_ev = SquaresDetectionDataset(ev_pairs, transforms=tfm_ev)

    tr = cfg.get("train", {}) or {}
    ev = cfg.get("eval", {}) or {}

    dl_tr = DataLoader(
        ds_tr,
        batch_size=int(tr.get("batch_size", 2)),
        shuffle=True,
        num_workers=int(tr.get("num_workers", 2)),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=gen,
    )
    dl_ev = DataLoader(
        ds_ev,
        batch_size=int(ev.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(ev.get("num_workers", 2)),
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False,
        worker_init_fn=worker_init_fn,
        generator=gen,
    )
    return ds_tr, ds_ev, dl_tr, dl_ev


def _build_model(cfg: dict, device: torch.device):
    m = cfg.get("model", {}) or {}
    model = build_fasterrcnn(
        num_classes=int(m.get("num_classes", 2)),
        anchor_sizes=m.get("anchor_sizes", None),
        anchor_aspect_ratios=m.get("anchor_aspect_ratios", None),
        weights=m.get("weights", None),
        trainable_backbone_layers=m.get("trainable_backbone_layers", None),
        local_backbone_weights=m.get("local_backbone_weights", None),
    )
    model.to(device)
    return model


def _build_optim_sched(model: torch.nn.Module, cfg: dict):
    tr = cfg.get("train", {}) or {}
    lr = float(tr.get("lr", 1e-4))
    wd = float(tr.get("weight_decay", 1e-4))
    name = (tr.get("optimizer", "adamw") or "adamw").lower()

    if name == "sgd":
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


def _det_header_from_metrics(m: Dict[str, float]) -> List[str]:
    header = ["epoch", "lr", "loss", "ap50_global", "ap_global", "ar_global"]
    for prefix in ["ap50_", "ap_", "ar_"]:
        keys = sorted([k for k in m.keys() if k.startswith(prefix) and not k.endswith("_global")])
        header.extend(keys)
    header.append("time_sec")
    return header


def _det_row_from_metrics(header: List[str], epoch: int, lr: float, loss: float, m: Dict[str, float], time_sec: float) -> List[str]:
    row = []
    for col in header:
        if col == "epoch":
            row.append(epoch)
        elif col == "lr":
            row.append(f"{lr:.8f}")
        elif col == "loss":
            row.append(f"{loss:.6f}")
        elif col == "time_sec":
            row.append(f"{time_sec:.3f}")
        else:
            row.append(f"{float(m.get(col, float('nan'))):.6f}")
    return row


def train_one_epoch(model, optimizer, scaler, dl, device, log_every: int, epoch: int, epochs: int, max_norm: float | None):
    model.train()
    meters = defaultdict(lambda: Smoothed(window=200))
    bar = tqdm(dl, total=len(dl), desc=f"E{epoch:03d}/{epochs:03d}", leave=True)

    for i, (imgs, targets) in enumerate(bar, start=1):
        imgs = [im.to(device, non_blocking=True) for im in imgs]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]

        with _amp_ctx(enabled=(scaler is not None), device=device):
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

        optimizer.zero_grad(set_to_none=True)
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

        meters["loss"].update(float(loss.detach().item()))
        for k, v in loss_dict.items():
            meters[k].update(float(v.detach().item()))

        if (i % max(1, log_every)) == 0 or (i == len(dl)):
            echo_line(
                "DET_TRAIN",
                {
                    "epoch": f"{epoch}/{epochs}",
                    "iter": f"{i}/{len(dl)}",
                    "lr": _lr(optimizer),
                    "loss": meters["loss"].avg,
                    **{f"loss_{k}": meters[k].avg for k in sorted(loss_dict.keys())},
                },
                order=["epoch", "iter", "lr", "loss"],
            )

    return meters["loss"].global_avg


@torch.no_grad()
def evaluate(model, dl, device, out_dir: Path, tag: str = "val") -> dict:
    metrics, _, _ = evaluate_ap_by_size(model, dl, device, out_dir=str(out_dir), tag=tag)
    echo_line(
        "DET_VAL",
        {
            "split": tag,
            "ap50": float(metrics.get("ap50_global", 0.0)),
            "ap": float(metrics.get("ap_global", 0.0)),
            "ar": float(metrics.get("ar_global", 0.0)),
        },
        order=["split", "ap50", "ap", "ar"],
    )
    (out_dir / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = _load_cfg(args.config)
    seed = int(cfg.get("seed", 42))

    os.environ["DATA_WORKER_SEED"] = str(seed)
    set_seed(seed)
    gen = make_generator(seed)

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    out_dir = Path(cfg.get("out_dir", "outputs/det/train"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.used.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    ds_tr, ds_ev, dl_tr, dl_ev = _build_loaders(cfg, gen)

    model = _build_model(cfg, device)
    optimizer, scheduler = _build_optim_sched(model, cfg)

    tr = cfg.get("train", {}) or {}
    ev = cfg.get("eval", {}) or {}
    epochs = int(tr.get("epochs", 20))
    log_every = int(tr.get("log_every", 100))
    grad_clip = tr.get("grad_clip", None)
    grad_clip = float(grad_clip) if grad_clip is not None else None

    use_amp = bool(tr.get("amp", True)) and (device.type == "cuda")
    scaler = GradScaler(enabled=use_amp) if use_amp else None

    monitor = (cfg.get("early_stopping", {}) or {}).get("monitor", "ap50_global")
    best_val = float("-inf")
    t0 = time.time()

    metrics_csv_path = out_dir / "metrics.csv"
    header: List[str] | None = None

    for ep in range(1, epochs + 1):
        echo_line("DET_EPOCH", {"phase": "train", "epoch": f"{ep}/{epochs}"}, order=["phase", "epoch"])
        train_loss = train_one_epoch(model, optimizer, scaler, dl_tr, device, log_every, ep, epochs, grad_clip)
        echo_line("DET_TRAIN_SUM", {"epoch": ep, "loss": train_loss, "lr": _lr(optimizer)}, order=["epoch", "loss", "lr"])

        if scheduler is not None:
            scheduler.step()

        do_val = bool(ev.get("run_during_train", True))
        val_stats = {}
        if do_val:
            echo_line("DET_EPOCH", {"phase": "val", "epoch": f"{ep}/{epochs}"}, order=["phase", "epoch"])
            val_stats = evaluate(model, dl_ev, device, out_dir=out_dir, tag="val")

        rec = {
            "epoch": ep,
            "loss": float(train_loss),
            "lr": _lr(optimizer),
            "val": {k: float(v) if isinstance(v, (int, float)) else v for k, v in val_stats.items()},
            "time_sec": round(time.time() - t0, 3),
        }
        with (out_dir / "train.log.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        if do_val:
            if header is None:
                header = _det_header_from_metrics(val_stats)
                with metrics_csv_path.open("w", newline="") as fcsv:
                    import csv as _csv
                    _csv.writer(fcsv).writerow(header)
            row = _det_row_from_metrics(header, ep, _lr(optimizer), train_loss, val_stats, rec["time_sec"])
            with metrics_csv_path.open("a", newline="") as fcsv:
                import csv as _csv
                _csv.writer(fcsv).writerow(row)

        ckpt = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": ep}
        if scaler is not None:
            ckpt["scaler"] = scaler.state_dict()

        torch.save(ckpt, out_dir / "det_fasterrcnn.last.ckpt")
        if do_val and (monitor in val_stats):
            cur = float(val_stats[monitor])
            if cur > best_val:
                best_val = cur
                torch.save(ckpt, out_dir / "det_fasterrcnn.best.ckpt")

    echo_line("DET_DONE", {"best": best_val if best_val != float("-inf") else None})


if __name__ == "__main__":
    main()
