# scripts/cls/train.py
import argparse
import time
import json
from pathlib import Path
from collections import defaultdict, deque
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
import yaml
from tqdm.auto import tqdm

# --- Modern AMP aliases (with fallback for older Torch) ---
try:
    # PyTorch 1.12+ / 2.x: unified AMP API
    from torch.amp import GradScaler as AmpGradScaler, autocast as amp_autocast
except Exception:  # fallback if torch.amp not available
    from torch.cuda.amp import GradScaler as AmpGradScaler
    def amp_autocast(device_type: str, **kwargs):  # type: ignore
        if device_type == "cuda":
            return torch.cuda.amp.autocast(**kwargs)
        if device_type == "cpu":
            return torch.autocast("cpu", **kwargs)
        return nullcontext()

from src.utils.echo import echo_line
from src.utils.determinism import set_seed
from src.data.voc import paired_image_xml_list
from src.data.cls import SquaresClassificationDatasetStream
from src.transforms.cls import CLSTransform
from src.models.cls_resnet import build_resnet_backbone as build_resnet_classifier
from src.metrics.cls import evaluate_size_regression  # MAE/MSE/MAPE on (short,long)


class Smoothed:
    def __init__(self, window: int = 200):
        self.win = deque(maxlen=window); self.total = 0.0; self.count = 0
    def update(self, x: float, n: int = 1):
        x = float(x); self.win.append(x); self.total += x * n; self.count += n
    @property
    def avg(self) -> float:
        return (sum(self.win) / max(1, len(self.win))) if self.win else 0.0
    @property
    def gavg(self) -> float:
        return self.total / max(1, self.count)


def _autocast_ctx(device: torch.device):
    """
    CUDA: torch.amp.autocast('cuda', dtype=float16)  (preferred)
    CPU : torch.amp.autocast('cpu', dtype=bfloat16) if available; else no-op
    """
    if device.type == "cuda":
        return amp_autocast("cuda", dtype=torch.float16)
    try:
        return amp_autocast("cpu", dtype=torch.bfloat16)
    except Exception:
        return nullcontext()


def build_dataloaders(cfg, split_train: str = "train", split_val: str = "val"):
    data_cfg  = cfg.get("data", {})
    root_str  = data_cfg.get("root", None)
    if not root_str:
        raise KeyError("configs YAML must define data.root pointing to your dataset root")
    root      = Path(root_str)
    limit     = data_cfg.get("limit_per_split")

    img_tr = root / split_train
    img_va = root / split_val
    ann_dir = root / "annotations"

    pairs_tr = paired_image_xml_list(img_tr, ann_dir, limit=limit)
    pairs_va = paired_image_xml_list(img_va, ann_dir, limit=limit)

    input_size = int(cfg.get("model", {}).get("input_size", 224))
    tfm_train = CLSTransform(size=input_size, train=True)
    tfm_val   = CLSTransform(size=input_size, train=False)

    ds_tr = SquaresClassificationDatasetStream(pairs_tr, transforms=tfm_train)
    ds_va = SquaresClassificationDatasetStream(pairs_va, transforms=tfm_val)

    train_cfg = cfg.get("train", {})
    eval_cfg  = cfg.get("eval", {})
    dl_tr = DataLoader(
        ds_tr,
        batch_size=train_cfg.get("batch_size", 64),
        shuffle=True,
        num_workers=train_cfg.get("num_workers", 2),
        pin_memory=True,
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=eval_cfg.get("batch_size", 128),
        shuffle=False,
        num_workers=eval_cfg.get("num_workers", 2),
        pin_memory=True,
    )
    return dl_tr, dl_va, (img_tr, img_va, ann_dir)


def train_one_epoch(model, optimizer, scaler, dl, device, log_every: int, epoch: int, epochs: int):
    model.train()
    meters = defaultdict(lambda: Smoothed(window=200))
    iters_total = len(dl)
    loss_fn = torch.nn.MSELoss(reduction="mean")

    bar = tqdm(dl, total=iters_total, desc=f"E{epoch:03d}/{epochs:03d}", leave=True)
    for i, (x, y) in enumerate(bar, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with _autocast_ctx(device):
            y_hat = model(x)
            loss = loss_fn(y_hat, y)

        # AMP or fallback
        if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        meters["loss"].update(loss.item(), n=x.size(0))
        bar.set_postfix({"lr": f"{optimizer.param_groups[0]['lr']:.1e}",
                         "loss": f"{meters['loss'].avg:.4f}"})

        if (i % log_every) == 0 or i == iters_total:
            tqdm.write(
                f"it {i:04d}/{iters_total:04d} | "
                f"lr={optimizer.param_groups[0]['lr']:.1e} | "
                f"loss={meters['loss'].avg:.4f}"
            )

    means = {k: v.gavg for k, v in meters.items()}
    means.setdefault("loss", 0.0)
    return means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--train_split", type=str, default="train")
    ap.add_argument("--val_split", type=str, default="val")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))
    set_seed(cfg.get("seed", 42))

    device = torch.device(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    default_out = Path("outputs") / "cls" / Path(args.config).stem
    out_dir = Path(cfg.get("out_dir", default_out))
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "epoch_log_cls.jsonl"

    dl_tr, dl_va, (img_tr, img_va, ann_dir) = build_dataloaders(cfg, args.train_split, args.val_split)

    # Build model (filter args to what builder accepts)
    mcfg = cfg.get("model", {})
    model_args = {k: mcfg[k] for k in ("model_name", "out_dim", "pretrained", "final_act", "p_drop") if k in mcfg}
    model = build_resnet_classifier(**model_args).to(device)

    optim_cfg = cfg.get("optim", {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=optim_cfg.get("lr", 1e-3),
        weight_decay=optim_cfg.get("weight_decay", 1e-4),
    )

    # Modern AMP scaler (works on CPU too, but does nothing there)
    try:
        scaler = AmpGradScaler(enabled=(device.type == "cuda"))
    except Exception:
        scaler = None  # fallback path if AMP not available

    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None:
            try:
                scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                print("[CLS] WARNING: couldn't load scaler state; continuing with fresh scaler.")
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"[CLS] Resumed from {args.resume} at epoch {start_epoch}")

    train_cfg = cfg.get("train", {})
    eval_cfg  = cfg.get("eval", {})
    epochs    = int(train_cfg.get("epochs", 10))
    log_every = int(train_cfg.get("log_every", 100))
    run_val   = bool(eval_cfg.get("run_during_train", True))

    echo_line(
        tag="SETUP",
        kv_pairs=dict(
            device=str(device),
            cuda=torch.cuda.is_available(),
            seed=cfg.get("seed", 42),
            epochs=epochs,
            train_bs=train_cfg.get("batch_size", 64),
            val_bs=eval_cfg.get("batch_size", 128),
            lr=optim_cfg.get("lr", 1e-3),
        ),
        order=["device", "cuda", "seed", "epochs", "train_bs", "val_bs", "lr"],
    )
    echo_line("DATA",
              {"train_img": str(img_tr), "val_img": str(img_va), "ann_dir": str(ann_dir)},
              order=["train_img", "val_img", "ann_dir"])

    best_metric = float("inf")
    for ep in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_means = train_one_epoch(model, optimizer, scaler, dl_tr, device, log_every, ep, epochs)

        val_res = {}
        if run_val:
            val_res = evaluate_size_regression(model, dl_va, device=device, out_dir=str(out_dir), tag=args.val_split)

        echo_line(
            tag=f"E{ep:03d}/{epochs:03d}",
            kv_pairs=dict(
                lr=optimizer.param_groups[0]["lr"],
                loss=train_means["loss"],
                MAE_short=val_res.get("MAE_short", 0.0),
                MAE_long=val_res.get("MAE_long", 0.0),
                MAPE_short=val_res.get("MAPE_short", 0.0),
                MAPE_long=val_res.get("MAPE_long", 0.0),
                N=val_res.get("N", 0),
                time_sec=round(time.time() - t0, 1),
            ),
            order=["lr", "loss", "MAE_short", "MAE_long", "MAPE_short", "MAPE_long", "N", "time_sec"],
        )

        with open(jsonl_path, "a", encoding="utf-8") as f:
            rec = {
                "epoch": ep,
                "lr": optimizer.param_groups[0]["lr"],
                **train_means,
                **{k: v for k, v in val_res.items()},
                "time_sec": round(time.time() - t0, 3),
            }
            f.write(json.dumps(rec) + "\n")

        ckpt = {"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": (scaler.state_dict() if scaler is not None else {}),
                "epoch": ep}
        torch.save(ckpt, out_dir / "cls_resnet.last.ckpt")
        score = val_res.get("MAE_long", float("inf"))
        if run_val and score < best_metric:
            best_metric = score
            torch.save(ckpt, out_dir / "cls_resnet.best.ckpt")


if __name__ == "__main__":
    main()
