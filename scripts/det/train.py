# scripts/det/train.py
import argparse
from pathlib import Path
import time
import json
from collections import defaultdict, deque
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
import yaml
from tqdm.auto import tqdm

from src.utils.echo import echo_line
from src.utils.determinism import set_seed
from src.data.voc import paired_image_xml_list
from src.data.det import SquaresDetectionDataset, collate_fn
from src.transforms.det import Compose, ToTensor, ClampBoxes, RandomHorizontalFlip, AnisotropicScale
from src.metrics.det import evaluate_ap_by_size as eval_detector
from src.models.det_fasterrcnn import build_fasterrcnn, build_fasterrcnn_custom

# AMP
try:
    from torch.amp import GradScaler as AmpGradScaler, autocast as amp_autocast
except Exception:
    from torch.cuda.amp import GradScaler as AmpGradScaler # type: ignore

    def amp_autocast(device_type: str, **kwargs): # type: ignore
        if device_type == "cuda":
            return torch.cuda.amp.autocast(**kwargs)
        if device_type == "cpu":
            return torch.autocast("cpu", **kwargs)
        return nullcontext()

def _autocast_ctx(device: torch.device):
    if device.type == "cuda":
        return amp_autocast("cuda", dtype=torch.float16)
    try:
        return amp_autocast("cpu", dtype=torch.bfloat16)
    except Exception:
        return nullcontext()

def _to_float(x, default):
    try:
        return float(x)
    except Exception:
        return float(default)

def _to_int(x, default):
    try:
        return int(x)
    except Exception:
        return int(default)

def _to_bool(x, default=False):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.strip().lower() in ("1", "true", "yes", "y", "on")
    return bool(default)

def implicit_det_paths(root: Path, split_train: str, split_val: str):
    root = Path(root)
    img_train = root / split_train
    img_val   = root / split_val
    ann_root  = root / "annotations"
    return img_train, img_val, ann_root

class Smoothed:
    def __init__(self, window: int = 200):
        self.win = deque(maxlen=window); self.total = 0.0; self.count = 0
    def update(self, x: float, n: int = 1):
        x = float(x); self.win.append(x); self.total += x * n; self.count += n
    @property
    def avg(self) -> float:  return (sum(self.win) / max(1, len(self.win))) if self.win else 0.0
    @property
    def gavg(self) -> float: return self.total / max(1, self.count)

# Target validator (guards against bad boxes)
def _is_valid_target(t):
    boxes = t.get("boxes", None)
    if boxes is None or boxes.numel() == 0:
        return False
    if not torch.isfinite(boxes).all():
        return False
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    if (w <= 0).any() or (h <= 0).any():
        return False
    return True

def build_dataloaders(cfg, split_train="train", split_val="val"):
    root = Path(cfg["data"]["root"])
    limit = cfg["data"].get("limit_per_split")

    img_train, img_val, ann_root = implicit_det_paths(root, split_train, split_val)
    pairs_train = paired_image_xml_list(img_train, ann_root, limit=limit)
    pairs_val   = paired_image_xml_list(img_val,   ann_root, limit=limit)

    # Train: mild rectangle-friendly aug (flip + anisotropic scaling)
    tfm_train = Compose([
        ToTensor(),
        RandomHorizontalFlip(p=0.5),
        AnisotropicScale(scale_range=(0.8, 1.25)),
        ClampBoxes(),
    ])
    tfm_val = Compose([ToTensor(), ClampBoxes()])

    ds_train = SquaresDetectionDataset(pairs_train, transforms=tfm_train)
    ds_val   = SquaresDetectionDataset(pairs_val,   transforms=tfm_val)

    dl_train = DataLoader(
        ds_train,
        batch_size=_to_int(cfg["train"].get("batch_size", 2), 2),
        shuffle=True,
        num_workers=_to_int(cfg["train"].get("num_workers", 2), 2),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=_to_int(cfg["eval"].get("batch_size", 1), 1),
        shuffle=False,
        num_workers=_to_int(cfg["eval"].get("num_workers", 2), 2),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    return dl_train, dl_val, (img_train, img_val, ann_root)

def build_model(cfg, device):
    if cfg["model"].get("custom", False):
        model = build_fasterrcnn_custom(cfg["model"])
    else:
        model = build_fasterrcnn(num_classes=cfg["model"]["num_classes"])
    return model.to(device)

WARMUP_ITERS = 100         # first N iters run in FP32 to avoid AMP jitters
CLIP_MAX_NORM = 0.0        # set >0 (e.g., 5.0) to enable grad clipping

def train_one_epoch(model, optimizer, scaler, dl, device, log_every:int, epoch:int, epochs:int):
    model.train()
    meters = defaultdict(lambda: Smoothed(window=200))
    iters_total = len(dl)

    bar = tqdm(dl, total=iters_total, desc=f"E{epoch:03d}/{epochs:03d}", leave=True)
    for i, (imgs, targets) in enumerate(bar, start=1):
        imgs = [im.to(device) for im in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # filter invalid items in batch
        good = [j for j, t in enumerate(targets) if _is_valid_target(t)]
        if len(good) != len(targets):
            imgs = [imgs[j] for j in good]
            targets = [targets[j] for j in good]
            if len(imgs) == 0:
                continue

        # AMP warm-start: FP32 for first WARMUP_ITERS global iterations
        global_iter = (epoch - 1) * iters_total + i
        use_amp = (scaler is not None and getattr(scaler, "is_enabled", lambda: False)())
        ctx = _autocast_ctx(device) if (use_amp and global_iter > WARMUP_ITERS) else nullcontext()

        with ctx:
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())

        # NaN/Inf guard
        if not torch.isfinite(loss):
            tqdm.write(f"[WARN] non-finite loss at it {i}: {loss.item() if loss.numel()==1 else 'tensor'}; skipping step")
            optimizer.zero_grad(set_to_none=True)
            continue

        if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
            scaler.scale(loss).backward()
            if CLIP_MAX_NORM and CLIP_MAX_NORM > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_MAX_NORM)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if CLIP_MAX_NORM and CLIP_MAX_NORM > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_MAX_NORM)
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        batch_sz = len(imgs)
        meters["loss"].update(loss.item(), n=batch_sz)
        for k in ("loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"):
            if k in loss_dict and torch.isfinite(loss_dict[k]):
                meters[k].update(loss_dict[k].item(), n=batch_sz)

        bar.set_postfix({
            "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
            "loss": f"{meters['loss'].avg:.4f}",
        })

        if (i % log_every) == 0 or i == iters_total:
            tqdm.write(
                f"it {i:04d}/{iters_total:04d} | "
                f"lr={optimizer.param_groups[0]['lr']:.1e} | "
                f"loss={meters['loss'].avg:.4f} | "
                f"loss_classifier={meters['loss_classifier'].avg:.4f} | "
                f"loss_box_reg={meters['loss_box_reg'].avg:.4f} | "
                f"loss_objectness={meters['loss_objectness'].avg:.4f} | "
                f"loss_rpn_box_reg={meters['loss_rpn_box_reg'].avg:.4f}"
            )

    means = {k: v.gavg for k, v in meters.items()}
    for k in ("loss","loss_classifier","loss_box_reg","loss_objectness","loss_rpn_box_reg"):
        means.setdefault(k, 0.0)
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

    default_out = Path("outputs") / "det" / Path(args.config).stem
    out_dir = Path(cfg.get("out_dir", default_out))
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "epoch_log.jsonl"

    dl_train, dl_val, (img_tr, img_va, ann_root) = build_dataloaders(cfg, args.train_split, args.val_split)
    model = build_model(cfg, device)

    optim_cfg = cfg.get("optim", {})
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=_to_float(optim_cfg.get("lr", 1e-3), 1e-3),
        momentum=_to_float(optim_cfg.get("momentum", 0.9), 0.9),
        weight_decay=_to_float(optim_cfg.get("weight_decay", 1e-4), 1e-4),
        nesterov=_to_bool(optim_cfg.get("nesterov", False), False),
    )

    # AMP scaler (portable across torch versions)
    try:
        scaler = AmpGradScaler(enabled=(device.type == "cuda"))
    except Exception:
        scaler = None  # fallback if AMP not available

    # Info banner
    echo_line(
        tag="SETUP",
        kv_pairs=dict(
            device=str(device),
            cuda=torch.cuda.is_available(),
            seed=cfg.get("seed", 42),
            epochs=int(cfg["train"]["epochs"]),
            train_bs=cfg["train"].get("batch_size", 2),
            val_bs=cfg["eval"].get("batch_size", 1),
            lr=_to_float(optim_cfg.get("lr", 1e-3), 1e-3),
        ),
        order=["device","cuda","seed","epochs","train_bs","val_bs","lr"],
    )
    echo_line("DATA", {
        "train_img": str(img_tr),
        "val_img": str(img_va),
        "ann_dir": str(ann_root),
    }, order=["train_img","val_img","ann_dir"])

    # Train
    epochs    = int(cfg["train"]["epochs"])
    log_every = int(cfg["train"].get("log_every", 100))
    run_val   = bool(cfg["eval"].get("run_during_train", True))

    best_ap = -1.0
    for epoch in range(1, epochs + 1):  # fixed start_epoch
        t0 = time.time()
        train_means = train_one_epoch(model, optimizer, scaler, dl_train, device, log_every, epoch, epochs)

        val_ap50 = None
        if run_val:
            det_metrics, _, _ = eval_detector(model, dl_val, device=device, out_dir=str(out_dir), tag=args.val_split)
            val_ap50 = float(det_metrics.get("ap50_global", 0.0))

        echo_line(
            tag=f"E{epoch:03d}/{epochs:03d}",
            kv_pairs=dict(
                lr=optimizer.param_groups[0]["lr"],
                loss=train_means["loss"],
                loss_classifier=train_means["loss_classifier"],
                loss_box_reg=train_means["loss_box_reg"],
                loss_objectness=train_means["loss_objectness"],
                loss_rpn_box_reg=train_means["loss_rpn_box_reg"],
                val_ap50=(0.0 if val_ap50 is None else val_ap50),
                time_sec=round(time.time() - t0, 1),
            ),
            order=["lr","loss","loss_classifier","loss_box_reg","loss_objectness","loss_rpn_box_reg","val_ap50","time_sec"],
        )

        with open(jsonl_path, "a", encoding="utf-8") as f:
            rec = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                **train_means,
                "val_ap50": (0.0 if val_ap50 is None else val_ap50),
                "time_sec": round(time.time() - t0, 3),
            }
            f.write(json.dumps(rec) + "\n")

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": (scaler.state_dict() if scaler is not None else {}),
            "epoch": epoch,
        }
        torch.save(ckpt, out_dir / "det_fasterrcnn.last.ckpt")
        if val_ap50 is not None and val_ap50 > best_ap:
            best_ap = val_ap50
            torch.save(ckpt, out_dir / "det_fasterrcnn.best.ckpt")

if __name__ == "__main__":
    main()
