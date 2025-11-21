import os, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataio.voc_parser import paired_image_xml_list
from src.dataio.det_dataset import SquaresDetectionDataset, collate_fn
from src.dataio.det_transforms import Compose, ToTensor, RandomHorizontalFlip
from src.models.fasterrcnn_cpu import build_fasterrcnn_cpu
from src.utils.metrics_det import evaluate_ap_by_size
from src.dataio.split_utils import subsample_pairs

# ----- Config -----
DATA_ROOT   = r"E:\WPT-Project\Data\sized_squares_filled"   # adjust folder name

IMG_DIR_TRAIN = fr"{DATA_ROOT}\train"
IMG_DIR_VAL   = fr"{DATA_ROOT}\val"
IMG_DIR_TEST  = fr"{DATA_ROOT}\test"
IMG_DIR_TEST_RECT = fr'E:\WPT-Project\Data\sized_rectangles_filled\test'  # (not used here)

# ALL XMLs live here (shared)
XML_DIR_ALL   = fr"{DATA_ROOT}\annotations"
XML_DIR_ALL_RECT   = fr'E:\WPT-Project\Data\sized_rectangles_filled\annotations'  # (not used here)


OUTPUT_DIR = "outputs"
SAVE_CKPT  = os.path.join(OUTPUT_DIR, "fasterrcnn_squares_cpu.pt")

EPOCHS      = 2
BATCH_SIZE  = 1      # CPU
LR          = 0.003
SEED        = 42
NUM_WORKERS = 0

# Fractions (grow these over time: 0.05 → 0.1 → 0.25 → 0.5 → 1.0)
F_TRAIN = 0.0005   # begin with 5% of train
F_VAL   = 0.005
F_TEST  = 0.005
MAX_TRAIN_ITEMS = None  # e.g., 50000 to hard-cap

# ------------------

def make_loader(pairs, train=False, bs=1):
    if len(pairs) == 0:
        return None
    ds = SquaresDetectionDataset(
        pairs,
        transforms=Compose([ToTensor(), RandomHorizontalFlip(0.5)]) if train else Compose([ToTensor()])
    )
    return DataLoader(ds, batch_size=bs, shuffle=train and len(pairs) > 1,
                      num_workers=NUM_WORKERS, collate_fn=collate_fn)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    # Pair each split against the single annotations folder
    train_pairs_all = paired_image_xml_list(IMG_DIR_TRAIN, XML_DIR_ALL)
    val_pairs_all   = paired_image_xml_list(IMG_DIR_VAL,   XML_DIR_ALL)
    test_pairs_all  = paired_image_xml_list(IMG_DIR_TEST_RECT,  XML_DIR_ALL_RECT)

    # Subsample deterministically
    train_pairs = subsample_pairs(train_pairs_all, F_TRAIN, seed=SEED, max_items=MAX_TRAIN_ITEMS)
    val_pairs   = subsample_pairs(val_pairs_all,   F_VAL,   seed=SEED)
    test_pairs  = subsample_pairs(test_pairs_all,  F_TEST,  seed=SEED)

    print(f"Train pairs: {len(train_pairs)} / {len(train_pairs_all)} "
          f"(frac={F_TRAIN})")
    print(f"Val   pairs: {len(val_pairs)} / {len(val_pairs_all)} "
          f"(frac={F_VAL})")
    print(f"Test  pairs: {len(test_pairs)} / {len(test_pairs_all)} "
          f"(frac={F_TEST})")

    train_loader = make_loader(train_pairs, train=True,  bs=BATCH_SIZE)
    val_loader   = make_loader(val_pairs,   train=False, bs=BATCH_SIZE)
    test_loader  = make_loader(test_pairs,  train=False, bs=BATCH_SIZE)

    device = torch.device("cpu")
    model = build_fasterrcnn_cpu(num_classes=2).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, EPOCHS//2), gamma=0.1)

    best_val_ap = -1.0
    for ep in range(EPOCHS):
        model.train()
        losses = []
        t0 = time.time()
        for imgs, tgts in tqdm(train_loader, desc=f"Epoch {ep+1}/{EPOCHS}"):
            imgs = [img.to(device) for img in imgs]
            tgts = [{k:v.to(device) for k,v in t.items()} for t in tgts]
            loss_dict = model(imgs, tgts)
            loss = sum(loss_dict.values())
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))
        sch.step()

        # quick val AP
        val_js, _, _ = evaluate_ap_by_size(model, val_loader, device=device, out_dir=OUTPUT_DIR, tag=f"val_ep{ep+1}")
        val_ap = val_js["ap50_global"]
        print(f"Epoch {ep+1}: loss={np.mean(losses):.4f} | val AP@0.5={val_ap:.3f} | {time.time()-t0:.1f}s")
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            torch.save(model.state_dict(), SAVE_CKPT)
            print(f"  ✓ saved best → {SAVE_CKPT}")

    # Final eval on full val/test subsets
    model.load_state_dict(torch.load(SAVE_CKPT, map_location=device))
    evaluate_ap_by_size(model, val_loader,  device=device, out_dir=OUTPUT_DIR, tag="val_final")
    if test_loader is not None:
        evaluate_ap_by_size(model, test_loader, device=device, out_dir=OUTPUT_DIR, tag="test_final")

    # Run summary
    with open(os.path.join(OUTPUT_DIR, "det_run_summary.json"), "w") as f:
        json.dump({
            "epochs": EPOCHS,
            "train_pairs_used": len(train_pairs),
            "val_pairs_used": len(val_pairs),
            "test_pairs_used": len(test_pairs),
            "fractions": {"train": F_TRAIN, "val": F_VAL, "test": F_TEST}
        }, f, indent=2)
    print("Done. Metrics saved under", OUTPUT_DIR)

if __name__ == "__main__":
    main()
