import os, random, numpy as np, torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.dataio.voc_parser import paired_image_xml_list
from src.dataio.split_utils import subsample_pairs
from src.dataio.cls_dataset_stream import SquaresClassificationDatasetStream
from src.models.resnet_cls import build_resnet_classifier
from src.utils.cls_viz import save_confusion_matrix, save_sample_grid, print_classification_report
from src.utils.metrics_cls import summarize_classifier

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
SAVE_CKPT  = os.path.join(OUTPUT_DIR, "resnet_cls.pt")

BATCH_SIZE = 64          # CPU okay with 224 canvas? lower if needed
EPOCHS     = 2
LR         = 1e-3
SEED       = 42

# Fractions (grow train over time)
F_TRAIN = 0.001   # begin with 5% of train
F_VAL   = 0.005
F_TEST  = 0.005

CANVAS    = 224
USE_PAD   = True         # Approach A
# -------------------

def set_seed(s=42):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    set_seed(SEED)
    device = torch.device("cpu")

    # Pair splits
    train_pairs_all = paired_image_xml_list(IMG_DIR_TRAIN, XML_DIR_ALL)
    val_pairs_all   = paired_image_xml_list(IMG_DIR_VAL,   XML_DIR_ALL)
    test_pairs_all  = paired_image_xml_list(IMG_DIR_TEST_RECT,  XML_DIR_ALL_RECT)

    train_pairs = subsample_pairs(train_pairs_all, F_TRAIN, seed=SEED)
    val_pairs   = subsample_pairs(val_pairs_all,   F_VAL,   seed=SEED)
    test_pairs  = subsample_pairs(test_pairs_all,  F_TEST,  seed=SEED)

    print(f"Train pairs: {len(train_pairs)} / {len(train_pairs_all)} "
          f"(frac={F_TRAIN})")
    print(f"Val   pairs: {len(val_pairs)} / {len(val_pairs_all)} "
          f"(frac={F_VAL})")
    print(f"Test  pairs: {len(test_pairs)} / {len(test_pairs_all)} "
          f"(frac={F_TEST})")

    # Streaming datasets (build index only)
    ds_train = SquaresClassificationDatasetStream(train_pairs, canvas=CANVAS, train=True,  use_padding_canvas=USE_PAD)
    ds_val   = SquaresClassificationDatasetStream(val_pairs,   canvas=CANVAS, train=False, use_padding_canvas=USE_PAD)
    ds_test  = SquaresClassificationDatasetStream(test_pairs,  canvas=CANVAS, train=False, use_padding_canvas=USE_PAD)

    # Optional: cap max items by Subset
    # ds_train = Subset(ds_train, list(range(min(len(ds_train), 200000))))

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    

    model = build_resnet_classifier(num_classes=5).to(device)
    # (optional tiny-class boost if needed later)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    best_acc = 0.0
    for ep in range(EPOCHS):
        model.train()
        losses = []
        for x,y in tqdm(train_loader, desc=f"Epoch {ep+1}/{EPOCHS}"):
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()

        # val
        model.eval()
        correct,total = 0,0
        with torch.no_grad():
            for x,y in val_loader:
                out = model(x); pred = out.argmax(1)
                correct += (pred==y).sum().item()
                total   += y.size(0)
        acc = correct/total if total else 0.0
        print(f"Epoch {ep+1}: loss={np.mean(losses):.4f} | val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), SAVE_CKPT)
            print(f"  ✓ saved best → {SAVE_CKPT}")

    print("Best val acc:", best_acc)

    # Test + visuals + JSON
    model.load_state_dict(torch.load(SAVE_CKPT, map_location=device))
    model.eval()
    all_y, all_pred, total, correct = [], [], 0, 0
    sample_images = []
    with torch.no_grad():
        for x,y in tqdm(test_loader, desc="Test"):
            out = model(x); pred = out.argmax(1)
            correct += (pred==y).sum().item(); total += y.size(0)
            all_y.extend(y.tolist()); all_pred.extend(pred.tolist())
            if len(sample_images) < 36:
                need = 36 - len(sample_images)
                sample_images.extend(x[:need])
    test_acc = correct/total if total else 0.0
    print(f"Test acc: {test_acc:.3f}")

    from torch import stack
    cm_path = save_confusion_matrix(all_y, all_pred, out_path=os.path.join(OUTPUT_DIR, "confmat_cls.png"))
    print("Saved:", cm_path)
    if len(sample_images) > 0:
        grid_path = save_sample_grid(stack(sample_images), all_y[:len(sample_images)], all_pred[:len(sample_images)],
                                     out_path=os.path.join(OUTPUT_DIR, "preds_grid.png"))
        print("Saved:", grid_path)
    summarize_classifier(all_y, all_pred, out_dir=OUTPUT_DIR, tag="test")

if __name__ == "__main__":
    main()
