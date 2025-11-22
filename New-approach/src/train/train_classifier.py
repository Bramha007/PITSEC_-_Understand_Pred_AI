# File: train_classifier.py (Refactored for Size Classification and CUDA)

import os, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import List, Tuple

# --- MODULAR IMPORTS ---
# Load settings from the Classification config file
from src.setup import config_cls as config 
from src.utils.device_utils import select_device
from src.dataio.voc_parser import paired_image_xml_list
from src.dataio.split_utils import subsample_pairs
# Use the new semantic dataset name
from src.dataio.cls_dataset_stream import GeometricShapeClassificationDatasetStream 
from src.models.resnet import build_resnet_classifier
from src.utils.cls_viz import save_confusion_matrix, save_sample_grid, print_classification_report
from src.utils.metrics_cls import summarize_classifier


def main():
    # --- 1. SETUP AND DEVICE INITIALIZATION ---
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    # Use torch.manual_seed only for PyTorch reproducibility
    torch.manual_seed(config.SEED)
    
    # DYNAMIC DEVICE SELECTION (CUDA READY)
    device = select_device(config.DEVICE)
    is_cuda = device.type == "cuda"

    # --- 2. DATA PREPARATION ---

    # Training/Validation (Squares Data)
    train_pairs_all = paired_image_xml_list(config.IMG_DIR_TRAIN, config.XML_DIR_ALL)
    val_pairs_all   = paired_image_xml_list(config.IMG_DIR_VAL, config.XML_DIR_ALL)

    # Testing (Rectangles Data - Project Requirement)
    test_pairs_all  = paired_image_xml_list(config.IMG_DIR_TEST_RECT, config.XML_DIR_ALL_RECT)

    # Subsample deterministically using config fractions
    train_pairs = subsample_pairs(train_pairs_all, config.F_TRAIN, seed=config.SEED, max_items=config.MAX_TRAIN_ITEMS)
    val_pairs   = subsample_pairs(val_pairs_all, config.F_VAL, seed=config.SEED)
    test_pairs  = subsample_pairs(test_pairs_all, config.F_TEST, seed=config.SEED)

    print(f"Data Subsets (Used/Total): Train={len(train_pairs)}/{len(train_pairs_all)}, Val={len(val_pairs)}/{len(val_pairs_all)}, Test={len(test_pairs)}")

    # --- 3. DATASETS AND DATALOADERS (GPU Optimized) ---
    ds_train = GeometricShapeClassificationDatasetStream(
        train_pairs, canvas=config.CANVAS_SIZE, train=True, use_padding_canvas=config.USE_PADDING_CANVAS
    )
    ds_val = GeometricShapeClassificationDatasetStream(
        val_pairs, canvas=config.CANVAS_SIZE, train=False, use_padding_canvas=config.USE_PADDING_CANVAS
    )
    ds_test = GeometricShapeClassificationDatasetStream(
        test_pairs, canvas=config.CANVAS_SIZE, train=False, use_padding_canvas=config.USE_PADDING_CANVAS
    )
    
    # DataLoaders setup
    train_loader = DataLoader(ds_train, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=is_cuda)
    val_loader   = DataLoader(ds_val,   batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=is_cuda)
    test_loader  = DataLoader(ds_test,  batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=is_cuda)

    # --- 4. MODEL, OPTIMIZER, SCHEDULER ---
    NUM_CLASSES = config.NUM_CLS_CLASSES
    model = build_resnet_classifier(num_classes=NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss()
    params = model.parameters()
    
    # Modular Optimizer Selection
    if config.OPTIMIZER_NAME.lower() == "adam":
        optimizer = optim.Adam(params, lr=config.LR)
    elif config.OPTIMIZER_NAME.lower() == "adamw":
        optimizer = optim.AdamW(params, lr=config.LR)
    elif config.OPTIMIZER_NAME.lower() == "sgd":
        optimizer = optim.SGD(params, lr=config.LR, momentum=0.9, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer: {config.OPTIMIZER_NAME}")

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)

    # --- 5. TRAINING LOOP (GPU-Optimized) ---
    best_acc = 0.0
    for ep in range(config.EPOCHS):
        model.train()
        losses = []
        t0 = time.time()
        
        for x, y in tqdm(train_loader, desc=f"Epoch {ep+1}/{config.EPOCHS}"):
            # CRITICAL: Move data to device with non_blocking=True
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
        scheduler.step()

        # --- 6. VALIDATION ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                out = model(x)
                pred = out.argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        acc = correct / total if total else 0.0
        print(f"Epoch {ep+1}: loss={np.mean(losses):.4f} | val_acc={acc:.3f} | {time.time()-t0:.1f}s")
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), config.SAVE_CKPT)
            print(f"  ✓ saved best → {config.SAVE_CKPT}")

    print("\nBest val acc:", best_acc)

    # --- 7. FINAL TEST AND VISUALIZATION ---
    model.load_state_dict(torch.load(config.SAVE_CKPT, map_location=device))
    model.eval()
    
    # Test on Rectangles (Cross-Domain Test)
    all_y, all_pred, total, correct = [], [], 0, 0
    sample_images = []
    
    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Test (Rectangles)"):
            x = x.to(device)
            out = model(x)
            pred = out.argmax(1)
            
            correct += (pred == y.to(device)).sum().item() 
            total += y.size(0)
            
            all_y.extend(y.tolist())
            all_pred.extend(pred.cpu().tolist())
            
            if len(sample_images) < 36:
                need = 36 - len(sample_images)
                sample_images.extend(x[:need].cpu()) # Save samples on CPU
    
    test_acc = correct / total if total else 0.0
    print(f"\nFinal Test Accuracy (Rectangles): {test_acc:.3f}")

    # Visualization and Reporting (Assuming utilities are available)
    if len(all_y) > 0:
        cm_path = save_confusion_matrix(all_y, all_pred, out_path=os.path.join(config.OUTPUT_DIR, "confmat_cls.png"))
        print("Saved:", cm_path)
        
        if len(sample_images) > 0:
            grid_path = save_sample_grid(torch.stack(sample_images), all_y[:len(sample_images)], all_pred[:len(sample_images)],
                                         out_path=os.path.join(config.OUTPUT_DIR, "preds_grid.png"))
            print("Saved:", grid_path)
        
        summarize_classifier(all_y, all_pred, out_dir=config.OUTPUT_DIR, tag="test_rectangles")

if __name__ == "__main__":
    main()