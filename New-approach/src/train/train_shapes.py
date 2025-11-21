# File: train_shapes.py (Refactored for Modularity and CUDA)

import os, time, json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- MODULAR IMPORTS ---
from src.setup import config_det as config
from src.utils.device_utils import select_device
from src.dataio.voc_parser import paired_image_xml_list
from src.dataio.det_dataset import GeometricShapeDataset, collate_fn
from src.dataio.det_transforms import Compose, ToTensor, RandomHorizontalFlip, ClampBoxes 
from src.models.fasterrcnn import build_fasterrcnn # GPU-agnostic model builder
from src.utils.metrics_det import evaluate_ap_by_size
from src.dataio.split_utils import subsample_pairs

# --- 1. Modular DataLoader Function ---
def make_loader(pairs, train=False):
    """Creates a DataLoader using configuration settings."""
    if not pairs: return None
    
    # Use config.DEVICE to check for GPU and enable pin_memory if CUDA is active
    is_cuda = select_device(config.DEVICE).type == 'cuda'
    
    # Transforms include ToTensor, RandomHorizontalFlip (for train), and ClampBoxes (for robustness)
    transforms_list = [ToTensor(), ClampBoxes()]
    if train:
        transforms_list.insert(1, RandomHorizontalFlip(0.5)) # Insert flip after ToTensor
    
    ds = GeometricShapeDataset(pairs, transforms=Compose(transforms_list))
    
    return DataLoader(
        ds, 
        batch_size=config.BATCH_SIZE, 
        shuffle=train and len(pairs) > 1,
        num_workers=config.NUM_WORKERS, 
        collate_fn=collate_fn,
        pin_memory=is_cuda # Pin memory dramatically speeds up host-to-device transfers
    )


def main():
    # --- 2. SETUP AND DEVICE INITIALIZATION ---
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(config.SEED)
    
    # DYNAMIC DEVICE SELECTION: Uses 'auto' from config.py to select CUDA or CPU
    device = select_device(config.DEVICE) 

    # --- 3. DATA PREPARATION ---
    
    # TRAINING/VALIDATION (SQUARES DATA)
    train_pairs_all = paired_image_xml_list(config.IMG_DIR_TRAIN, config.XML_DIR_ALL)
    val_pairs_all   = paired_image_xml_list(config.IMG_DIR_VAL,   config.XML_DIR_ALL)
    
    # TESTING (RECTANGLES DATA - Project Requirement)
    test_pairs_all  = paired_image_xml_list(config.IMG_DIR_TEST_RECT, config.XML_DIR_ALL_RECT)

    # Subsample deterministically using config fractions
    train_pairs = subsample_pairs(train_pairs_all, config.F_TRAIN, seed=config.SEED, max_items=config.MAX_TRAIN_ITEMS)
    val_pairs   = subsample_pairs(val_pairs_all,   config.F_VAL,   seed=config.SEED)
    test_pairs  = subsample_pairs(test_pairs_all,  config.F_TEST,  seed=config.SEED)

    print(f"Data Subsets (Used/Total): Train={len(train_pairs)}/{len(train_pairs_all)}, Val={len(val_pairs)}/{len(val_pairs_all)}, Test={len(test_pairs)}")

    train_loader = make_loader(train_pairs, train=True)
    val_loader   = make_loader(val_pairs,   train=False)
    test_loader  = make_loader(test_pairs,  train=False)

    # --- 4. MODEL, OPTIMIZER, SCHEDULER ---
    # Dynamically get the number of classes (2: Background + Shape Class)
    NUM_CLASSES = GeometricShapeDataset.get_num_classes() 
    
    # Build model using GPU-agnostic function and move to the selected device
    model = build_fasterrcnn(num_classes=NUM_CLASSES).to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=config.LR, momentum=0.9, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, config.EPOCHS//2), gamma=0.1)

    # --- 5. TRAINING LOOP (CRUCIAL CUDA TRANSFERS) ---
    best_val_ap = -1.0
    for ep in range(config.EPOCHS):
        model.train()
        losses = []
        t0 = time.time()
        
        for imgs, tgts in tqdm(train_loader, desc=f"Epoch {ep+1}/{config.EPOCHS}"):
            # CRITICAL: Move data to device with non_blocking=True (optimized for CUDA)
            imgs = [img.to(device, non_blocking=True) for img in imgs]
            tgts = [{k:v.to(device, non_blocking=True) for k,v in t.items()} for t in tgts]
            
            loss_dict = model(imgs, tgts)
            loss = sum(loss_dict.values())
            
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(float(loss.item()))
            
        sch.step()

        # --- 6. EVALUATION AND CHECKPOINTING ---
        val_js, _, _ = evaluate_ap_by_size(model, val_loader, device=device, out_dir=config.OUTPUT_DIR, tag=f"val_ep{ep+1}")
        val_ap = val_js["ap50_global"]
        print(f"Epoch {ep+1}: loss={np.mean(losses):.4f} | val AP@0.5={val_ap:.3f} | {time.time()-t0:.1f}s")
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            torch.save(model.state_dict(), config.SAVE_CKPT)
            print(f"  ✓ saved best → {config.SAVE_CKPT}")

    # --- 7. FINAL EVALUATION (SQUARES AND RECTANGLES) ---
    model.load_state_dict(torch.load(config.SAVE_CKPT, map_location=device))
    
    print("\n--- Final Evaluation (Trained on Squares) ---")
    # Final eval on validation (Squares)
    evaluate_ap_by_size(model, val_loader,  device=device, out_dir=config.OUTPUT_DIR, tag="val_final_squares")
    
    # Final eval on test (Rectangles)
    if test_loader is not None:
        evaluate_ap_by_size(model, test_loader, device=device, out_dir=config.OUTPUT_DIR, tag="test_final_rectangles")

    # Run summary
    with open(os.path.join(config.OUTPUT_DIR, "det_run_summary.json"), "w") as f:
        json.dump({
            "epochs": config.EPOCHS,
            "train_pairs_used": len(train_pairs),
            "val_pairs_used": len(val_pairs),
            "test_pairs_used": len(test_pairs),
            "fractions": {"train": config.F_TRAIN, "val": config.F_VAL, "test": config.F_TEST}
        }, f, indent=2)
    print("Done. Metrics saved under", config.OUTPUT_DIR)

if __name__ == "__main__":
    main()
    # NEW: Automatically call the visualization function
    print("\n--- Starting Automatic Visualization ---")
    from src.utils.infer_and_viz import run_and_visualize_all # <-- Import the function

    # Visualize Rectangles (Test Domain)
    run_and_visualize_all(test_on_rectangles=True, limit_count=20) 

    # Optionally visualize Squares (Training Domain)
    run_and_visualize_all(test_on_rectangles=False, limit_count=20)