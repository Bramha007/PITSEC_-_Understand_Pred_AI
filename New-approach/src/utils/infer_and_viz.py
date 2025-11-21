# File: infer_and_viz.py (Refactored)

import torch
import os
from tqdm import tqdm

# --- MODULAR IMPORTS ---
from src.setup import config_det # Use your modular config file
from src.utils.device_utils import select_device # For dynamic device selection
from src.utils.viz import show_prediction 
from src.dataio.voc_parser import paired_image_xml_list
# Use the new, semantic dataset name: GeometricShapeDataset
from src.dataio.det_dataset import GeometricShapeDataset, collate_fn 
from src.dataio.det_transforms import Compose, ToTensor
from src.dataio.split_utils import subsample_pairs
from src.models.fasterrcnn import build_fasterrcnn # Use the GPU-agnostic model builder
from torch.utils.data import DataLoader

# NOTE: The hardcoded configuration block is removed.
# ----------------------------------------------------------------

@torch.inference_mode()
def run_and_visualize_all(test_on_rectangles: bool = True, limit_count: int | None = 10):
    """
    Runs inference on a test subset and saves visualizations for predicted and GT boxes.
    
    Args:
        test_on_rectangles: If True, tests on Rectangles data; otherwise, tests on Squares data.
        limit_count: Maximum number of images to visualize. Set to None to visualize all.
    """
    
    # 1. SETUP AND DEVICE INITIALIZATION
    device = select_device(config_det.DEVICE)
    torch.manual_seed(config_det.SEED)
    
    # Determine which dataset split to use based on the argument
    if test_on_rectangles:
        data_root = config_det.RECT_DATA_ROOT
        img_dir = config_det.IMG_DIR_TEST_RECT 
        xml_dir = config_det.XML_DIR_ALL_RECT 
        tag = "rectangles"
    else:
        # Use Squares validation data for comparison
        data_root = config_det.DATA_ROOT
        img_dir = config_det.IMG_DIR_VAL 
        xml_dir = config_det.XML_DIR_ALL 
        tag = "squares"
        
    output_viz_dir = os.path.join(config_det.OUTPUT_DIR, f"viz_{tag}")
    os.makedirs(output_viz_dir, exist_ok=True)
    
    # 2. PREPARE DATA LOADER
    print(f"Preparing data for visualization (Set: {tag.upper()})...")
    
    pairs_all = paired_image_xml_list(img_dir, xml_dir)
    # Use config's fraction and seed to ensure consistency with training
    pairs = subsample_pairs(pairs_all, config_det.F_TEST, seed=config_det.SEED) 
    
    ds_val = GeometricShapeDataset( # <-- Use the semantic dataset name
        pairs,
        transforms=Compose([ToTensor()])
    )
    
    val_loader = DataLoader(
        ds_val, 
        batch_size=1, # BATCH_SIZE must be 1 for easy visualization
        shuffle=False,
        num_workers=config_det.NUM_WORKERS,
        collate_fn=collate_fn 
    )

    if len(ds_val) == 0:
        print("Error: No data found or selected.")
        return

    # 3. LOAD THE MODEL
    print(f"Loading model from: {config_det.SAVE_CKPT}")
    num_classes = GeometricShapeDataset.get_num_classes() # Get class count dynamically
    
    # Use the GPU-agnostic model builder
    model = build_fasterrcnn(num_classes=num_classes).to(device)

    if not os.path.exists(config_det.SAVE_CKPT):
        print(f"ERROR: Checkpoint file not found at {config_det.SAVE_CKPT}. Please run training first.")
        return

    model.load_state_dict(torch.load(config_det.SAVE_CKPT, map_location=device))
    model.eval() 

    # 4. ITERATE, PREDICT, AND SAVE
    print(f"\nStarting inference and saving visualizations to {output_viz_dir}...")
    
    for i, (imgs, tgts) in enumerate(tqdm(val_loader, desc=f"Visualizing {tag}")):
        if limit_count is not None and i >= limit_count: break # Apply limit
        
        # CRITICAL: Move data to the device before inference
        imgs = [img.to(device) for img in imgs]
        
        predictions_list = model(imgs) 
        pred = predictions_list[0]
        
        # Move Tensors back to CPU for Matplotlib visualization (show_prediction handles this)
        img_tensor = imgs[0].cpu()
        target_dict = tgts[0]
        
        # Get filename using the index of the pair list
        original_filename = os.path.basename(pairs[i][0])
        output_filename = f"pred_{original_filename.rsplit('.', 1)[0]}.png"
        output_image_path = os.path.join(output_viz_dir, output_filename)
        
        # Visualize and Save (using the function from viz.py)
        show_prediction(
            image_tensor=img_tensor, 
            pred=pred, 
            gt=target_dict, 
            score_thr=0.7, 
            save_path=output_image_path 
        )

    print(f"\n✅ Visualizations complete. Saved to: {output_viz_dir}")
    
if __name__ == "__main__":
    # --- Execute visualization for Rectangles (Cross-Domain Test) ---
    run_and_visualize_all(test_on_rectangles=True, limit_count=20) 
    
    # --- Execute visualization for Squares (Training Domain Check) ---
    # To check the squares data, uncomment the line below:
    run_and_visualize_all(test_on_rectangles=False, limit_count=20)