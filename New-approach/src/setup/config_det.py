# File: config.py

import os

# ====================================================================
# A. GENERAL SETUP & PATHS
# ====================================================================

# The root directory for the dataset. Adjust this to your environment.
DATA_ROOT = r"E:\WPT-Project\Data\sized_squares_filled"  
# DATA_ROOT = "/pitsec_sose2025_team3_1/data/sized_squares_filled" # for Linux

# Output directory for checkpoints and metrics
OUTPUT_DIR = "outputs_v2"
# Checkpoint save path (uses os.path.join for cross-platform compatibility)
SAVE_CKPT = os.path.join(OUTPUT_DIR, "fasterrcnn_shapes_final.pt")

# Training, Validation, and Annotations paths for the SQUARES dataset
IMG_DIR_TRAIN = os.path.join(DATA_ROOT, "train")
IMG_DIR_VAL   = os.path.join(DATA_ROOT, "val")
XML_DIR_ALL   = os.path.join(DATA_ROOT, "annotations")

# Testing paths for the RECTANGLES dataset (Used for cross-domain testing)
# You MUST change the path below to point to your 'sized_rectangles_filled' location
RECT_DATA_ROOT = r'E:\WPT-Project\Data\sized_rectangles_filled' 
# RECT_DATA_ROOT = "/pitsec_sose2025_team3_1/data/sized_rectangles_filled" # for Linux
IMG_DIR_TEST_RECT  = os.path.join(RECT_DATA_ROOT, 'test')
XML_DIR_ALL_RECT   = os.path.join(RECT_DATA_ROOT, 'annotations')


# ====================================================================
# B. MODEL & TRAINING PARAMETERS
# ====================================================================

EPOCHS      = 4      # Increased for better training
BATCH_SIZE  = 4       # Increased for GPU efficiency
LR          = 0.003
SEED        = 42      # Ensures deterministic results
NUM_WORKERS = 4       # Use 4-8 workers when on GPU to prevent data bottlenecks

# DEVICE: 'auto' checks for CUDA first, then defaults to CPU.
# Use 'cpu' to force CPU, or 'cuda' to force GPU.
DEVICE      = "auto" 

OPTIMIZER_NAME = "SGD" #"AdamW" # Options: "SGD", "AdamW"

# ====================================================================
# C. DATA SUBSET FRACTIONS
# ====================================================================

# Fractions (0.0 to 1.0) to subsample the dataset splits
F_TRAIN = 0.0005 
F_VAL   = 0.005
F_TEST  = 0.005

# Hard-cap the training items if the fraction still yields too much data
MAX_TRAIN_ITEMS = None