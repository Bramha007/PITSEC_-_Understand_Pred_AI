# File: src/setup/config_cls.py

import os

# --- A. GENERAL SETUP & PATHS (Using absolute paths for Linux stability) ---
# NOTE: Replace the absolute path below with your confirmed path from the server
# DATA_ROOT = r"E:\WPT-Project\Data\sized_squares_filled"  
DATA_ROOT = "/pitsec_sose2025_team3_1/data/sized_squares_filled" # for Linux


# Output directory for classification results
OUTPUT_DIR = "outputs_v3"
SAVE_CKPT = os.path.join(OUTPUT_DIR, "resnet_cls_best.pt")

# Training, Validation, and Annotations paths
IMG_DIR_TRAIN = os.path.join(DATA_ROOT, "train")
IMG_DIR_VAL = os.path.join(DATA_ROOT, "val")
XML_DIR_ALL = os.path.join(DATA_ROOT, "annotations")

# RECT_DATA_ROOT = r'E:\WPT-Project\Data\sized_rectangles_filled' 
RECT_DATA_ROOT = "/pitsec_sose2025_team3_1/data/sized_rectangles_filled" # for Linux
IMG_DIR_TEST_RECT  = os.path.join(RECT_DATA_ROOT, 'test')
XML_DIR_ALL_RECT   = os.path.join(RECT_DATA_ROOT, 'annotations')


# --- B. CLASSIFICATION MODEL & TRAINING PARAMETERS ---

EPOCHS = 5                 # Increased for better convergence
BATCH_SIZE = 4              # Good size for 224x224 crops on GPU
LR = 1e-3
SEED = 42
NUM_WORKERS = 4              # Keep high for GPU data loading

# DEVICE: 'auto' will select CUDA if available
DEVICE = "auto"

# RESNET SETTINGS
CANVAS_SIZE = 224            # Input size for ResNet18
USE_PADDING_CANVAS = True    # Use white canvas padding approach

# OPTIMIZER
OPTIMIZER_NAME = "Adam"      # Adam is common for classification tasks


# --- C. CLASSIFICATION CLASSES (This is the file you originally confirmed) ---

SIZE_CLASS_MAP = {
    # -----------------------------------------------------
    # 1. SQUARE CLASSES (W = H) - IDs 0 through 4
    # -----------------------------------------------------
    "8x8": 0, "16x16": 1, "32x32": 2, "64x64": 3, "128x128": 4,

    # -----------------------------------------------------
    # 2. RECTANGLE CLASSES (W != H) - IDs 5 through 24
    # -----------------------------------------------------
    # Base 8px
    "8x16": 5, "16x8": 6,
    "8x32": 7, "32x8": 8,
    "8x64": 9, "64x8": 10,
    "8x128": 11, "128x8": 12,

    # Base 16px (excluding 8x16 and 16x8)
    "16x32": 13, "32x16": 14,
    "16x64": 15, "64x16": 16,
    "16x128": 17, "128x16": 18,

    # Base 32px (excluding smaller dimensions)
    "32x64": 19, "64x32": 20,
    "32x128": 21, "128x32": 22,

    # Base 64px (excluding smaller dimensions)
    "64x128": 23, "128x64": 24,
    
    # Total unique classes: 25
}

NUM_CLS_CLASSES = 25 # Total number of unique WxH combinations

# --- D. DATA SUBSET FRACTIONS ---
F_TRAIN = 0.2
F_VAL = 0.05
F_TEST = 0.05
MAX_TRAIN_ITEMS = None