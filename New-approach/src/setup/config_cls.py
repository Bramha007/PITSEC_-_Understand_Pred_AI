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

EPOCHS = 10                 # Increased for better convergence
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

SIZE_BINS = [8, 16, 32, 64, 128]
NUM_CLS_CLASSES = len(SIZE_BINS) 

# --- D. DATA SUBSET FRACTIONS ---
F_TRAIN = 0.1
F_VAL = 0.05
F_TEST = 0.05
MAX_TRAIN_ITEMS = None