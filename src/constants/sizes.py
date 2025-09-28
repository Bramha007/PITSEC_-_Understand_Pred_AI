# src/constants/sizes.py
# Unified size bucket edges and labels for cls and det
DET_EDGES = [0, 12, 24, 48, 96, float("inf")]
DET_LABELS = ["XS", "S", "M", "L", "XL"]
CLS_EDGES = [8, 16, 32, 64, 128]
CLS_LABELS = ["XS", "S", "M", "L", "XL"]
