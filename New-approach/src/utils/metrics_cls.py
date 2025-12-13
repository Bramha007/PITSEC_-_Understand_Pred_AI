# File: src/utils/metrics_cls.py (CORRECTED for WxH Class Labels)

import os, json
import numpy as np
from sklearn.metrics import confusion_matrix

# FIX: Import the configuration file which contains the SIZE_CLASS_MAP
from src.setup import config_cls as config 

# FIX: CLASS_NAMES is now derived from the keys of the WxH map (e.g., "8x8", "32x64").
# This list will contain all 25 class names (0 through 24).
CLASS_NAMES = list(config.SIZE_CLASS_MAP.keys()) 

def summarize_classifier(y_true, y_pred, out_dir="outputs", tag="test"):
    os.makedirs(out_dir, exist_ok=True)
    y_true = np.array(y_true); y_pred = np.array(y_pred)
    
    # The labels list must span all possible class indices (0 up to 24)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    
    per_class_acc = {}
    for i, name in enumerate(CLASS_NAMES):
        # The logic for calculating per-class accuracy remains correct
        denom = max(1, cm[i].sum())
        per_class_acc[name] = float(cm[i, i] / denom)
        
    overall = float((y_true == y_pred).mean()) if len(y_true) else 0.0

    js = {
        "tag": tag,
        "overall_acc": overall,
        "per_class_acc": per_class_acc,
        "support": {CLASS_NAMES[i]: int(cm[i].sum()) for i in range(len(CLASS_NAMES))}
    }
    out_json = os.path.join(out_dir, f"cls_metrics_{tag}.json")
    with open(out_json, "w") as f:
        json.dump(js, f, indent=2)
    return js, out_json