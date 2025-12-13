# import os
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix, classification_report

# # Map indices to human-readable size buckets (adjust if you change bins)
# CLASS_NAMES = ["8", "16", "32", "64", "128"]

# def save_confusion_matrix(y_true, y_pred, out_path="outputs/confmat_cls.png"):
#     cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
#     fig, ax = plt.subplots(figsize=(5,5))
#     im = ax.imshow(cm, interpolation='nearest')
#     ax.figure.colorbar(im, ax=ax)
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]),
#            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
#            ylabel='True label', xlabel='Predicted label',
#            title='Confusion Matrix')
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
#     thresh = cm.max() / 2.0 if cm.size else 0
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, format(cm[i, j], 'd'),
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
#     return out_path

# def save_sample_grid(images, y_true, y_pred, out_path="outputs/preds_grid.png", max_samples=36):
#     """
#     images: tensor [N,3,H,W] in normalized space (ImageNet stats)
#     y_true, y_pred: lists/arrays of ints
#     """
#     # de-normalize for display
#     import torch
#     mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
#     std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

#     n = min(len(images), max_samples)
#     cols = 6
#     rows = math.ceil(n / cols)

#     fig, axes = plt.subplots(rows, cols, figsize=(cols*2.3, rows*2.3))
#     axes = np.array(axes).reshape(rows, cols)

#     for i in range(rows*cols):
#         ax = axes.flat[i]
#         ax.axis("off")
#         if i >= n: 
#             continue
#         img = images[i].cpu() * std + mean
#         img = img.clamp(0,1).permute(1,2,0).numpy()

#         t = int(y_true[i]); p = int(y_pred[i])
#         ok = (t == p)
#         title = f"T:{CLASS_NAMES[t]} • P:{CLASS_NAMES[p]}"
#         ax.imshow(img)
#         ax.set_title(title, color=("green" if ok else "red"), fontsize=9)

#     plt.tight_layout()
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
#     return out_path

# def print_classification_report(y_true, y_pred):
#     print("\n=== Classification Report ===")
#     print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))


import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import torch # Need this for save_sample_grid tensor operations

# --- FIX: Dynamically import configuration and class names ---
from src.setup import config_cls as config 

# DYNAMIC CLASS NAMES: Load all 25 WxH class names (e.g., "8x8", "32x64")
CLASS_NAMES = list(config.SIZE_CLASS_MAP.keys()) 
# --- END FIX ---


def save_confusion_matrix(y_true, y_pred, out_path="outputs/confmat_cls.png"):
    num_classes = len(CLASS_NAMES) 
    # Use labels=list(range(num_classes)) which is [0, 1, ..., 24]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes))) 
    
    # Increase figsize for the 25x25 matrix
    fig, ax = plt.subplots(figsize=(10,10)) 
    im = ax.imshow(cm, interpolation='nearest')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
           ylabel='True label', xlabel='Predicted label',
           title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # Only display text for non-zero counts
            if cm[i, j] > 0:
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out_path

def save_sample_grid(images, y_true, y_pred, out_path="outputs/preds_grid.png", max_samples=36):
    """
    images: tensor [N,3,H,W] in normalized space (ImageNet stats)
    y_true, y_pred: lists/arrays of ints
    """
    # de-normalize for display
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    n = min(len(images), max_samples)
    cols = 6
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2.3, rows*2.3))
    axes = np.array(axes).reshape(rows, cols)

    for i in range(rows*cols):
        ax = axes.flat[i]
        ax.axis("off")
        if i >= n: 
            continue
        img = images[i].cpu() * std + mean
        img = img.clamp(0,1).permute(1,2,0).numpy()

        t = int(y_true[i]); p = int(y_pred[i])
        ok = (t == p)
        # FIX: This line now uses the 25-item list, resolving the IndexError
        title = f"T:{CLASS_NAMES[t]} • P:{CLASS_NAMES[p]}" 
        
        ax.imshow(img)
        ax.set_title(title, color=("green" if ok else "red"), fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return out_path

def print_classification_report(y_true, y_pred):
    print("\n=== Classification Report ===")
    # Ensure target_names is the 25-item list
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3))