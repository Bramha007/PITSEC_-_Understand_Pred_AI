# # File: src/dataio/cls_dataset_stream.py

# from torch.utils.data import Dataset
# from PIL import Image
# import torchvision.transforms as T
# from typing import List, Tuple
# from .voc_parser import parse_voc
# # NEW: Import size definitions from modular configuration
# from src.setup.config_cls import SIZE_BINS 

# class GeometricShapeClassificationDatasetStream(Dataset):
#     """
#     Dataset for single-shape classification (size classification).
#     Crops the object on-the-fly and assigns a class label based on side length.
#     """
#     def __init__(self, pairs: List[Tuple[str,str]], canvas=224, train=True,
#                  use_padding_canvas=True, margin_px=16):
#         self.canvas = canvas
#         self.train = train
#         self.use_padding_canvas = use_padding_canvas
#         self.margin_px = margin_px

#         # Build a flat index of all boxes for the given (img, xml) pairs.
#         self.index = []  # list of tuples: (img_path, x1,y1,x2,y2, label, W,H)
#         for img_path, xml_path in pairs:
#             rec = parse_voc(xml_path)
#             W, H = rec["width"], rec["height"]
            
#             # NOTE: We assume the detection model works perfectly, so we use GT boxes
#             for x1, y1, x2, y2 in rec["boxes"]:
#                 # Side length is max(width, height) for safe size binning
#                 side = max(x2-x1, y2-y1) 
#                 label = self._size_to_class(side)
#                 self.index.append((img_path, x1, y1, x2, y2, label, W, H))

#         # Standard transforms for classification backbone (ImageNet normalization)
#         tfms = [T.ToTensor(),
#                 T.Normalize(mean=[0.485,0.456,0.406],
#                             std=[0.229,0.224,0.225])]
#         if train:
#             # Apply random flip BEFORE ToTensor/Normalization
#             tfms = [T.RandomHorizontalFlip()] + tfms
            
#         self.transform = T.Compose(tfms)

#     def __len__(self):
#         return len(self.index)

#     def __getitem__(self, idx):
#         img_path, x1, y1, x2, y2, label, W, H = self.index[idx]
#         img = Image.open(img_path).convert("RGB")

#         if self.use_padding_canvas:
#             # Use cropping and centering on a white canvas (robust method)
#             crop = img.crop((x1, y1, x2, y2))
#             canvas = Image.new("RGB", (self.canvas, self.canvas), (255,255,255))
#             ox = (self.canvas - crop.size[0]) // 2
#             oy = (self.canvas - crop.size[1]) // 2
#             canvas.paste(crop, (ox, oy))
#             sample_img = canvas
#         else:
#             # Use cropping with margin and resizing
#             x1m = max(0, x1 - self.margin_px)
#             y1m = max(0, y1 - self.margin_px)
#             x2m = min(W, x2 + self.margin_px)
#             y2m = min(H, y2 + self.margin_px)
#             crop = img.crop((x1m, y1m, x2m, y2m))
#             sample_img = crop.resize((self.canvas, self.canvas), Image.BILINEAR)

#         # Apply transformation pipeline and return image tensor and integer label
#         return self.transform(sample_img), label

#     @staticmethod
#     def _size_to_class(side: int) -> int:
#         """
#         Maps the side length of the bounding box to the closest integer class ID (0 to 4).
#         """
#         bins = SIZE_BINS # Use the modular list of sizes [8, 16, 32, 64, 128]
#         # Finds the index (class ID) of the bin closest to the object's side length
#         return min(range(len(bins)), key=lambda i: abs(side - bins[i]))


# File: src/dataio/cls_dataset_stream.py (UPDATED FOR WxH LABELS)
# File: src/dataio/cls_dataset_stream.py

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from typing import List, Tuple
from .voc_parser import parse_voc
# FIX: Import the full WxH class map from the setup configuration
from src.setup.config_cls import SIZE_CLASS_MAP 

class GeometricShapeClassificationDatasetStream(Dataset):
    """
    Dataset for multi-class WxH size classification.
    Crops the object on-the-fly and assigns a unique WxH class label (0-24).
    """
    def __init__(self, pairs: List[Tuple[str,str]], canvas=224, train=True,
                 use_padding_canvas=True, margin_px=16):
        self.canvas = canvas
        self.train = train
        self.use_padding_canvas = use_padding_canvas
        self.margin_px = margin_px

        # Build a flat index of all boxes for the given (img, xml) pairs.
        # list of tuples: (img_path, x1,y1,x2,y2, label_ID, W,H)
        self.index = []  
        for img_path, xml_path in pairs:
            # FIX: We rely on parse_voc returning 'labels_wxh_str'
            rec = parse_voc(xml_path)
            W, H = rec["width"], rec["height"]
            
            # Ensure the parser returned the necessary WxH strings
            if "labels_wxh_str" not in rec:
                print(f"Error: Missing WxH strings in parser output for {xml_path}. Check voc_parser.py.")
                continue

            # Iterate over boxes and their corresponding WxH string labels
            for (x1, y1, x2, y2), wxh_str in zip(rec["boxes"], rec["labels_wxh_str"]):
                
                # --- NEW: Convert WxH string to integer class ID (0-24) ---
                try:
                    # Look up the class ID using the WxH string (e.g., '64x32')
                    label_id = SIZE_CLASS_MAP[wxh_str]
                except KeyError:
                    # Skip samples with WxH combinations not defined in our map
                    continue 
                
                self.index.append((img_path, x1, y1, x2, y2, label_id, W, H))

        # Standard transforms for classification backbone (ImageNet normalization)
        # --- NEW: Added Random Rotation for better generalization ---
        tfms = [
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225])
        ]
        
        if train:
            # Apply Random Flip and Random Rotation (e.g., up to 45 degrees)
            tfms = [T.RandomHorizontalFlip(), T.RandomRotation(degrees=45)] + tfms
            
        self.transform = T.Compose(tfms)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # FIX: Retrieve label_ID instead of 'label'
        img_path, x1, y1, x2, y2, label_id, W, H = self.index[idx]
        img = Image.open(img_path).convert("RGB")

        if self.use_padding_canvas:
            # Use cropping and centering on a white canvas (robust method)
            crop = img.crop((x1, y1, x2, y2))
            canvas = Image.new("RGB", (self.canvas, self.canvas), (255,255,255))
            ox = (self.canvas - crop.size[0]) // 2
            oy = (self.canvas - crop.size[1]) // 2
            canvas.paste(crop, (ox, oy))
            sample_img = canvas
        else:
            # Use cropping with margin and resizing
            x1m = max(0, x1 - self.margin_px)
            y1m = max(0, y1 - self.margin_px)
            x2m = min(W, x2 + self.margin_px)
            y2m = min(H, y2 + self.margin_px)
            crop = img.crop((x1m, y1m, x2m, y2m))
            sample_img = crop.resize((self.canvas, self.canvas), Image.BILINEAR)

        # Apply transformation pipeline and return image tensor and integer label_ID
        return self.transform(sample_img), label_id

    # FIX: The _size_to_class static method is REMOVED as it is no longer used
    # due to the direct mapping from WxH string.