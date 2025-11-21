# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# from .voc_parser import parse_voc


# class SquaresDetectionDataset(Dataset):
#     def __init__(self, pairs, transforms=None):
#         self.pairs = pairs
#         self.transforms = transforms

#     def __len__(self):
#         return len(self.pairs)

#     def __getitem__(self, idx):
#         img_path, xml_path = self.pairs[idx]
#         img = Image.open(img_path)
#         rec = parse_voc(xml_path)

#         boxes = torch.tensor(rec["boxes"], dtype=torch.float32)
#         labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # single class = 1
#         image_id = torch.tensor([idx])
#         area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
#         iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

#         target = {
#             "boxes": boxes,
#             "labels": labels,
#             "image_id": image_id,
#             "area": area,
#             "iscrowd": iscrowd,
#         }

#         if self.transforms is not None:
#             img, target = self.transforms(img, target)
#         return img, target


# def collate_fn(batch):
#     return tuple(zip(*batch))


# File: src/dataio/det_dataset.py (Revised for single-class detection)

# import torch
# from torch.utils.data import Dataset
# from PIL import Image
# # Assuming parse_voc is now simple (just returns rec with 'boxes')
# from .voc_parser import parse_voc 


# class SquaresDetectionDataset(Dataset):
#     """
#     Dataset class for loading AMSL images and annotations, set up for 
#     SINGLE-CLASS object detection (detecting 'a shape').
#     """
#     # Define a single constant for the number of classes (Background + Shape)
#     NUM_CLASSES_DETECTION = 2 # Class ID 0 is background, Class ID 1 is the shape.

#     @classmethod
#     def get_num_classes(cls):
#         """Returns the total number of classes required by the model (2: Background + Shape)."""
#         return cls.NUM_CLASSES_DETECTION

#     def __init__(self, pairs, transforms=None):
#         self.pairs = pairs
#         self.transforms = transforms

#     def __len__(self):
#         return len(self.pairs)

#     def __getitem__(self, idx):
#         img_path, xml_path = self.pairs[idx]
#         img = Image.open(img_path)
        
#         # Assume parse_voc is simple, returning 'boxes' and other info
#         rec = parse_voc(xml_path) 

#         boxes = torch.tensor(rec["boxes"], dtype=torch.float32)
        
#         # --- Single-Class Detection: All objects get Label 1 ---
#         # The number of labels must match the number of boxes.
#         labels = torch.ones((boxes.shape[0],), dtype=torch.int64) 

#         # Standard target dictionary components
#         image_id = torch.tensor([idx])
#         area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
#         iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

#         target = {
#             "boxes": boxes,
#             "labels": labels,
#             "image_id": image_id,
#             "area": area,
#             "iscrowd": iscrowd,
#         }

#         if self.transforms is not None:
#             img, target = self.transforms(img, target)
#         return img, target


# def collate_fn(batch):
#     """Collates a list of (image, target) tuples into two separate lists."""
#     return tuple(zip(*batch))


# File: src/dataio/det_dataset.py

import torch
from torch.utils.data import Dataset
from PIL import Image
from .voc_parser import parse_voc


class GeometricShapeDataset(Dataset): # <-- RENAMED CLASS
    """
    Dataset class for loading AMSL images and annotations, set up for 
    SINGLE-CLASS object detection (detecting 'a shape').
    """
    NUM_CLASSES_DETECTION = 2 # Class ID 0 is background, Class ID 1 is the shape.

    @classmethod
    def get_num_classes(cls):
        return cls.NUM_CLASSES_DETECTION

    def __init__(self, pairs, transforms=None):
        self.pairs = pairs
        self.transforms = transforms

    # ... __len__ and __getitem__ methods remain the same ...

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, xml_path = self.pairs[idx]
        img = Image.open(img_path)
        rec = parse_voc(xml_path) 
        # ... rest of __getitem__ logic ...
        boxes = torch.tensor(rec["boxes"], dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64) 
        image_id = torch.tensor([idx])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes, "labels": labels, "image_id": image_id, "area": area, "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))