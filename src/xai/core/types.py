from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class ROIBox:
    x1: float; y1: float; x2: float; y2: float

@dataclass
class XAIReportEntry:
    image: str
    header_json: str
    pred_overlay: Optional[str]
    rpn_heatmap: Optional[str]
    roi_heatmap: Optional[str]
    counterfactuals: Dict[str, Optional[str]]

@dataclass
class CFProbe:
    kind: str                   # 'occlusion' | 'jitter' | 'scale'
    params: Dict[str, float]
    logit: float
