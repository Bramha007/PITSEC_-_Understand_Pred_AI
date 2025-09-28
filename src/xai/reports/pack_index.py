import os, json, glob
from pathlib import Path

def pack_xai_index(xai_dir: str) -> str:
    report = os.path.join(xai_dir, "XAI_REPORT_INDEX.json")
    entries = []
    headers = sorted(glob.glob(os.path.join(xai_dir, "*_xai_header.json")))
    for hpath in headers:
        stem = Path(hpath).stem.replace("_xai_header","")
        rec = {
            "image": stem,
            "header": os.path.basename(hpath),
            "pred_overlay": f"{stem}_pred.png" if os.path.exists(os.path.join(xai_dir, f"{stem}_pred.png")) else None,
            "rpn_heatmap":  f"{stem}_rpn_heatmap.png" if os.path.exists(os.path.join(xai_dir, f"{stem}_rpn_heatmap.png")) else None,
            "roi_heatmap":  f"{stem}_roi_heatmap.png" if os.path.exists(os.path.join(xai_dir, f"{stem}_roi_heatmap.png")) else None,
            "counterfactuals": {
                "occlusion_top":    f"{stem}_occl_top.png" if os.path.exists(os.path.join(xai_dir, f"{stem}_occl_top.png")) else None,
                "occlusion_bottom": f"{stem}_occl_bottom.png" if os.path.exists(os.path.join(xai_dir, f"{stem}_occl_bottom.png")) else None,
                "occlusion_left":   f"{stem}_occl_left.png" if os.path.exists(os.path.join(xai_dir, f"{stem}_occl_left.png")) else None,
                "occlusion_right":  f"{stem}_occl_right.png" if os.path.exists(os.path.join(xai_dir, f"{stem}_occl_right.png")) else None,
                "jitter":           f"{stem}_jitter.png" if os.path.exists(os.path.join(xai_dir, f"{stem}_jitter.png")) else None,
                "scale":            f"{stem}_scale.png" if os.path.exists(os.path.join(xai_dir, f"{stem}_scale.png")) else None,
                "cf_json":          f"{stem}_cf_summary.json" if os.path.exists(os.path.join(xai_dir, f"{stem}_cf_summary.json")) else None
            }
        }
        entries.append(rec)
    os.makedirs(xai_dir, exist_ok=True)
    with open(report, "w") as f:
        json.dump({"entries": entries}, f, indent=2)
    return report
