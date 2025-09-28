import os, json, glob
from pathlib import Path

XAI_DIR = "outputs/xai_det"
REPORT  = os.path.join(XAI_DIR, "XAI_REPORT_INDEX.json")

def main():
    entries = []
    headers = sorted(glob.glob(os.path.join(XAI_DIR, "*_xai_header.json")))
    for hpath in headers:
        stem = Path(hpath).stem.replace("_xai_header","")
        rec = {
            "image": stem,
            "header": os.path.basename(hpath),
            "pred_overlay": f"{stem}_pred.png" if os.path.exists(os.path.join(XAI_DIR, f"{stem}_pred.png")) else None,
            "rpn_heatmap":  f"{stem}_rpn_heatmap.png" if os.path.exists(os.path.join(XAI_DIR, f"{stem}_rpn_heatmap.png")) else None,
            "roi_heatmap":  f"{stem}_roi_heatmap.png" if os.path.exists(os.path.join(XAI_DIR, f"{stem}_roi_heatmap.png")) else None,
            "counterfactuals": {
                "occlusion_top":    f"{stem}_occl_top.png" if os.path.exists(os.path.join(XAI_DIR, f"{stem}_occl_top.png")) else None,
                "occlusion_bottom": f"{stem}_occl_bottom.png" if os.path.exists(os.path.join(XAI_DIR, f"{stem}_occl_bottom.png")) else None,
                "occlusion_left":   f"{stem}_occl_left.png" if os.path.exists(os.path.join(XAI_DIR, f"{stem}_occl_left.png")) else None,
                "occlusion_right":  f"{stem}_occl_right.png" if os.path.exists(os.path.join(XAI_DIR, f"{stem}_occl_right.png")) else None,
                "jitter":           f"{stem}_jitter.png" if os.path.exists(os.path.join(XAI_DIR, f"{stem}_jitter.png")) else None,
                "scale":            f"{stem}_scale.png" if os.path.exists(os.path.join(XAI_DIR, f"{stem}_scale.png")) else None,
                "cf_json":          f"{stem}_cf_summary.json" if os.path.exists(os.path.join(XAI_DIR, f"{stem}_cf_summary.json")) else None
            }
        }
        entries.append(rec)

    with open(REPORT, "w") as f:
        json.dump({"entries": entries}, f, indent=2)
    print("Wrote report index →", REPORT)

if __name__ == "__main__":
    main()
