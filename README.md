# Understanding Predictive AI

This repo contains two pipelines sharing a VOC-style dataset:
- **Object Detection**: Faster R-CNN (MobileNetV3-FPN) with size-wise metrics.
- **Classification-as-Regression**: crop around boxes and predict `(short, long)` side lengths.

## Layout
See `src/` for library code, `scripts/` for CLIs, `configs/` for YAMLs.

## Quickstart
```bash
# detection
python scripts/train_det.py --data-root <root> --output-dir outputs
python scripts/eval_det.py  --images <root>/test/images --ann <root>/annotations --checkpoint outputs/fasterrcnn_best.pt --tag test

# classification
python scripts/train_cls.py --data-root <root> --output-dir outputs
python scripts/eval_cls.py  --images <root>/test/images --ann <root>/annotations --checkpoint outputs/resnet_cls_best.pt --tag test
```
