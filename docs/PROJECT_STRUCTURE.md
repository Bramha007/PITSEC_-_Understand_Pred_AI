# Project Structure

This document provides an overview of the codebase layout for **PITSEC_-_Understand_Pred_AI**.  
It was generated using the `tree` command for clarity.

```bash
PITSEC_-_Understand_Pred_AI
+---.venv
+---configs
|   +---cls
|   |       default.yaml
|   |       test.yaml   
|   +---det
|   |       default.yaml
|   |       default_cust.yaml
|   |       test.yaml
|   |       test_cust.yaml
|
+---data
|   +---sized_rectangles_filled
|   |   +---annotations
|   |   |       0.xml
|   |   |       1.xml
|   |   |       2.xml
|   |   |       ...
|   |   +---test
|   |   |       0.bmp
|   |   |       ...
|   |   +---train
|   |   |       1.bmp
|   |   |       ...
|   |   \---val
|   |           2.bmp
|   |           ...
|   +---sized_rectangles_unfilled
|   |   +---annotations
|   |   |       0.xml
|   |   |       1.xml
|   |   |       2.xml
|   |   |       ...
|   |   +---test
|   |   |       0.bmp
|   |   |       ...
|   |   +---train
|   |   |       1.bmp
|   |   |       ...
|   |   \---val
|   |           2.bmp
|   |           ...
|   +---sized_squares_filled
|   |   +---annotations
|   |   |       0.xml
|   |   |       1.xml
|   |   |       2.xml
|   |   |       ...
|   |   +---test
|   |   |       0.bmp
|   |   |       ...
|   |   +---train
|   |   |       1.bmp
|   |   |       ...
|   |   \---val
|   |           2.bmp
|   |           ...
|   \---sized_squares_unfilled
|       +---annotations
|       |       0.xml
|       |       1.xml
|       |       2.xml
|       |       ...
|       +---test
|       |       0.bmp
|       |       ...
|       +---train
|       |       1.bmp
|       |       ...
|       \---val
|               2.bmp
|               ...
+---docs
|       PITSEC_-_T3_-_Pred_AI.pdf
|       PROJECT_STRUCTURE.md
|       README.md
+---outputs
+---scripts
|   |   check_data.py
|   +---cls
|   |   |   explain.py
|   |   |   test.py
|   |   |   train.py
|   +---det
|   |   |   explain.py
|   |   |   test.py
|   |   |   train.py
|   |
|
\---src
|   +---constants
|   |   |   norms.py
|   |   |   sizes.py
|   |
|   +---data
|   |   |   cls.py
|   |   |   det.py
|   |   |   split.py
|   |   |   voc.py
|   |
|   +---metrics
|   |   |   ar.py
|   |   |   cls.py
|   |   |   det.py
|   |
|   +---models
|   |   |   cls_resnet.py
|   |   |   det_backbone.pth
|   |   |   det_fasterrcnn.py
|   |
|   +---transforms
|   |   |   cls.py
|   |   |   det.py
|   |
|   +---utils
|   |   |   determinism.py
|   |   |   echo.py
|   |
|   +---xai
|   |   |   gradcam.py
|   |   |   integrated_gradients.py
|   |   |   occlusion.py
|   |   |   rise.py
|   |   |   scorecam.py
|   |   |   utils_vis.py
|   |
|
```