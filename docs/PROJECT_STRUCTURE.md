# Project Structure

This document provides an overview of the codebase layout for **UnderstandPredAI**.  
It was generated using the `tree` command for clarity.

```bash
UnderstandPredAI
|   .gitignore
|   PROJECT_STRUCTURE.md
|   pyproject.toml
|   README.md
|   requirements.txt
|   
+---.venv
|
+---configs
|   +---cls
|   |       default.yaml
|   |       test.yaml
|   |       
|   +---det
|   |       default.yaml
|   |       default_cust.yaml
|   |       default_cust_extr.yaml
|   |       test.yaml
|   |       test_cust.yaml
|   |       test_cust_extr.yaml
|   |       
|   \---xai
|           cls.yaml
|           det.yaml
|           
+---data
|   +---sized_rectangles_filled
|   |   +---annotations
|   |   |       0.xml
|   |   |       1.xml
|   |   |       2.xml
|   |   |       ...
|   |   |
|   |   +---test
|   |   |       0.bmp
|   |   |       ...
|   |   |
|   |   +---train
|   |   |       1.bmp
|   |   |       ...
|   |   |
|   |   \---val
|   |           2.bmp
|   |           ...
|   |
|   +---sized_rectangles_unfilled
|   |   +---annotations
|   |   |       0.xml
|   |   |       1.xml
|   |   |       2.xml
|   |   |       ...
|   |   |
|   |   +---test
|   |   |       0.bmp
|   |   |       ...
|   |   |
|   |   +---train
|   |   |       1.bmp
|   |   |       ...
|   |   |
|   |   \---val
|   |           2.bmp
|   |           ...
|   |
|   +---sized_squares_filled
|   |   +---annotations
|   |   |       0.xml
|   |   |       1.xml
|   |   |       2.xml
|   |   |       ...
|   |   |
|   |   +---test
|   |   |       0.bmp
|   |   |       ...
|   |   |
|   |   +---train
|   |   |       1.bmp
|   |   |       ...
|   |   |
|   |   \---val
|   |           2.bmp
|   |           ...
|   |
|   \---sized_squares_unfilled
|       +---annotations
|       |       0.xml
|       |       1.xml
|       |       2.xml
|       |       ...
|       |
|       +---test
|       |       0.bmp
|       |       ...
|       |
|       +---train
|       |       1.bmp
|       |       ...
|       |
|       \---val
|               2.bmp
|               ...
|
+---docs
|       PITSEC - T3 - Pred AI.pdf
|
+---outputs
|
+---scripts
|   |   check_data.py
|   |
|   +---cls
|   |   |   eval.py
|   |   |   explain.py
|   |   |   train.py
|   |
|   +---det
|       |   eval.py
|       |   explain.py
|       |   train.py
|
\---src
    |   __init__.py
    |
    +---constants
    |   |   norms.py
    |   |   sizes.py
    |   |   __init__.py
    |
    +---data
    |   |   cls.py
    |   |   det.py
    |   |   split.py
    |   |   voc.py
    |   |   __init__.py
    |
    +---metrics
    |   |   ar.py
    |   |   cls.py
    |   |   det.py
    |   |   __init__.py
    |
    +---models
    |   |   cls_resnet.py
    |   |   det_backbone.pth
    |   |   det_fasterrcnn.py
    |   |   __init__.py
    |
    +---transforms
    |   |   cls.py
    |   |   det.py
    |
    +---utils
    |   |   config.py
    |   |   determinism.py
    |   |   echo.py
    |   |   __init__.py
    |
    +---xai
        |   __init__.py
        |
        +---core
        |       attributes.py
        |       gradcam.py
        |       overlays.py
        |       roi_ops.py
        |       types.py
        |       __init__.py
        |
        \---reports
                pack_index.py
                __init__.py
