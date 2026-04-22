# Campus Infrastructure Detection — Project Guide

A computer vision pipeline that trains a **YOLOv11** object detector on a custom dataset to recognize four classes of campus infrastructure:

| Class ID | Class Name         |
|----------|--------------------|
| 0        | Projector          |
| 1        | Whiteboard         |
| 2        | Fire Extinguisher  |
| 3        | Door Sign          |

The work is broken into five Jupyter notebooks under `notebooks/`, each owning one stage of the end-to-end pipeline.

---

## Pipeline Overview

```
[1] Acquisition & Inventory  →  [2] Combine + Remap + Split  →  [3] Health Check
                                                                       ↓
                                         [5] Evaluation  ←  [4] Training
```

---

## Stage 1 — Data Acquisition & Inventory
**Notebook:** `notebooks/01_data_collection.ipynb`

- Download **already-annotated** datasets from Roboflow / Kaggle — one per class — and drop each into its own folder under `data/sources/<class_name>/`.
- Each source is a single-class YOLO export, so every label inside starts with `class_id = 0`.
- Inventory the folders, verify image↔label pairing, and peek at each source's `data.yaml` (when present).

## Stage 2 — Combine Sources into the Unified Split Dataset
**Notebook:** `notebooks/02_data_annotation.ipynb`

- Walk the four per-class source folders and merge them into `data/dataset/{images,labels}/{train,val,test}/`, **honoring the Roboflow source splits** (`train` → `train`, `valid` → `val`, `test` → `test`). Flat sources fall back to `train`.
- **Remap** each source's `class_id` from `0` to the global index (`projector=0`, `whiteboard=1`, `fire_extinguisher=2`, `door_sign=3`).
- Normalize filenames to `<class>_<split>_<n>.jpg`, convert images to RGB JPG.
- Structural validation + visual spot-check + emit `data.yaml` for Ultralytics.

## Stage 3 — Dataset Health Check
**Notebook:** `notebooks/03_data_preprocessing_split.ipynb`

- Diagnostic only — no files are modified.
- Per-split class balance table and bar chart; box-size distribution per class; image-dimension scatter; empty-label and tiny-box flags; cross-split filename leakage guard.

## Stage 4 — Model Preparation & Training
**Notebook:** `notebooks/04_model_training.ipynb`

- Choose a YOLOv11 variant (`yolo11n`, `yolo11s`, or `yolo11m`) — defaults to `yolo11s` as a size/accuracy compromise.
- Configure hyperparameters: `epochs=100`, `imgsz=640`, `batch=16`, SGD, early stopping.
- Log per-epoch loss and mAP to `runs/detect/train*`.
- Save best weights to `weights/best.pt`.

## Stage 5 — Evaluation
**Notebook:** `notebooks/05_model_evaluation.ipynb`

Compute on the held-out test split:
- **Precision**, **Recall**
- **mAP@0.5**, **mAP@0.5:0.95**
- **Confusion Matrix** (with per-class analysis)
- **PR curves** per class
- **Comparison tables** (model variants, baseline vs. tuned)
- Qualitative grid of predictions on sample test images

---

## Repo Layout

```
ai_cv_project/
├── docs/
│   └── PROJECT_GUIDE.md
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_data_annotation.ipynb
│   ├── 03_data_preprocessing_split.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
├── data/           # sources (per-class) → annotated (merged) → dataset (split)  [gitignored]
├── weights/        # trained checkpoints (gitignored)
└── runs/           # Ultralytics run logs (gitignored)
```

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install ultralytics==8.3.* opencv-python pandas matplotlib seaborn scikit-learn pyyaml tqdm pillow
```

Run the notebooks in numeric order.
