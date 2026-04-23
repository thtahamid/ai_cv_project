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
[1] Aggregate per class (200/class)  →  [2] Combine + Remap + Split  →  [3] Health Check
                                                                                ↓
                                                  [5] Evaluation  ←  [4] Training
```

---

## Stage 1 — Aggregate Per-Class Datasets to 200/class
**Notebook:** `notebooks/01_data_collection.ipynb`

- For each class we've downloaded one or more Roboflow/Kaggle exports under `datasets/<class>/<dataset_N>/` (each with `train/valid/test/data.yaml`). A class folder may contain **one sub-dataset or several** — the notebook walks whatever is there.
- Walk every sub-dataset and pool image/label pairs **across all splits**.
- Randomly sample **200 per class** and write them flat into `data/aggregated/<class>/{images,labels}/`, keeping the original single-class `class_id = 0`.
- Write `data/aggregated/<class>/info.json` recording which sub-dataset + split each selected file came from.

## Stage 2 — Combine Aggregated Classes into the Unified Split Dataset
**Notebook:** `notebooks/02_data_annotation.ipynb`

- Read `data/aggregated/<class>/` for all four classes.
- **Stratified 70 / 20 / 10 split** per class → every split contains every class.
- **Remap** each file's `class_id` from `0` to the global index (`projector=0`, `whiteboard=1`, `fire_extinguisher=2`, `door_sign=3`).
- Write to `data/dataset/{images,labels}/{train,val,test}/` with filenames `<class>_<split>_<nnnn>.jpg`.
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
├── datasets/       # raw per-class Roboflow/Kaggle exports (scaffold tracked, contents gitignored)
│   ├── projector/<dataset_N>/
│   ├── whiteboard/<dataset_N>/
│   ├── fire_extinguisher/<dataset_N>/
│   └── door_sign/<dataset_N>/
├── data/           # aggregated (200/class) → dataset (split)  [gitignored]
├── weights/        # trained checkpoints (gitignored)
└── runs/           # Ultralytics run logs (gitignored)
```

## Environment Setup

Create and register the virtual environment, then run notebooks in numeric order:

```bash
python -m venv .venv && source .venv/bin/activate
# Windows: .venv\Scripts\Activate.ps1

.venv/Scripts/python -m ipykernel install --user --name=ai_cv_project --display-name "Python (.venv)"
```

Each notebook has its own `%pip install` cell at the top scoped to what that stage needs. Details below.

---

## Per-Notebook Dependencies

### Notebook 01 — Data Collection

```python
%pip install -q pandas pillow pyyaml
print("nb01 dependencies ready")
```

| Package | Why |
|---|---|
| `pandas` | Tabular summary of available image/label pairs per class |
| `pillow` | Validate image file integrity before sampling |
| `pyyaml` | Parse Roboflow `data.yaml` files to discover class names and split paths |

---

### Notebook 02 — Data Annotation & Split

```python
%pip install -q pandas pillow matplotlib pyyaml
print("nb02 dependencies ready")
```

| Package | Why |
|---|---|
| `pandas` | Manage per-class split counts and filename tables |
| `pillow` | Open images for the bounding-box spot-check grid |
| `matplotlib` | Render annotated bounding box overlays during spot-check |
| `pyyaml` | Read source `data.yaml` files and write the unified `data.yaml` |

---

### Notebook 03 — Dataset Health Check

```python
%pip install -q pandas pillow matplotlib seaborn
print("nb03 dependencies ready")
```

| Package | Why |
|---|---|
| `pandas` | Build per-split class balance tables |
| `pillow` | Read image dimensions for the size scatter plot |
| `matplotlib` | Draw bar charts, scatter plots, and distribution histograms |
| `seaborn` | Styled bar chart for the class balance overview |

---

### Notebook 04 — Model Training

```python
# Training framework + plotting
%pip install -q "ultralytics==8.3.*" pandas matplotlib
print("nb04 base dependencies ready")

# CUDA PyTorch — force-reinstall so the CUDA build always wins over the
# +cpu wheel that ultralytics pulls from PyPI by default
%pip install -q --force-reinstall torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126
print("nb04 CUDA PyTorch ready — restart kernel before running further cells")
```

| Package | Why |
|---|---|
| `ultralytics==8.3.*` | YOLOv11 training, validation, and inference framework |
| `pandas` | Read `results.csv` to plot loss and mAP curves per epoch |
| `matplotlib` | Plot box/cls/dfl loss and mAP training curves |
| `torch` / `torchvision` / `torchaudio` (cu126) | GPU-accelerated tensor ops; the CUDA build is required for RTX 4060 — the default PyPI wheel is CPU-only |

---

### Notebook 05 — Evaluation

```python
%pip install -q "ultralytics==8.3.*" pandas numpy matplotlib seaborn pillow
print("nb05 dependencies ready")
```

| Package | Why |
|---|---|
| `ultralytics==8.3.*` | Run `model.val()` and `model.predict()` on the held-out test split |
| `pandas` | Build metric comparison tables across model variants |
| `numpy` | Numerical operations on raw metric arrays |
| `matplotlib` | PR curves, confusion matrix, and prediction image grids |
| `seaborn` | Styled confusion matrix heatmap |
| `pillow` | Load test images for the qualitative prediction grid |

---

## Install All Requirements at Once

Use this if you prefer to pre-install everything before opening any notebook.
After running this, restart your kernel and skip the individual `%pip install` cells.

**Step 1 — upgrade pip:**
```bash
python -m pip install --upgrade pip
```

**Step 2 — common packages:**
```bash
pip install -q pandas pillow pyyaml matplotlib seaborn numpy "ultralytics==8.3.*"
```

**Step 3 — CUDA PyTorch (GPU support):**
```bash
pip install -q --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

**Step 4 — verify GPU is detected:**
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

**Other Packages:**
```bash
pip install opencv-python numpy
```

