# Batch 04 — Documentation Outputs
## `04_training_batch_1_48am_24_04_2026`

Generated: 2026-04-24 · Model: **yolo11n** (2.6M params) · Dataset: 800 images, 4 classes

---

### Folder Map

| Folder | Notebook | Contents |
|--------|----------|----------|
| `nb01_data_collection/` | 01 | Available-pairs counts, source-split provenance, aggregation verification |
| `nb02_data_annotation/` | 02 | Stratified split table, box counts, structural validation, spot-check grid, data.yaml |
| `nb03_dataset_health/`  | 03 | Class-balance bar chart, box-area histogram, dimension scatter, health summary |
| `nb04_model_training/`  | 04 | Training config, results.csv, loss+mAP curves, label dist., batch images, sanity predictions |
| `nb05_model_evaluation/`| 05 | Test metrics (overall + per-class), confusion matrices, PR curves, qualitative predictions |

---

### Key Results (Test Split — 80 images, 102 boxes)

| Metric | Value |
|--------|-------|
| mAP@0.5 | **0.9456** |
| mAP@0.5:0.95 | **0.8284** |
| Macro Precision | 0.9468 |
| Macro Recall | 0.9030 |

#### Per-Class

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|--------------|
| projector | 1.000 | 0.797 | 0.927 | 0.754 |
| whiteboard | 0.952 | 0.739 | 0.882 | 0.792 |
| fire_extinguisher | 1.000 | 0.957 | 0.978 | 0.893 |
| door_sign | 0.988 | 1.000 | 0.995 | 0.876 |

---

### Dataset

| Split | Images | Boxes | Empty Labels |
|-------|--------|-------|--------------|
| train | 560 | 695 | 25 |
| val | 160 | 200 | 7 |
| test | 80 | 102 | 3 |
| **total** | **800** | **997** | **35** |

Classes: `projector=0`, `whiteboard=1`, `fire_extinguisher=2`, `door_sign=3`

---

### Training Config

| Param | Value |
|-------|-------|
| model | yolo11n.pt |
| epochs | 100 |
| imgsz | 640 |
| batch | 16 |
| optimizer | SGD |
| lr0 | 0.01 |
| patience | 15 |
| seed | 42 |
| device | CUDA |
