# Batch Docs — 05_training_batch_12_06am_12_05_2026

| Folder | Notebook | Contents |
|--------|----------|----------|
| `nb01_data_collection/`  | 01 | Pair counts, provenance, aggregation verification |
| `nb02_data_annotation/`  | 02 | Split tables, box counts, structural validation, spot-check grid |
| `nb03_dataset_health/`   | 03 | Class-balance chart, box-area histogram, dimension scatter, health summary |
| `nb04_model_training/`   | 04 | Training config, results.csv, loss+mAP curves, batch images, sanity predictions |
| `nb05_model_evaluation/` | 05 | Test metrics, confusion matrices, PR curves, qualitative predictions |

---

## Test Results (nb05)

| Metric | Value |
|--------|-------|
| mAP@0.5 | 0.8728 |
| mAP@0.5:0.95 | 0.8728 |
| Precision (macro) | 1.0000 |
| Recall (macro) | 0.9804 |

### Per-Class

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|-----------|--------|---------|---------------|
| projector | 1.0000 | 1.0000 | 0.9950 | 0.9377 |
| whiteboard | 1.0000 | 0.9500 | 0.9750 | 0.9413 |
| fire_extinguisher | 1.0000 | 1.0000 | 0.9950 | 0.8762 |
| door_sign | 1.0000 | 0.9714 | 0.9855 | 0.7359 |

---

## Dataset

| Split | Images | Boxes |
|-------|--------|-------|
| **test** | 80 | 112 |
| **train** | 560 | 810 |
| **val** | 160 | 229 |
| **total** | 800 | 1151 |

Classes: `projector=0`, `whiteboard=1`, `fire_extinguisher=2`, `door_sign=3`
