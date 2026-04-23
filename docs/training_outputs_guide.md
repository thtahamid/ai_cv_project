# What You Get After Training — A Simple Guide

Once `notebooks/04_model_training.ipynb` finishes, Ultralytics writes a collection of files, folders, and figures under `runs/` and `weights/`. This guide walks through every one of them in plain English: **what the file is, what's inside, and why you care.**

---

## Top-Level Layout

```
ai_cv_project/
├── weights/
│   ├── best.pt          ← the trained model you actually use
│   └── best.onnx        ← portable version (from nb07)
└── runs/
    └── runs/detect/campus_yolo11s/
        ├── weights/
        │   ├── best.pt  ← same best weights (source copy)
        │   └── last.pt  ← weights from the final epoch
        ├── args.yaml
        ├── results.csv
        ├── results.png
        ├── labels.jpg
        ├── train_batch*.jpg
        ├── val_batch*_labels.jpg
        ├── val_batch*_pred.jpg
        ├── confusion_matrix.png
        ├── confusion_matrix_normalized.png
        ├── BoxP_curve.png / BoxR_curve.png / BoxF1_curve.png / BoxPR_curve.png
        └── sanity/       ← one prediction per class, quick visual proof
```

The `runs/detect/val/` folder contains the same kinds of figures but from a **separate validation pass** (nb05) run on the held-out test set.

---

## 1. The Trained Model — `best.pt` and `last.pt`

These are the outputs that actually matter. Everything else is there to help you trust them.

### `weights/best.pt`
- **What it is:** the model checkpoint from the epoch with the **highest validation mAP**. A PyTorch file containing the network architecture + trained weights.
- **Why it matters:** this is the model you load in nb05 (evaluation), nb06 (live inference), and nb07 (ONNX export). This is "the product" of training.
- **How to use:**
  ```python
  from ultralytics import YOLO
  model = YOLO("weights/best.pt")
  results = model.predict("some_image.jpg")
  ```

### `weights/last.pt`
- **What it is:** the checkpoint from the **final epoch** training reached (either epoch 100 or where early-stopping halted).
- **Why it matters:** useful if you want to resume training (`resume=True`) or compare "final" vs. "best" to see how much overfitting happened after the peak.
- **Rule of thumb:** ignore it for deployment. Always use `best.pt`.

### `weights/best.onnx`
- **What it is:** `best.pt` re-saved in the **ONNX** format (Open Neural Network Exchange) — a framework-agnostic standard.
- **Why it matters:** lets the model run outside PyTorch (e.g., in a browser via ONNX Runtime Web, on mobile, on a C++ server, on edge hardware). Produced by `notebooks/07_export_onnx.ipynb`.

---

## 2. `args.yaml` — The Training Recipe

A plain-text YAML file recording **every single hyperparameter** the training run used: model variant, epochs, batch size, learning rate, augmentation settings, device, optimizer, loss weights, and more.

**Why it matters:**
- Reproducibility — months from now you can see exactly how `best.pt` was produced.
- Debugging — if results look off, you can compare `args.yaml` across runs to find what changed.
- Tuning — it's the starting point for your next experiment.

Key fields to know:
| Field | Meaning |
|---|---|
| `model` | Starting checkpoint (e.g., `yolo11n.pt` — transfer learning base) |
| `data` | Path to `data.yaml` describing the dataset |
| `epochs` | Max epochs (100) |
| `patience` | Early-stop patience (15 → stop if no improvement for 15 epochs) |
| `batch` | Images per update step (16) |
| `imgsz` | Input resolution (640×640) |
| `optimizer` | SGD |
| `lr0`, `lrf` | Initial / final learning rate |
| `device` | `mps` / `cuda` / `cpu` |
| `seed` | Random seed (42 → deterministic runs) |

---

## 3. `results.csv` — The Training Log (Raw Numbers)

One row per epoch. Columns:

| Column | What it tells you |
|---|---|
| `epoch` | Epoch number |
| `time` | Seconds elapsed since training started |
| `train/box_loss` | Bounding-box regression loss on training data |
| `train/cls_loss` | Classification loss on training data |
| `train/dfl_loss` | Distribution Focal Loss (fine box-edge refinement) |
| `val/box_loss`, `val/cls_loss`, `val/dfl_loss` | Same losses on the **validation** split |
| `metrics/precision(B)` | Precision over box predictions |
| `metrics/recall(B)` | Recall over box predictions |
| `metrics/mAP50(B)` | mAP at IoU 0.5 — the headline accuracy number |
| `metrics/mAP50-95(B)` | Stricter mAP averaged over IoU 0.5 → 0.95 |
| `lr/pg0`, `lr/pg1`, `lr/pg2` | Learning rates for 3 parameter groups |

**Why it matters:** it is the single source of truth for how training progressed. Every plot under `runs/` is generated from this file. You can also load it yourself in pandas to make custom charts.

(For the meaning of each loss, see `docs/training_metrics_glossary.md`.)

---

## 4. `results.png` — The Training Dashboard

A single image with **10 sub-plots** showing each column of `results.csv` over epochs.

**What to look for:**
- `train/box_loss`, `train/cls_loss`, `train/dfl_loss` → should decrease smoothly.
- `val/*_loss` → should track close to training losses. A big gap means **overfitting**.
- `metrics/mAP50` → should climb and plateau. Ideally past **0.80** by the end.
- `metrics/mAP50-95` → a stricter, lower curve. Still increasing is a good sign.

If you're going to look at **one** image to judge training, this is it.

---

## 5. Confusion Matrices

### `confusion_matrix.png`
A 4×4 grid (plus a "background" row/col) with **raw counts** of how often each true class was predicted as each class.

- Rows = true label.
- Columns = predicted label.
- Diagonal = correct predictions (want these **high**).
- Off-diagonal = confusions (want these **low**).
- The "background" column = objects the model missed entirely (false negatives).
- The "background" row = phantom detections where nothing exists (false positives).

### `confusion_matrix_normalized.png`
Same matrix but each **row sums to 1.0** — shows the percentage of each true class predicted as each class. Easier to compare classes with different amounts of data.

**Why they matter:** they tell you **which classes are getting confused for each other**. If "projector" and "whiteboard" keep swapping, that's a dataset or labeling issue, not a model issue.

---

## 6. PR / P / R / F1 Curves

These are all generated from the validation pass and show how performance varies with the **confidence threshold** you choose at inference time.

### `BoxPR_curve.png` — Precision-Recall Curve
- One curve per class + a mean curve.
- Area under the curve = **mAP@0.5**.
- A curve hugging the top-right corner = excellent. A curve collapsing to the bottom-left = bad.

### `BoxP_curve.png` — Precision vs. Confidence
- Shows: "If I require confidence ≥ X, what % of my detections are correct?"
- Higher confidence threshold → higher precision but fewer detections.

### `BoxR_curve.png` — Recall vs. Confidence
- Shows: "If I require confidence ≥ X, what % of real objects do I find?"
- Higher threshold → lower recall (we miss things).

### `BoxF1_curve.png` — F1 vs. Confidence
- F1 = balance of precision and recall.
- **The peak of this curve tells you the best confidence threshold to use at inference.**

**Why they matter:** they help you pick the right confidence threshold for deployment. If you want few false alarms, read off `BoxP_curve`. If you must catch every object, read off `BoxR_curve`. If you want balance, use the peak of `BoxF1_curve`.

---

## 7. Visual Spot-Check Images

These are image mosaics saved so you can **see** what the model saw and what it predicted — numbers only tell half the story.

### `labels.jpg`
A histogram-style plot showing the **dataset distribution**: counts per class, box position heatmap, and box size distribution. Written before training starts. Quick sanity check that your dataset is balanced.

### `train_batch0.jpg`, `train_batch1.jpg`, `train_batch2.jpg`
Three sample **training batches** as the model sees them — augmentation included (mosaic, flips, color jitter, etc.). Ground-truth boxes drawn on top.

**Use case:** if these look like garbage (boxes misaligned, wrong class colors), your labels or augmentation are broken.

### `val_batch{0,1,2}_labels.jpg` and `val_batch{0,1,2}_pred.jpg`
Three validation batches, shown **twice**:
- `_labels.jpg` → ground-truth boxes.
- `_pred.jpg` → model predictions with confidence scores.

Flip between the two side-by-side to visually judge accuracy on real validation images.

### `sanity/` folder
Four JPEGs — one per class — showing the trained model's prediction on a single representative image (`0024_projector.jpg`, `0339_whiteboard.jpg`, `0412_fire_extinguisher.jpg`, `0667_door_sign.jpg`). Written at the end of nb04 as a final "yes it works" check before moving to evaluation.

---

## 8. `runs/detect/val/` — The Evaluation Pass

This sibling folder is populated by `notebooks/05_model_evaluation.ipynb` when you run `model.val(split="test")`. It contains the **same kinds of figures** (`confusion_matrix.png`, `BoxPR_curve.png`, `val_batch*.jpg`, etc.), but computed on the held-out **test** split that the model never saw during training or early-stopping.

**Why separate:** the figures under `runs/runs/detect/campus_yolo11s/` reflect performance on the **validation** split (used to pick `best.pt`). The figures under `runs/detect/val/` reflect performance on the **test** split — a more honest estimate of real-world accuracy.

---

## Quick Reference — "Which file do I look at when I want to…"

| I want to… | Open this |
|---|---|
| Use the model for inference | `weights/best.pt` |
| Deploy outside PyTorch | `weights/best.onnx` |
| See overall training health | `results.png` |
| Get raw per-epoch numbers | `results.csv` |
| Check which classes confuse the model | `confusion_matrix_normalized.png` |
| Pick the inference confidence threshold | `BoxF1_curve.png` |
| Reproduce the exact run | `args.yaml` |
| Eyeball predictions on real images | `val_batch*_pred.jpg`, `sanity/*.jpg` |
| Report final test accuracy | `runs/detect/val/` (from nb05) |

---

## Terms Cheat-Sheet

| Term | One-line meaning |
|---|---|
| **Epoch** | One full pass over the training set |
| **Batch** | A small group of images processed before a weight update (16 here) |
| **Weights (`.pt`)** | The learned parameters of the neural network |
| **Checkpoint** | A snapshot of weights saved to disk |
| **Loss** | How wrong the model is — lower is better |
| **Precision** | Of the boxes the model predicted, what fraction are correct |
| **Recall** | Of the real objects in the image, what fraction the model found |
| **F1** | Harmonic mean of precision and recall |
| **IoU** | Intersection-over-Union — overlap between predicted and true box |
| **mAP@0.5** | Mean Average Precision at IoU ≥ 0.5 — the standard accuracy metric |
| **mAP@0.5:0.95** | Averaged mAP across IoU thresholds 0.5 → 0.95 — stricter |
| **Confusion matrix** | Per-class correct-vs-wrong prediction grid |
| **ONNX** | Portable model format for running outside PyTorch |
| **Early stopping** | Halting training automatically when mAP stops improving (`patience=15`) |
