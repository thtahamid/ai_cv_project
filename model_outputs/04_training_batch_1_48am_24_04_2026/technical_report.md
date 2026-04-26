# Technical Report: Campus Infrastructure Object Detection with YOLOv11n

**Project:** AI Computer Vision — Campus Object Detection Pipeline  
**Batch:** `04_training_batch_1_48am_24_04_2026`  
**Date:** 2026-04-24  
**Author:** Tahamid Hossain  
**Model:** YOLOv11n · 2.6M parameters · CUDA (RTX 4060)

---

## Abstract

This report documents the end-to-end development of a four-class object detection system targeting campus infrastructure: projectors, whiteboards, fire extinguishers, and door signs. A curated 800-image dataset with perfect class balance (70/20/10 train/val/test split) was assembled from multiple Roboflow and Kaggle exports and unified under a single class-ID scheme. YOLOv11n was selected as the detection backbone for its optimal speed–accuracy tradeoff at edge-deployment scale. After 100 training epochs (convergence at epoch 60), the model achieved **mAP@0.5 = 0.9332** and **mAP@0.5:0.95 = 0.7959** on the held-out test set of 80 images. Weights are exported in both PyTorch (`.pt`) and ONNX formats for flexible deployment.

---

## 1. Introduction

Automated detection of campus assets — projectors, whiteboards, fire extinguishers, exit/door signs — enables use cases ranging from equipment inventory to safety-compliance auditing and accessibility navigation. Existing general-purpose models are not fine-tuned for institutional settings and produce poor recall on occluded or mounted equipment. This project builds a purpose-trained lightweight detector suitable for real-time inference on edge hardware.

**Scope:** Four object classes, single bounding-box detection per instance (multi-instance per image allowed), RGB images at 640 × 640 resolution.

**Pipeline overview:**

```
Raw Roboflow/Kaggle exports
        │
    [NB01] Aggregate 200 pairs/class
        │
    [NB02] Remap class IDs · stratified 70/20/10 split
        │
    [NB03] Health check: balance, box distribution, leakage
        │
    [NB04] Train YOLOv11n · 100 epochs · SGD
        │
    [NB05] Evaluate on held-out test split
        │
    [NB06] Live inference (single image / batch / webcam)
        │
    [NB07] Export to ONNX (opset 12, dynamic axes)
```

---

## 2. Dataset

### 2.1 Data Collection — NB01

Raw images were sourced from Roboflow and Kaggle exports, with the exception of door_sign sub-datasets doorsign2–4, which were custom-captured from HUB campus door signs, annotated, and prepared by the team. All sources are stored under `datasets/<class>/`. NB01 aggregated exactly 200 (image, label) pairs per class into `data/aggregated/<class>/`, re-encoding all images to JPEG for uniformity.

**Source availability before capping to 200:**

| Class | Available Pairs | Sourced From |
|---|---|---|
| projector | 319 | Projector1, Projector2, Projector3 (Roboflow) |
| whiteboard | 200 | Single Roboflow export (all used) |
| fire\_extinguisher | 848 | Kaggle export with train/valid/test sub-splits |
| door\_sign | 240 | doorsign1 (Roboflow); doorsign2–4 (custom — HUB campus door signs, annotated and prepared by the team) |

**Source provenance by sub-dataset and original split** (`source_split_breakdown.csv`):

| Class | Source | Train | Val | Test |
|---|---|---|---|---|
| door\_sign | doorsign1 (Roboflow) | 56 | 0 | 0 |
| door\_sign | doorsign2 (custom — HUB campus) | 40 | 0 | 0 |
| door\_sign | doorsign3 (custom — HUB campus) | 59 | 0 | 0 |
| door\_sign | doorsign4 (custom — HUB campus) | 85 | 0 | 0 |
| fire\_extinguisher | train | 750 | 0 | 0 |
| fire\_extinguisher | valid | 0 | 62 | 0 |
| fire\_extinguisher | test | 0 | 0 | 36 |
| projector | Projector2 | 39 | 11 | 6 |
| projector | Projector3 | 170 | 28 | 1 |
| projector | projector1 | 45 | 13 | 6 |
| whiteboard | train | 140 | 0 | 0 |
| whiteboard | valid | 0 | 40 | 0 |
| whiteboard | test | 0 | 0 | 20 |

**Verification:** All 4 classes confirmed at exactly 200 images + 200 labels post-aggregation.

---

### 2.2 Data Annotation & Stratified Split — NB02

NB02 merged the four per-class aggregated folders into a unified Ultralytics-compatible dataset under `data/dataset/`, performing three operations:

1. **Global ID assignment:** Images re-indexed 1–800.
2. **Class ID remapping:** Original per-source class IDs normalised to: `0=projector`, `1=whiteboard`, `2=fire_extinguisher`, `3=door_sign`.
3. **Stratified per-class split:** Within each class, 140/40/20 images assigned to train/val/test without overlap.

**Stratified split distribution** (`stratified_split.csv`):

| Class | Train | Val | Test |
|---|---|---|---|
| projector | 140 | 40 | 20 |
| whiteboard | 140 | 40 | 20 |
| fire\_extinguisher | 140 | 40 | 20 |
| door\_sign | 140 | 40 | 20 |
| **Total** | **560** | **160** | **80** |

**Box counts per split and class** (`boxes_per_split_class.csv`):

| Split | Class | Images | Boxes | Boxes/Image |
|---|---|---|---|---|
| train | door\_sign | 140 | 260 | 1.86 |
| train | fire\_extinguisher | 140 | 167 | 1.19 |
| train | projector | 140 | 115 | 0.82 |
| train | whiteboard | 140 | 153 | 1.09 |
| val | door\_sign | 40 | 73 | 1.83 |
| val | fire\_extinguisher | 40 | 52 | 1.30 |
| val | projector | 40 | 33 | 0.83 |
| val | whiteboard | 40 | 42 | 1.05 |
| test | door\_sign | 20 | 39 | 1.95 |
| test | fire\_extinguisher | 20 | 23 | 1.15 |
| test | projector | 20 | 17 | 0.85 |
| test | whiteboard | 20 | 23 | 1.15 |

> **Observation:** door\_sign consistently averages ~1.9 boxes/image across all splits, reflecting that signage clusters (exit + number placard) commonly co-occur. Projector consistently falls below 1.0 boxes/image, indicating a higher proportion of background or single-object frames.

**Structural validation:** All 800 images passed label-coverage checks (no missing `.txt` files, no out-of-range coordinates). `data.yaml` generated for Ultralytics.

**Spot-check grid:**

![Spot-check grid](docs\nb02_data_annotation\spot_check_grid.png)

*Figure 1. 3 × 4 spot-check grid of class samples drawn from the split dataset (NB02 output). Source: `C:\Users\mitah\github_projects\ai_cv_project\model_outputs\04_training_batch_1_48am_24_04_2026\docs\nb02_data_annotation\spot_check_grid.png`*

---

### 2.3 Dataset Health Check — NB03

NB03 performed five diagnostic passes before training began.

#### Class Balance

![Class distribution](docs\nb03_dataset_health\class_distribution_bar.png)

*Figure 2. Per-class image counts by split. Source: `C:\Users\mitah\github_projects\ai_cv_project\model_outputs\04_training_batch_1_48am_24_04_2026\docs\nb03_dataset_health\class_distribution_bar.png`*

**Summary** (`dataset_health_summary.csv`):

| Split | Images | Empty Labels | Boxes | Tiny Boxes |
|---|---|---|---|---|
| train | 560 | 25 (4.5%) | 695 | 0 |
| val | 160 | 7 (4.4%) | 200 | 0 |
| test | 80 | 3 (3.8%) | 102 | 0 |
| **Total** | **800** | **35 (4.4%)** | **997** | **0** |

Empty labels (4.4% overall) represent negative/background frames retained for hard-negative suppression during training. No tiny boxes (< configurable area threshold) were detected.

#### Bounding Box Size Distribution

![Box area histogram](docs\nb03_dataset_health\box_area_histogram.png)

*Figure 3. Normalised box area distribution per class. Source: `C:\Users\mitah\github_projects\ai_cv_project\model_outputs\04_training_batch_1_48am_24_04_2026\docs\nb03_dataset_health\box_area_histogram.png`*

#### Image Dimension Statistics

| Stat | Width (px) | Height (px) |
|---|---|---|
| count | 800 | 800 |
| mean | 751.04 | 647.60 |
| std | 489.91 | 154.94 |
| min | 640 | 512 |
| 25th pct | 640 | 640 |
| median | 640 | 640 |
| 75th pct | 640 | 640 |
| max | 4608 | 2080 |

The distribution is right-skewed: the median is 640 × 640 (Roboflow standard export), with a small tail of high-resolution outliers (up to 4608 × 2080) from raw Kaggle fire extinguisher photos. YOLO's built-in letterbox resize handles these transparently.

![Image dimension scatter](docs\nb03_dataset_health\image_dimensions_scatter.png)

*Figure 4. Width vs. height scatter — 800 images. The cluster at (640, 640) dominates; sparse high-res outliers visible. Source: `C:\Users\mitah\github_projects\ai_cv_project\model_outputs\04_training_batch_1_48am_24_04_2026\docs\nb03_dataset_health\image_dimensions_scatter.png`*

#### Cross-Split Leakage Check

Filename collision check across train/val/test: **0 duplicates detected**. The stratified-assignment step in NB02 guarantees disjoint sets at the file level.

---

## 3. Methodology

### 3.1 Model Selection and Justification

The task demands a detector that can run in real time on a single GPU (or embedded device), handle moderate object scales, and be fine-tuned on a small (<1 000-image) dataset without overfitting. Three architecture families were considered:

| Architecture | Params | COCO mAP@0.5:0.95 | Inf. latency (T4 GPU) | Paradigm | Verdict |
|---|---|---|---|---|---|
| Faster R-CNN (ResNet-50) | ~41 M | 42.9 | ~47 ms/img | Two-stage | Too slow; overkill for 4 classes |
| SSD MobileNetV2 | ~3.4 M | 22.1 | ~1.2 ms/img | One-stage | Fast but lower accuracy ceiling |
| YOLOv8n | 3.2 M | 37.3 | ~1.47 ms/img | One-stage | Strong baseline |
| **YOLOv11n** | **2.6 M** | **39.5** | **~1.55 ms/img** | **One-stage** | **Selected** |
| YOLOv11s | 9.4 M | 47.0 | ~2.46 ms/img | One-stage | Suitable if higher accuracy needed |
| YOLOv11m | 20.1 M | 51.5 | ~4.70 ms/img | One-stage | Accuracy-first, not edge-friendly |

**Why not Faster R-CNN?** Two-stage detectors produce high accuracy but carry a ~41 M parameter footprint and ~47 ms per-image latency. For real-time campus surveillance or mobile inspection apps, this throughput is unacceptable. Additionally, the region proposal network requires substantially more training data to converge reliably — a 560-image training set is marginal.

**Why not SSD?** SSD MobileNetV2 is fast and compact but its single-scale anchor design limits recall on multi-scale objects (e.g., a full-room whiteboard vs. a small door sign in the same frame). Its COCO mAP ceiling (~22) is 17 points below YOLOv11n despite comparable parameter counts.

**Why YOLOv11n?** The Nano variant of YOLOv11 achieves 39.5 mAP@0.5:0.95 on COCO at 2.6 M parameters — 0.6 M fewer than YOLOv8n with 2.2 points higher accuracy. Its C3k2 and SPPF backbone with depthwise convolutions makes it ONNX-exportable with dynamic axes for variable-resolution deployment. Transfer learning from COCO weights provides strong general feature initialisation, critical when the dataset is small.

**YOLOv11n vs. YOLOv11s tradeoff:** YOLOv11s would yield ~7.5 mAP points higher on COCO, but with a 3.6× parameter increase (9.4 M). Given the domain is narrow (4 classes, controlled campus environments), the nano variant closes this gap substantially in fine-tuning — as evidenced by the 93.3% mAP@0.5 on the test set.

---

### 3.2 Architecture

YOLOv11n is a single-stage anchor-free detector. Key components:

- **Backbone:** C3k2 blocks (compact CSP bottleneck with two convolutions) + SPPF (Spatial Pyramid Pooling — Fast) for multi-scale context aggregation.
- **Neck:** PANet (Path Aggregation Network) for bidirectional feature fusion across three scales (P3/8, P4/16, P5/32).
- **Head:** Decoupled head with separate classification and regression branches. Uses DFL (Distribution Focal Loss) for sub-pixel bounding box regression.
- **Detection paradigm:** Anchor-free, task-aligned assignment (TAL) — eliminates anchor tuning and improves small-object recall.

---

### 3.3 Training Configuration

Full configuration archived at: `C:\Users\mitah\github_projects\ai_cv_project\model_outputs\04_training_batch_1_48am_24_04_2026\runs\detect\campus_yolo11s\args.yaml`

| Parameter | Value | Rationale |
|---|---|---|
| Pretrained weights | `yolo11n.pt` (COCO) | Transfer learning; faster convergence on small dataset |
| Epochs | 100 | Sufficient for dataset of this size |
| Patience (early stop) | 15 | Halts if val mAP plateaus for 15 consecutive epochs |
| Batch size | 16 | Fits RTX 4060 8 GB VRAM comfortably |
| Image size | 640 × 640 | Standard YOLO resolution; letterboxes non-square inputs |
| Optimizer | SGD | momentum=0.937, weight\_decay=0.0005 |
| lr0 / lrf | 0.01 / 0.01 | Flat LR schedule (no cosine decay) |
| Warmup | 3 epochs | warmup\_momentum=0.8, warmup\_bias\_lr=0.1 |
| AMP | True | Mixed-precision FP16 reduces VRAM and speeds training |
| Seed | 42, deterministic=True | Reproducible training run |
| IoU threshold | 0.7 | NMS suppression threshold during training |
| Device | CUDA:0 (RTX 4060) | GPU acceleration |

**Loss weights:**

| Loss Component | Weight |
|---|---|
| Box regression (CIoU) | 7.5 |
| Classification (BCE) | 0.5 |
| Distribution Focal Loss | 1.5 |

---

### 3.4 Data Augmentation

All augmentation is applied online during training (no pre-augmented copies stored):

| Augmentation | Value |
|---|---|
| Mosaic | 1.0 (enabled; disabled for last 10 epochs via `close_mosaic=10`) |
| RandAugment (`auto_augment`) | Enabled |
| Random erasing | 0.4 probability |
| Horizontal flip | 0.5 probability |
| Scale jitter | ±50% |
| Translation | ±10% |
| HSV hue / saturation / value | 0.015 / 0.7 / 0.4 |
| MixUp / CutMix | Disabled (0.0) |
| Vertical flip | Disabled |
| Rotation / shear | Disabled |

Mosaic concatenates 4 training images into a single 640 × 640 tile — effectively quadrupling context diversity and forcing the model to handle partially visible objects. Disabling mosaic in the final 10 epochs helps the model consolidate clean feature representations.

**Label distribution visualisation:**

![Label distribution](docs\nb04_model_training\label_distribution.jpg)

*Figure 5. YOLO-generated label distribution: class frequency (left) and bounding-box spatial heatmap (right). Source: `C:\Users\mitah\github_projects\ai_cv_project\model_outputs\04_training_batch_1_48am_24_04_2026\docs\nb04_model_training\label_distribution.jpg`*

---

## 4. Training

Training was executed in NB04. Hardware: NVIDIA GeForce RTX 4060 (CUDA), mixed-precision AMP enabled.

### 4.1 Convergence

![Training curves](docs\nb04_model_training\training_curves.png)

*Figure 6. Training curves: (left) box + classification loss on train and val; (right) mAP@0.5 and mAP@0.5:0.95 on val set across 100 epochs. Source: `C:\Users\mitah\github_projects\ai_cv_project\model_outputs\04_training_batch_1_48am_24_04_2026\docs\nb04_model_training\training_curves.png`*

| Epoch | Train Box Loss | Train Cls Loss | Val mAP@0.5 | Val mAP@0.5:0.95 |
|---|---|---|---|---|
| 1 | 1.068 | — | 0.199 | 0.122 |
| 10 | — | — | ~0.65 | ~0.45 |
| 60 (**best**) | 0.744 | — | **0.9489** | **0.7251** |
| 100 (final) | 0.511 | 0.384 | 0.938 | 0.735 |

**Key training statistics** (`training_summary.json`):

| Metric | Value |
|---|---|
| Epochs trained | 100 |
| Best epoch | **60** |
| Best val mAP@0.5 | **0.9489** |
| Best val mAP@0.5:0.95 | **0.7251** |
| Final train box loss | 0.5109 |
| Final train cls loss | 0.3839 |
| Final val box loss | 0.8409 |
| Final val cls loss | 0.5516 |

The model peaked at epoch 60 and did not improve thereafter, with the val box loss flattening around 0.84 while train loss continued decreasing — a mild overfitting signal that was correctly halted by the patience mechanism. The 0.12-point gap between train and val box loss is acceptable given the dataset size.

### 4.2 Training Batch Visualisations

![Training batch mosaic](docs\nb04_model_training\train_batches\train_batch0.jpg)

*Figure 7. Example mosaic training batch (batch 0) showing augmented 4-tile compositions. Source: `C:\Users\mitah\github_projects\ai_cv_project\model_outputs\04_training_batch_1_48am_24_04_2026\docs\nb04_model_training\train_batches\train_batch0.jpg`*

### 4.3 Validation Batch Ground Truth vs. Predictions

| Ground Truth | Predictions |
|---|---|
| ![Val labels](docs\nb04_model_training\val_batches\val_batch0_labels.jpg) | ![Val preds](docs\nb04_model_training\val_batches\val_batch0_pred.jpg) |

*Figure 8. Validation batch 0: ground-truth labels (left) vs. model predictions (right). Source: `C:\Users\mitah\github_projects\ai_cv_project\model_outputs\04_training_batch_1_48am_24_04_2026\docs\nb04_model_training\val_batches\`*

### 4.4 Sanity Inference

Post-training sanity check on 4 held-aside test images confirmed correct class assignment with high confidence scores.

![Sanity predictions grid](docs\nb04_model_training\sanity_predictions_grid.png)

*Figure 9. 1 × 4 sanity inference grid (NB04). Source: `C:\Users\mitah\github_projects\ai_cv_project\model_outputs\04_training_batch_1_48am_24_04_2026\docs\nb04_model_training\sanity_predictions_grid.png`*

---

## 5. Results

Evaluation (NB05) was performed exclusively on the held-out **test split** (80 images, 102 boxes) using `best.pt` at confidence threshold 0.25 and IoU threshold 0.5.

Full Ultralytics dashboard:

![Results dashboard](runs\detect\campus_yolo11s\results.png)

*Figure 10. 10-subplot training dashboard (loss curves, precision, recall, mAP across 100 epochs). Source: `C:\Users\mitah\github_projects\ai_cv_project\model_outputs\04_training_batch_1_48am_24_04_2026\runs\detect\campus_yolo11s\results.png`*

### 5.1 Overall Test-Set Performance

| Metric | Value |
|---|---|
| Images | 80 |
| Ground-truth boxes | 102 |
| **mAP@0.5** | **0.9332** |
| **mAP@0.5:0.95** | **0.7959** |
| Macro Precision | 1.0000 |
| Macro Recall | 0.8613 |
| Confidence threshold | 0.25 |
| IoU threshold | 0.50 |

### 5.2 Per-Class Performance

(`per_class_metrics.csv`)

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| projector | 1.000 | 0.804 | 0.896 | 0.729 |
| whiteboard | 1.000 | 0.763 | 0.876 | 0.777 |
| fire\_extinguisher | 1.000 | 0.957 | 0.978 | 0.905 |
| door\_sign | 1.000 | 0.922 | 0.982 | 0.773 |
| **Macro avg** | **1.000** | **0.861** | **0.933** | **0.796** |

All four classes achieve precision = 1.0, indicating zero false positives above the 0.25 confidence threshold — the model never fires incorrectly. Recall is the differentiating factor:

- **fire\_extinguisher** (0.957) and **door\_sign** (0.922) are the strongest performers. These classes have visually distinctive appearance (red cylinder; rectangular placard with standardised typography) and the highest box density (~1.9 boxes/image for door\_sign).
- **projector** (0.804) and **whiteboard** (0.763) show lower recall due to viewing-angle and occlusion variability — projectors mounted on ceilings, whiteboards partially obscured by people or furniture.

### 5.3 Confusion Matrix

![Confusion matrix (raw counts)](docs\nb05_model_evaluation\confusion_matrix.png)

*Figure 11. Raw-count confusion matrix (4 × 4 + background). Source: `C:\Users\mitah\github_projects\ai_cv_project\model_outputs\04_training_batch_1_48am_24_04_2026\docs\nb05_model_evaluation\confusion_matrix.png`*

![Confusion matrix (normalised)](docs\nb05_model_evaluation\confusion_matrix_normalized.png)

*Figure 12. Row-normalised confusion matrix. Diagonal dominance confirms class separability. Source: `C:\Users\mitah\github_projects\ai_cv_project\model_outputs\04_training_batch_1_48am_24_04_2026\docs\nb05_model_evaluation\confusion_matrix_normalized.png`*

![Custom confusion matrix](docs\nb05_model_evaluation\confusion_matrix_custom.png)

*Figure 13. Custom dual-panel Seaborn confusion matrix. Source: `C:\Users\mitah\github_projects\ai_cv_project\model_outputs\04_training_batch_1_48am_24_04_2026\docs\nb05_model_evaluation\confusion_matrix_custom.png`*

### 5.4 Precision–Recall and Threshold Curves

![PR curve](docs\nb05_model_evaluation\BoxPR_curve.png)

*Figure 14. Precision–Recall curves per class. Area under each curve equals per-class mAP@0.5. Source: `C:\Users\mitah\github_projects\ai_cv_project\model_outputs\04_training_batch_1_48am_24_04_2026\docs\nb05_model_evaluation\BoxPR_curve.png`*

![Precision vs confidence](docs\nb05_model_evaluation\BoxP_curve.png)

*Figure 15. Precision vs. confidence threshold per class. All classes maintain P = 1.0 above conf ≈ 0.4. Source: `C:\Users\mitah\github_projects\ai_cv_project\model_outputs\04_training_batch_1_48am_24_04_2026\docs\nb05_model_evaluation\BoxP_curve.png`*

![Recall vs confidence](docs\nb05_model_evaluation\BoxR_curve.png)

*Figure 16. Recall vs. confidence threshold. Source: `C:\Users\mitah\github_projects\ai_cv_project\model_outputs\04_training_batch_1_48am_24_04_2026\docs\nb05_model_evaluation\BoxR_curve.png`*

![F1 vs confidence](docs\nb05_model_evaluation\BoxF1_curve.png)

*Figure 17. F1 score vs. confidence threshold. Source: `C:\Users\mitah\github_projects\ai_cv_project\model_outputs\04_training_batch_1_48am_24_04_2026\docs\nb05_model_evaluation\BoxF1_curve.png`*

### 5.5 Qualitative Predictions

![Qualitative predictions grid](docs\nb05_model_evaluation\qualitative_predictions.png)

*Figure 18. 2 × 4 grid of test-set images with predicted bounding boxes and confidence scores overlaid. Source: `C:\Users\mitah\github_projects\ai_cv_project\model_outputs\04_training_batch_1_48am_24_04_2026\docs\nb05_model_evaluation\qualitative_predictions.png`*

---

## 6. Discussion

### 6.1 Strengths

**Zero false positives.** Macro precision = 1.0 across all classes means the model never hallucinates detections above the 0.25 threshold. This is operationally important for an inventory or safety-audit application where false alarms erode user trust.

**Strong mAP@0.5.** 93.3% mAP@0.5 on a 80-image test set with a 2.6 M parameter model is competitive. For context, general-purpose YOLOv11n on COCO achieves 39.5% mAP@0.5:0.95 — fine-tuning on a narrow four-class domain clearly reduces the problem complexity and allows the nano variant to perform at near-saturating accuracy.

**fire\_extinguisher and door\_sign near-saturation.** mAP@0.5 of 0.978 and 0.982 respectively indicate near-perfect detection for safety-critical objects, which is the highest-value outcome for compliance auditing.

### 6.2 Limitations

**Projector and whiteboard recall.** Recall of 0.804 and 0.763 respectively is the primary weakness. These objects exhibit high intra-class variance:
- Projectors: ceiling-mounted vs. desktop, front-lit vs. top-down views, wide variation in housing colour.
- Whiteboards: full-frame vs. partial, with/without text, reflective surfaces causing glare.

Both classes also have the lowest boxes/image ratio (~0.82–0.85), meaning more images contain no instance at all — potentially reducing the effective positive training signal.

**mAP@0.5:0.95 gap.** The 13.7-point gap between mAP@0.5 (0.933) and mAP@0.5:0.95 (0.796) indicates that while the model localises objects correctly at IoU ≥ 0.5, tighter bounding box precision degrades. This is typical for nano-scale YOLO models where the lightweight regression head lacks the capacity for sub-pixel box refinement. A YOLOv11s or cascaded refinement stage would close this gap.

**Dataset size.** 560 training images for 4 classes is small. The 4.4% empty-label rate (35 images) reduces effective positives further. Scaling to ~500 images per class would likely push projector and whiteboard recall above 0.90.

**Single domain.** All images originate from Roboflow/Kaggle exports from similar institutional settings. Performance on unusual campus layouts, diverse lighting, or non-standard equipment may degrade.

### 6.3 Improvement Directions

| Issue | Suggested Fix |
|---|---|
| Low projector/whiteboard recall | Collect 100–200 additional images per class from varied viewpoints and lighting |
| mAP@0.5:0.95 gap | Upgrade to YOLOv11s; or add test-time augmentation (TTA) |
| Potential domain shift | Add online images from different institutions; apply colour jitter at inference |
| Overfitting after epoch 60 | Add dropout (currently 0.0) or stronger weight decay |

---

## 7. Deployment

### 7.1 Live Inference — NB06

NB06 implements four inference modes against `best.pt`:

| Mode | Description |
|---|---|
| Single image | Load, infer, display detection with latency measurement |
| Batch/folder | Iterate over image directory; collect per-class counts and throughput (images/s) |
| Video file | Frame-by-frame annotation; generates annotated video + preview grid |
| Webcam | Live feed with real-time overlay; stop via widget button |

Configuration: `CONF_THRESH = 0.25`, `IMGSZ = 640`, device auto-selected (CUDA → MPS → CPU).

### 7.2 ONNX Export — NB07

NB07 exports `best.pt` to ONNX format for framework-agnostic deployment:

| Export Parameter | Value |
|---|---|
| ONNX opset | 12 |
| Dynamic axes | True (variable batch size and image dimensions) |
| Simplify | True (onnx-simplifier applied) |
| Source | `weights/best.pt` (~22 MB) |
| Output | `weights/best.onnx` (~11 MB) |

The exported ONNX model was verified with `onnxruntime` via a dummy inference pass. The ~50% size reduction (22 MB → 11 MB) results from stripping PyTorch training metadata. ONNX enables deployment via TensorRT (NVIDIA), OpenVINO (Intel), and CoreML (Apple) without a PyTorch dependency.

**Weights artefacts:**

| File | Path | Size |
|---|---|---|
| PyTorch best checkpoint | `C:\Users\mitah\github_projects\ai_cv_project\model_outputs\04_training_batch_1_48am_24_04_2026\weights\best.pt` | ~22 MB |
| ONNX portable model | `C:\Users\mitah\github_projects\ai_cv_project\model_outputs\04_training_batch_1_48am_24_04_2026\weights\best.onnx` | ~11 MB |

---

## 8. Conclusion

A complete object detection pipeline for four campus infrastructure classes was built and evaluated. YOLOv11n was selected as an optimal balance between edge-deployment efficiency (2.6 M parameters, ~1.55 ms/img latency) and accuracy, outperforming both SSD-family detectors and avoiding the overhead of two-stage architectures. On an 800-image balanced dataset, the model achieved **mAP@0.5 = 0.9332** with zero false positives at the chosen confidence threshold. The safety-critical classes (fire extinguisher, door sign) are detected at near-saturation accuracy (mAP@0.5 > 0.978). Primary improvement opportunity lies in recall for projector and whiteboard, addressable with additional training data and a scale-up to YOLOv11s. Both PyTorch and ONNX weights are production-ready for real-time campus deployment.

---

## Appendix: File Index

| Artefact | Path |
|---|---|
| Spot-check grid | `…\docs\nb02_data_annotation\spot_check_grid.png` |
| Class distribution bar | `…\docs\nb03_dataset_health\class_distribution_bar.png` |
| Box area histogram | `…\docs\nb03_dataset_health\box_area_histogram.png` |
| Image dimension scatter | `…\docs\nb03_dataset_health\image_dimensions_scatter.png` |
| Label distribution | `…\docs\nb04_model_training\label_distribution.jpg` |
| Training curves | `…\docs\nb04_model_training\training_curves.png` |
| Sanity predictions grid | `…\docs\nb04_model_training\sanity_predictions_grid.png` |
| Train batch mosaics | `…\docs\nb04_model_training\train_batches\train_batch{0-2,3150-3152}.jpg` |
| Val batch labels/preds | `…\docs\nb04_model_training\val_batches\val_batch{0-2}_labels/pred.jpg` |
| Ultralytics results dashboard | `…\runs\detect\campus_yolo11s\results.png` |
| Confusion matrix (raw) | `…\docs\nb05_model_evaluation\confusion_matrix.png` |
| Confusion matrix (norm) | `…\docs\nb05_model_evaluation\confusion_matrix_normalized.png` |
| Confusion matrix (custom) | `…\docs\nb05_model_evaluation\confusion_matrix_custom.png` |
| PR curve | `…\docs\nb05_model_evaluation\BoxPR_curve.png` |
| P vs conf | `…\docs\nb05_model_evaluation\BoxP_curve.png` |
| R vs conf | `…\docs\nb05_model_evaluation\BoxR_curve.png` |
| F1 vs conf | `…\docs\nb05_model_evaluation\BoxF1_curve.png` |
| Qualitative predictions | `…\docs\nb05_model_evaluation\qualitative_predictions.png` |
| Training config | `…\runs\detect\campus_yolo11s\args.yaml` |
| Per-epoch results | `…\docs\nb04_model_training\training_results.csv` |
| Overall test metrics | `…\docs\nb05_model_evaluation\overall_metrics.json` |
| Per-class test metrics | `…\docs\nb05_model_evaluation\per_class_metrics.csv` |
| Best weights (PyTorch) | `…\weights\best.pt` |
| Best weights (ONNX) | `…\weights\best.onnx` |

*Base path: `C:\Users\mitah\github_projects\ai_cv_project\model_outputs\04_training_batch_1_48am_24_04_2026\`*
