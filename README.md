# Campus Infrastructure Object Detection with YOLOv11n

**Canadian University Dubai** · BCS 407 Artificial Intelligence · Spring 2026  
Instructor: Dr. Najla Al Futaisi

End-to-end YOLOv11n pipeline detecting **projectors, whiteboards, fire extinguishers, and door signs** across campus — 800 balanced images, 100 epochs on CUDA RTX 4060.

| mAP@0.5 | mAP@0.5:0.95 | Precision | Recall |
|---------|-------------|-----------|--------|
| 93.32% | 79.59% | 1.000 | 86.13% |

---

## Group Members

| Name | ID |
|------|----|
| Tahamid Hossain | 20220001801 |
| Sumaid Bin Omar | 20220001454 |
| Parth Aggarwal | 20220001200 |
| Rufaid Bin Omar | 20230002171 |
| Arham Bin Azad | 20220001121 |

---

## Dataset

800 images · 200 per class · stratified 70/20/10 split (560 train / 160 val / 80 test)  
Sources: Roboflow Universe + Kaggle, YOLO-format annotations.

---

## Pipeline

Run notebooks in order:

| # | Notebook | Purpose |
|---|----------|---------|
| 01 | `data_collection` | Sample 200 images per class |
| 02 | `data_annotation` | Stratified split + class remapping |
| 03 | `data_preprocessing_split` | Health diagnostics (read-only) |
| 04 | `model_training` | Train YOLOv11n, saves `weights/best.pt` |
| 05 | `model_evaluation` | Test-set metrics, confusion matrix, PR curves |
| 06 | `live_inference` | Inference on image / video / webcam |
| 07 | `export_onnx` | Export `best.pt` → `best.onnx` |
| 08 | `export_docs` | Build interactive dashboard |

---

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install ultralytics==8.3.* pandas pillow matplotlib seaborn pyyaml
```

Place per-class Roboflow/Kaggle exports under `datasets/<class>/`, then run notebooks 01–08.  
Notebook 04 auto-detects CUDA → MPS → CPU.

---

## Results & Docs

- Interactive dashboard: [`docs/index.html`](docs/index.html)
- Pipeline guide: [`docs/PROJECT_GUIDE.md`](docs/PROJECT_GUIDE.md)
- Metrics glossary: [`docs/training_metrics_glossary.md`](docs/training_metrics_glossary.md)
