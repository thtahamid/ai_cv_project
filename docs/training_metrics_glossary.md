# YOLOv11 Training Metrics — Glossary

## Training Progress Bar Fields

```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances  Size
  3/100      2G     0.9298      1.017      1.323        33     640: 74% 26/35  5.7s/it  2:31<51s
```

| Field | Want it... | Meaning |
|---|---|---|
| **Epoch** | — | Current epoch / max epochs (e.g. 3/100) |
| **GPU_mem** | Non-zero, stable | VRAM used. `0G` means training on CPU — bad |
| **box_loss** | Decreasing → low | Error in predicted bounding box position & size |
| **cls_loss** | Decreasing → low | Error in predicted class label |
| **dfl_loss** | Decreasing → low | Distribution Focal Loss — fine-grained box edge precision |
| **Instances** | — | Object instances in the current batch |
| **Size** | — | Input image resolution (640 = 640×640 px) |
| **Progress** | — | Batches done / total batches this epoch |
| **Speed** | Low (GPU) | Seconds per batch. GPU: ~0.3–0.8 s/it. CPU: 5+ s/it |
| **Time** | — | Elapsed time < estimated remaining for this epoch |

---

## Core Training Concepts

### Batch
A batch is a small group of images fed into the model at once before the model updates its internal weights (parameters).

Training on all images at once would require enormous memory and produce very noisy updates. Instead, training splits the dataset into small batches and updates weights after each one.

`batch=16` means 16 images are processed together per update step.

```
Total training images: 640
Batch size:            16
──────────────────────────
Batches per epoch:     640 ÷ 16 = 40
```

So the progress bar showing `26/35` means 26 of 35 batches are done for that epoch. (35 here because YOLO may drop the last incomplete batch or use a slightly different split — the exact count can vary by a few.)

**Why not use batch=1 or batch=640?**
- Too small (e.g. 1): weight updates are extremely noisy and unstable
- Too large (e.g. 640): needs massive GPU memory and tends to find worse solutions
- 8–32 is the practical sweet spot for small datasets like this one

---

### Epoch
An epoch is one complete pass through the **entire training dataset** — every image seen exactly once.

With our dataset:
```
Training images:   640
Batch size:        16
Batches per epoch: 640 ÷ 16 = 40
```

So one epoch = 40 batch updates. After each epoch, YOLO runs a full validation pass on the held-out validation images and reports the losses and mAP scores you see in the progress bar.

**Why multiple epochs?**
One pass is not enough for the model to learn. Each epoch refines the weights slightly. Over many epochs the losses decrease and accuracy improves, until the model converges (stops improving meaningfully).

**Why not train for thousands of epochs?**
At some point the model stops improving on the validation set even as training loss keeps dropping — this is overfitting (memorising the training data rather than generalising). `patience=20` catches this: if mAP doesn't improve for 20 consecutive epochs, training stops automatically regardless of the `epochs=100` ceiling.

**Rough time estimate for this project (GPU):**
```
~0.5 s/batch × 40 batches/epoch × 70 epochs ≈ ~23 minutes total
```
On CPU (~5 s/batch) the same run takes ~4 hours.

---

## Loss Definitions

### box_loss (Bounding Box Loss)
Measures how far the predicted box is from the ground-truth box — position (x, y) and scale (w, h).
- **Good range by end of training:** 0.3 – 0.6
- High value → model struggles to localize objects

### cls_loss (Classification Loss)
Measures how wrong the class prediction is (e.g., confusing "phone" with "laptop").
- **Good range by end of training:** 0.2 – 0.5
- High value → model struggles to identify the correct class

### dfl_loss (Distribution Focal Loss)
Specific to YOLOv8/v11. Refines the exact position of each box edge by treating it as a probability distribution rather than a single number. Makes boxes tighter and more precise.
- **Good range by end of training:** 0.8 – 1.1
- Starts higher than the other losses and decreases more slowly — this is normal

---

## What "Good" Training Looks Like

- All three losses **decrease smoothly** each epoch with no sudden spikes
- `val/box_loss` and `val/cls_loss` track close to the training losses (large gap = overfitting)
- `mAP@0.5` climbs past **0.80** within 60–80 epochs on an 800-image dataset
- Early stopping (`patience=20`) halts training automatically once mAP stops improving

## Red Flags

| Symptom | Likely cause |
|---|---|
| `GPU_mem: 0G` | Training on CPU — CUDA torch not installed or kernel not restarted |
| Loss plateaus early (< epoch 20) | Learning rate too high, or data issue |
| Val loss rises while train loss drops | Overfitting — reduce epochs or add augmentation |
| Loss oscillates / diverges | Learning rate too high |
