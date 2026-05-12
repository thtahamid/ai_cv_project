# Further Improvements via Hyperparameter Tuning

**Project:** AI Computer Vision — Project 2 follow-up
**Author:** Tahamid Hossain
**Date:** 2026-05-12
**Status:** Proposed experiment plan (no runs executed yet)
**Baseline for all experiments below:** **Batch 06** — YOLOv11s + v2 dataset, the final combined Project 2 model.

---

## 1. Why this document exists

Batch 06 is the strongest model the project has produced on the strict mAP@0.5:0.95 metric (0.8808), and the deployment pipeline brings YOLOv11s inside real-time (20–30 ms / 30–35 FPS) — see `technical_report.md` §7. But three concrete weaknesses remain, and every one of them is **hyperparameter-shaped**, not data-shaped:

| Symptom | Where seen | Hypothesised root cause |
|---|---|---|
| `door_sign` precision broke its perfect streak (1.000 → 0.9697) | Batch 06 §6.2 | YOLOv11s has 3.6× the capacity of YOLOv11n; the *training recipe* hasn't been adjusted to absorb it (no extra regularisation). |
| `door_sign` recall dropped 5.7 pp on the strict bucket | Batch 06 §6.2 | Smallest objects in the test split. Training at `imgsz=640` discards too much pixel detail; the recipe never tells the model to focus on small objects. |
| Generalisation gap re-widened (Batch 05 → 06: 0.11 → 0.23) | Batch 06 §5.3 | Same recipe re-applied to a larger model overfits more aggressively. Needs LR / WD / augmentation re-tuning. |

The Project 2 brief lists hyperparameter optimisation as Category A #3. This document is the explicit follow-up plan: which knobs to turn, what we expect each to buy, and the order to attempt them.

---

## 2. Tuning principles for this project

A few non-obvious rules that should govern every experiment below:

1. **One-variable-at-a-time first, joint search second.** The project's existing single-variable runs (Batch 04 → 05 → 06) make every gain causally attributable. Burning the next two batches on isolated sweeps preserves that clean attribution before any wider Bayesian search.
2. **Optimise the *strict* metric (mAP@0.5:0.95), not mAP@0.5.** Batch 05 already saturated mAP@0.5 at 0.9876; there is essentially no headroom on the loose bucket. Tracking mAP@0.5 will reward noise.
3. **Watch the val/train gap, not just val accuracy.** Batch 06 showed that a smaller val loss can coexist with a re-widening gap, which is the leading indicator of over-fit. The decision criterion for "this hyperparameter helped" should include the gap.
4. **Hold the test split sacred.** All sweeps tune against the *validation* split. Test-set numbers are only computed once per final candidate to avoid leakage.
5. **Budget every sweep.** Each batch run is ~25 min on the RTX 4060; ten configs is ~4 hours. Plan sweeps in slices of 6–10 runs, not 50.

---

## 3. Proposed experiments

Grouped by category, each entry includes the **search range**, the **expected effect**, the **specific Batch 06 failure it targets**, and the **decision criterion**.

### 3.1 Input resolution

| Param | Current | Proposed sweep | Targets |
|---|---|---|---|
| `imgsz` | 640 | **{640, 768, 896, 1024}** | `door_sign` small-object recall + strict-IoU drift |

**Why.** Door signs in the test split have a long-edge of ~40 px at 640. Raising `imgsz` to 896 ≈ doubles the spatial information per object. Above 1024 the gains usually flatten and training cost climbs.

**Cost.** ~1.5× wall-clock per epoch at 896, ~2.5× at 1024.

**Decision criterion.** Best `door_sign` mAP@0.5:0.95 on val *without* precision falling below 0.99 on any other class.

### 3.2 Optimiser & learning-rate schedule

| Param | Current | Proposed sweep | Targets |
|---|---|---|---|
| Optimiser | SGD (mom 0.937) | **{SGD, AdamW}** | Small-dataset stability |
| `lr0` | 0.01 | **{0.001, 0.003, 0.005, 0.01, 0.02}** | Generalisation gap |
| `lrf` (final-lr fraction) | 0.01 | **{0.001, 0.01, 0.1}** | Convergence to flatter optimum |
| Warmup epochs | 3 (default) | **{3, 5, 10}** | Early-training stability with bigger backbone |
| Cosine vs linear | cosine | **cosine vs cosine-restart** | Escape local minima |

**Why.** For small datasets (~560 train images) AdamW with `lr0` ≈ 1e-3 is often more stable than SGD with `lr0` = 0.01. Lowering `lrf` extends the cool-down phase, which encourages flatter minima — directly relevant to Batch 06's overfit signature.

**Coupling note.** SGD and AdamW have *different* optimal `lr0` ranges; never compare them at the same `lr0`. Run them as two separate sub-sweeps.

**Decision criterion.** Lower train/val box-loss gap at equal or better val mAP@0.5:0.95.

### 3.3 Regularisation

| Param | Current | Proposed sweep | Targets |
|---|---|---|---|
| Weight decay | 5e-4 | **{2e-4, 5e-4, 1e-3, 2e-3}** | YOLOv11s overcapacity |
| Label smoothing | 0.0 | **{0.0, 0.05, 0.1}** | `door_sign` over-confident wrong calls |
| Dropout (head) | 0.0 | **{0.0, 0.1, 0.2}** | Generalisation gap |
| EMA decay | 0.9999 (default) | **{0.9999, 0.99975, 0.9995}** | Final-epoch stability |

**Why.** Label smoothing 0.05 is the single most cited fix for the "confident-but-wrong" pattern the Batch 06 `door_sign` precision regression shows. A weight-decay bump to 1e-3 specifically counters the 3.6× capacity jump from YOLOv11n.

**Decision criterion.** `door_sign` precision back at 1.000 without dropping `whiteboard` recall below 1.000.

### 3.4 Augmentation

| Param | Current | Proposed sweep | Targets |
|---|---|---|---|
| Mosaic close epoch | last 10 ep | **{10, 15, 20}** | More stable final epochs |
| `degrees` | 0 | **{0, 5, 10, 15}** | Perspective robustness for door signs |
| `perspective` | 0 | **{0, 0.0005, 0.001}** | Same — door signs at off-axis viewing angles |
| `scale` | 0.5 (default) | **{0.5, 0.75, 0.9}** | Small-object scale jitter |
| `mixup` | 0.0 | **{0.0, 0.1, 0.15}** | Generalisation gap |
| `copy_paste` | 0.0 | **{0.0, 0.1, 0.3}** | Multi-instance scenes + class balance |
| `erasing` | 0.4 | **{0.2, 0.4}** | Avoid masking small targets |
| HSV-S | 0.7 | **{0.4, 0.7}** | Door-sign colour-sensitivity check |

**Why.** Mixup + copy_paste are the project's untouched levers. Mixup is the classical regulariser for the "small dataset + big model" regime; copy_paste directly addresses the `door_sign` recall problem by manufacturing more multi-instance examples without new captures. Geometric augmentation (`degrees`, `perspective`) is the textbook fix for the precision wobble.

**Coupling note.** `mosaic` and `mixup` interact — never sweep both to 1.0 simultaneously.

**Decision criterion.** `door_sign` recall ≥ 0.94 *and* precision ≥ 0.99.

### 3.5 Loss weights

| Param | Current | Proposed sweep | Targets |
|---|---|---|---|
| `box` | 7.5 | **{5.0, 7.5, 10.0}** | Strict-IoU localisation |
| `cls` | 0.5 | **{0.3, 0.5, 1.0}** | Precision/recall trade |
| `dfl` | 1.5 | **{1.0, 1.5, 2.0}** | Bounding-box edge sharpness |

**Why.** Raising the box-loss weight directly pushes the model toward tighter regression — the most direct hyperparameter answer to "mAP@0.5:0.95 lags mAP@0.5". Lowering `cls` slightly trades a tiny precision drop for recall, which is the right direction for the smaller classes.

**Decision criterion.** Combined macro mAP@0.5:0.95 lift of at least 0.5 pp over Batch 06.

### 3.6 Multi-scale training (Cat. C #10, the textbook form)

| Param | Current | Proposed | Targets |
|---|---|---|---|
| `multi_scale` | False | **True** | All sizes |

**Why.** Project 2's C #10 was implemented at the *dataset-capture* layer (multi-scale subject distances), not the *training-loop* layer. Setting Ultralytics' `multi_scale=True` randomly varies `imgsz` by ±50 % each batch during training. This is the canonical interpretation of the strategy and the project hasn't tried it yet.

**Coupling note.** Combines naturally with §3.1 — pick a maximum `imgsz` (e.g. 896) and let `multi_scale` sample {448, 576, 704, 832, 896}.

**Decision criterion.** Better robustness to test-time inference resolution mismatches (measure by running test eval at `imgsz` ∈ {512, 640, 896}).

### 3.7 NMS post-processing (Cat. C #11 deepened)

The UI already exposes a confidence-threshold slider. The slider's *default* operating point, however, has never been tuned per-class.

| Param | Current | Proposed | Targets |
|---|---|---|---|
| Global `conf` threshold | 0.25 | **per-class {0.15, 0.25, 0.35, 0.45}** | `door_sign` false positives |
| Global `iou` threshold (NMS) | 0.7 | **{0.5, 0.6, 0.7}** | Overlapping detections |
| Class-aware NMS | off | **on** | Cross-class confusion (Batch 06 only) |

**Why.** Batch 06's `door_sign` precision regression came with 100% confidence — exactly the case where a small per-class threshold bump (e.g. 0.25 → 0.35 for `door_sign` only) silently restores perfect precision without any retraining. This is a *deployment* hyperparameter, not a *training* one, but should be calibrated against the val split before being baked into the UI default.

**Decision criterion.** A pure inference-time sweep on the val split that recovers `door_sign` precision = 1.000 with macro recall ≥ 0.95.

---

## 4. Search strategy

Three increasingly broad phases. Stop at the first phase that hits the §5 decision criteria.

### Phase 1 — Targeted manual sweep (recommended first batch of work)

A focused grid over the **highest-yield knobs** identified above, all aimed at the Batch 06 failure modes:

| Run | `imgsz` | Optimiser | `lr0` | WD | Label smoothing | mixup | copy_paste | NMS `door_sign` conf |
|---|---|---|---|---|---|---|---|---|
| Batch 07 | 896 | SGD | 0.01 | 1e-3 | 0.05 | 0.1 | 0.1 | (default) |
| Batch 07a | 896 | AdamW | 3e-3 | 1e-3 | 0.05 | 0.1 | 0.1 | (default) |
| Batch 07b | 896 | SGD | 0.01 | 1e-3 | 0.05 | 0.0 | 0.3 | (default) |

Three runs, ~3 hours total. Pick the winner against the §5 criteria; that's Batch 07 of record.

### Phase 2 — Ultralytics `model.tune()` (Bayesian / evolutionary)

If Phase 1 doesn't clear the criteria, escalate to Ultralytics' built-in tuner:

```python
from ultralytics import YOLO
model = YOLO("yolo11s.pt")
model.tune(
    data="data/dataset/data.yaml",
    epochs=50,                 # short per-trial budget
    iterations=30,             # 30 trials
    optimizer="AdamW",
    space={                    # only knobs Phase 1 left ambiguous
        "lr0":   (1e-4, 1e-2),
        "lrf":   (1e-3, 1e-1),
        "momentum": (0.85, 0.95),
        "weight_decay": (1e-4, 2e-3),
        "warmup_epochs": (1.0, 10.0),
        "box":   (5.0, 10.0),
        "cls":   (0.2, 1.0),
        "mixup": (0.0, 0.2),
        "copy_paste": (0.0, 0.4),
    },
    project="runs/tune",
    name="batch07_tune",
    plots=True,
    save=True,
)
```

Budget: ~15 hours overnight. Returns the best hyperparameter set plus a CSV of every trial for the report.

### Phase 3 — Joint Optuna / Ray Tune (only if Phase 2 is inconclusive)

Reserve for the case where the response surface is non-monotonic and we need TPE-class search. Document explicitly that this is overkill for an 800-image dataset and likely a sign that the underlying problem is a data limit, not a hyperparameter limit.

---

## 5. Decision criteria for promoting Batch 07 to "shippable"

The next batch is promoted to the project's official model **only if all four hold** on the held-out test split (not val):

1. `door_sign` precision ≥ 0.99 — fix the precision regression.
2. `door_sign` recall ≥ 0.94 — fix the recall regression.
3. Macro mAP@0.5:0.95 ≥ 0.88 — at least match Batch 06.
4. Generalisation gap (train vs val box-loss) ≤ 0.20 — narrower than Batch 06's 0.23.

If only (1)–(3) hold but (4) fails, the run is logged as a partial win and the next phase focuses on regularisation. If (1) fails repeatedly, the issue is dataset-shaped (more `door_sign` captures needed) and we stop tuning.

---

## 6. Risks and anti-patterns to avoid

- **Sweeping mAP@0.5.** Already saturated. Use mAP@0.5:0.95.
- **Tuning against the test split.** Leaks information; renders the final number meaningless. Tune on val; touch test only for the final candidate.
- **One-shot 50-trial sweeps with no logging.** Always save `args.yaml` and the full Ultralytics `results.csv` so a single config can be re-run deterministically.
- **Changing `seed` between sweep runs.** Defeats single-variable attribution. Keep `seed=42` for the whole sweep; vary it only for explicit robustness checks at the end.
- **Naively raising both `imgsz` and `batch`.** GPU memory will OOM on the RTX 4060 above `imgsz=896, batch=16`. Drop `batch` to 8 if `imgsz=1024` is needed.
- **Treating augmentation as free.** Each augmentation knob enlarges the effective training distribution; with only 560 training images the model can over-regularise. Always pair an augmentation bump with a `mosaic` close-epoch check.

---

## 7. Reproducibility template

Every Batch 07+ run should commit, alongside `best.pt` / `best.onnx`:

1. `args.yaml` — the full Ultralytics resolved config.
2. `results.csv` — the per-epoch metrics log.
3. A short markdown stub in `docs/batch07/training_notes.md` recording: the *exact* delta from Batch 06's config, the hypothesised effect, and the post-hoc observed effect.
4. The Ultralytics version (`pip freeze | grep ultralytics`) and the env spec from `technical_report.md` §4.2.

This is what allows the next report to keep the single-variable-attribution discipline that has made Project 2's findings reliable.

---

## 8. Concrete next step

The minimal, well-scoped next run is **Batch 07** as defined in Phase 1 row 1:

```bash
# Single-variable bundle: imgsz↑ + WD↑ + label_smoothing + mild mixup/copy_paste
yolo detect train \
    model=yolo11s.pt \
    data=data/dataset/data.yaml \
    imgsz=896 \
    epochs=120 \
    patience=20 \
    batch=8 \
    optimizer=SGD lr0=0.01 lrf=0.01 momentum=0.937 weight_decay=1e-3 \
    label_smoothing=0.05 \
    mixup=0.1 copy_paste=0.1 \
    seed=42 deterministic=True \
    project=runs/train name=07_training_batch
```

Wall-clock estimate: ~45 min on the RTX 4060. Evaluate against the §5 criteria and decide whether to ship Batch 07 or proceed to Phase 2.

---

## Appendix — Cross-references

- Baseline this plan improves upon: `docs/batch04_vs_batch05_vs_batch06/technical_report.md` §6.2 (Batch 06 per-class metrics) and §5.3 (generalisation-gap analysis).
- Strategy catalogue this plan deepens: `docs/technical_report.md` §3 (Project 2 brief mapping — Cat. A #3 hyperparameter optimisation is the strategy realised here).
- Environment spec: `docs/technical_report.md` §4.2.
