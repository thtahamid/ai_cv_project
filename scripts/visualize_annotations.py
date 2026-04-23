"""
Draw YOLO bounding box annotations over images and save to output folder.
Expects labels in YOLO format: class_id cx cy w h (normalized 0–1).
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────
IMAGES_DIR = Path("data/aggregated/door_sign/images")
LABELS_DIR = Path("data/aggregated/door_sign/labels")
OUTPUT_DIR = Path("data/aggregated/visualized/door_sign")

# Class names — must match data.yaml order
CLASS_NAMES = {
    0: "projector",
    1: "whiteboard",
    2: "fire_extinguisher",
    3: "door_sign",
}

# ── Visual settings ────────────────────────────────────────────────────────────
# One BGR colour per class; extends automatically for unknown class ids
PALETTE = [
    (0,   200, 255),   # 0 projector      — amber
    (0,   255, 100),   # 1 whiteboard      — green
    (255,  80,  80),   # 2 fire_extinguisher — blue
    (200,   0, 255),   # 3 door_sign        — violet
]
BOX_THICKNESS  = 2
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.55
FONT_THICKNESS = 1
LABEL_PADDING  = 4   # px above/below text inside the filled label bar


def color_for(class_id: int) -> tuple[int, int, int]:
    if class_id < len(PALETTE):
        return PALETTE[class_id]
    # deterministic fallback for unexpected class ids
    rng = np.random.default_rng(class_id)
    r, g, b = (int(v) for v in rng.integers(80, 230, 3))
    return (r, g, b)


def draw_boxes(image: np.ndarray, label_path: Path) -> np.ndarray:
    """Return a copy of *image* with all boxes from *label_path* drawn."""
    img = image.copy()
    h, w = img.shape[:2]

    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        cx, cy, bw, bh = map(float, parts[1:])

        # Convert normalised YOLO coords → pixel coords
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)

        color = color_for(cls_id)
        label = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

        cv2.rectangle(img, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # Label bar above the box (or inside if at top edge)
        (tw, th), _ = cv2.getTextSize(
            label, FONT, FONT_SCALE, FONT_THICKNESS
        )
        bar_y1 = y1 - th - 2 * LABEL_PADDING
        bar_y2 = y1
        if bar_y1 < 0:               # flip inside the box
            bar_y1 = y1
            bar_y2 = y1 + th + 2 * LABEL_PADDING

        cv2.rectangle(img, (x1, bar_y1), (x1 + tw + 2 * LABEL_PADDING, bar_y2), color, -1)
        cv2.putText(
            img, label,
            (x1 + LABEL_PADDING, bar_y2 - LABEL_PADDING),
            FONT, FONT_SCALE, (255, 255, 255), FONT_THICKNESS, cv2.LINE_AA,
        )

    return img


def process_image(img_path: Path) -> str:
    """Worker function executed in a subprocess for one image. Returns a status string."""
    label_path = LABELS_DIR / img_path.with_suffix(".txt").name
    if not label_path.exists():
        return f"skip:no_label:{img_path.name}"

    image = cv2.imread(str(img_path))
    if image is None:
        return f"skip:unreadable:{img_path.name}"

    annotated = draw_boxes(image, label_path)
    cv2.imwrite(str(OUTPUT_DIR / img_path.name), annotated)
    return f"ok:{img_path.name}"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )

    if not image_paths:
        print(f"No images found in {IMAGES_DIR}")
        return

    workers = min(os.cpu_count() or 4, len(image_paths))
    print(f"Processing {len(image_paths)} images with {workers} workers...")

    processed = skipped = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_image, p): p for p in image_paths}
        for future in as_completed(futures):
            result = future.result()
            if result.startswith("ok:"):
                processed += 1
            else:
                _, reason, name = result.split(":", 2)
                print(f"  [skip:{reason}] {name}")
                skipped += 1

    print(f"Done — {processed} saved to {OUTPUT_DIR}, {skipped} skipped.")


if __name__ == "__main__":
    main()
