"""Hurtigt IoU-tjek på et rigtigt val-composite (majs på blå plade).

Formål: se de FAKTISKE IoU-tal mellem modellens bokse og ground-truth-bokse,
før vi bygger IoU ind i demoen. Kører på modellens egen trænings-fordeling
(composite_v1_no_leak), så tallene er meningsfulde — ingen ellipse-forbehold.

Kør:
    .venv/bin/python scripts/iou_check.py
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "active" / "best.pt"
VAL_IMG_DIR = ROOT / "data" / "composite_v1_no_leak" / "images" / "val"
VAL_LBL_DIR = ROOT / "data" / "composite_v1_no_leak" / "labels" / "val"
IMGSZ = 1280            # samme som træning/eval
CONF = 0.25             # samme confidence-tærskel som demoen


def iou_xyxy(box_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU mellem én box (4,) og N bokse (N,4) — alt i pixel-xyxy."""
    # Skæringsrektanglets hjørner
    x1 = np.maximum(box_a[0], boxes_b[:, 0])
    y1 = np.maximum(box_a[1], boxes_b[:, 1])
    x2 = np.minimum(box_a[2], boxes_b[:, 2])
    y2 = np.minimum(box_a[3], boxes_b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)   # overlap-areal
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a + area_b - inter                                  # forenings-areal
    return inter / np.clip(union, 1e-9, None)


def load_gt_boxes(label_path: Path, img_w: int, img_h: int) -> np.ndarray:
    """Læs YOLO-labels (class cx cy w h, normaliseret) → pixel-xyxy (N,4)."""
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        _, cx, cy, w, h = (float(v) for v in line.split())
        # Normaliseret center+størrelse → absolutte hjørner
        bx, by, bw, bh = cx * img_w, cy * img_h, w * img_w, h * img_h
        boxes.append([bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2])
    return np.array(boxes, dtype=np.float32)


def main() -> int:
    model = YOLO(str(MODEL_PATH))
    img_path = sorted(VAL_IMG_DIR.glob("*.png"))[0]
    lbl_path = VAL_LBL_DIR / (img_path.stem + ".txt")
    img_w, img_h = Image.open(img_path).size
    print(f"Billede:  {img_path.name}  ({img_w}x{img_h})")

    gt = load_gt_boxes(lbl_path, img_w, img_h)
    print(f"Ground truth: {len(gt)} frø")

    res = model.predict(str(img_path), imgsz=IMGSZ, conf=CONF,
                        device="mps", verbose=False)[0]
    pred = res.boxes.xyxy.cpu().numpy()
    print(f"Detektioner:  {len(pred)} bokse (conf >= {CONF})")

    # Greedy: hver prediction tager sit bedste ledige GT
    used = np.zeros(len(gt), dtype=bool)
    ious = []
    for pbox in pred:
        scores = iou_xyxy(pbox, gt)
        scores[used] = -1                  # bloker allerede-matchede GT
        j = int(np.argmax(scores))
        if scores[j] > 0:
            used[j] = True
            ious.append(float(scores[j]))

    ious = np.array(ious)
    print(f"\nMatchede par: {len(ious)} / {len(gt)} GT")
    print(f"Median IoU:   {np.median(ious):.3f}")
    print(f"Mean   IoU:   {np.mean(ious):.3f}")
    print(f"Min/Max IoU:  {ious.min():.3f} / {ious.max():.3f}")
    # Hvor mange ville "bestå" ved de klassiske mAP-tærskler
    for t in (0.50, 0.75, 0.90, 0.95):
        print(f"  IoU >= {t:.2f}:  {int((ious >= t).sum())} / {len(ious)} "
              f"({100*(ious >= t).mean():.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
