"""Single-step frø-analyse — erstatter Videometers segmentering→blob→centroid.

Videometers nuværende pipeline (flere skridt):
    multispektral capture → segmentering → blob-analyse (split af rørende frø)
    → center-of-mass pr. blob (Excel-output)

Denne pipeline (ét detektions-skridt + let klassisk analyse):
    YOLO finder hvert frø i ÉT skridt (løser instans-splittingen, også når frø
    rører) → HSV adskiller frø-pixels fra plade → hver pixel tildeles nærmeste
    detektions-center → pr. frø: center-of-mass, maske og orientering.

Kernen ligger i scripts/03_demo/seed_morphology.py og deles med Streamlit-demoen,
så de to bruger præcis samme kode.

Kør:
    .venv/bin/python scripts/07_single_step_analysis.py <billede.png> [--conf 0.5] [--out <dir>]
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "active" / "best.pt"

# Genbrug den delte morfologi-kerne + lag-render fra demoen
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "03_demo"))
import seed_morphology  # noqa: E402
import overlays  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Single-step frø-analyse")
    parser.add_argument("image", help="sti til plade-billede (PNG)")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="confidence-tærskel (default 0.5 — ægte frø ligger 0.9+)")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--out", default="/tmp/fcm_single_step",
                        help="output-mappe til visualisering + CSV")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    # --- Skridt 1: ÉT detektions-skridt ---
    model = YOLO(str(MODEL_PATH))
    # cv2 læser BGR; vi arbejder i RGB (som demoen/PIL) og konverterer tilbage til sidst
    image_rgb = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    result = model.predict(args.image, imgsz=args.imgsz, conf=args.conf,
                           device="mps", verbose=False)[0]
    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    bbox_centers = np.stack([(xyxy[:, 0] + xyxy[:, 2]) / 2,
                             (xyxy[:, 1] + xyxy[:, 3]) / 2], axis=1)

    # --- Skridt 2-3: segmentér + tildel pixels (split af rørende frø) ---
    box_diag = np.sqrt((xyxy[:, 2] - xyxy[:, 0]) ** 2 + (xyxy[:, 3] - xyxy[:, 1]) ** 2)
    max_dist = float(np.median(box_diag)) if len(box_diag) else 200.0
    seed_mask, plate_bbox = seed_morphology.segment_foreground(image_rgb)
    labels = seed_morphology.assign_pixels_to_seeds(seed_mask, bbox_centers, max_dist=max_dist)

    # --- Skridt 4: mål center-of-mass, areal, orientering pr. frø ---
    measurements = seed_morphology.measure_seeds(labels, n_seeds=len(bbox_centers))

    # Sammenlign rigtig center-of-mass med bbox-center (hvor meget flytter det sig?)
    com_shift = np.array([
        np.hypot(m["com_x"] - bbox_centers[m["id"] - 1][0],
                 m["com_y"] - bbox_centers[m["id"] - 1][1])
        for m in measurements
    ])

    # --- Output: visualisering (samme lag-render som demoen) + CSV ---
    overlay_rgb = overlays.render_overlay(
        image_rgb, xyxy, confs,
        show_masks=True, show_boxes=False, show_bbox_center=False,
        show_com=True, show_orientation=True, show_confidence=False,
        labels=labels, measurements=measurements,
    )
    out_img = out_dir / f"{stem}_analysis.png"
    cv2.imwrite(str(out_img), cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

    out_csv = out_dir / f"{stem}_seeds.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "com_x", "com_y", "area_px", "angle_deg"])
        writer.writeheader()
        for m in measurements:
            writer.writerow({k: round(m[k], 2) if isinstance(m[k], float) else m[k]
                             for k in writer.fieldnames})

    # --- Resumé til terminalen ---
    plate_area = (plate_bbox[2] - plate_bbox[0]) * (plate_bbox[3] - plate_bbox[1])
    coverage = 100.0 * int(seed_mask.sum()) / plate_area if plate_area else 0.0
    print(f"Billede:            {stem}")
    print(f"Detektioner (conf>={args.conf}): {len(bbox_centers)}")
    print(f"Frø målt (efter areal-filter):  {len(measurements)}")
    print(f"Plade-coverage:     {coverage:.1f} %")
    if len(com_shift):
        print(f"Center-of-mass vs bbox-center: median {np.median(com_shift):.1f} px, "
              f"max {com_shift.max():.1f} px")
    print(f"Skrev: {out_img}")
    print(f"Skrev: {out_csv}")


if __name__ == "__main__":
    main()
