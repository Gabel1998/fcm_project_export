# v2_robust

**Trænet:** 2026-05-07, Colab T4 GPU
**Træningstid:** 4.5 min, 27 epochs (early stopping ved epoch 17)
**Arkitektur:** YOLOv11s (9.4M params, ~19 MB) — opgraderet fra v1's nano
**Ultralytics:** 8.4.47

## Data (1200 billeder, 80/20 split)

Mixed conditions pr. billede:
- Støj-niveau: random (0.02 / 0.05 / 0.10)
- Frø-størrelse: random (mindre / normal / større)
- Baggrunds-reflektans: uniform i [0.75, 0.95]

**Note:** Genereret ved `IMG_SIZE=512` men trænet ved `imgsz=256` (resize ned).
Lidt spild — i v3 blev det rettet til 256/256 match.

## Trænings-augmentations (kun standard)

`flipud=0.5`, `fliplr=0.5`, `degrees=180.0` plus Ultralytics defaults
(mosaic, HSV, scale, translate). **Ingen** mixup/copy_paste/perspective —
det kommer først i v3.

## Resultater (val-sæt, mixed conditions)

| Metric | Værdi |
|---|---|
| Precision | 0.9951 |
| Recall | 0.9965 |
| mAP@50 | 0.9950 |
| mAP@50-95 | 0.6176 |

mAP@50-95 lavere end v1 (0.7488) fordi val-sættet er sværere — modellen
testes på mixed conditions, ikke kun kalibreret.

## Plads i versions-historikken

V2 er **mellemtrinnet** mellem v1 (kun kalibreret) og v3 (mixed + plade-
imperfektioner). Hvis du vil vise progressionen i demoen, switch i
rækkefølge: v1 → v2 → v3.
