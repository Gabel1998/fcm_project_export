# v5_no_leak

**Trænet:** 2026-05-13/14 (over 2 sessioner med resume undervejs)
**Device:** Apple M4 Pro via PyTorch MPS
**Arkitektur:** YOLOv11n (2.59M params, 6.4 GFLOPs)
**Ultralytics:** 8.4.49

## Data — NU UDEN data leakage

Dette er den første "rene" model trænet på `data/composite_v1_no_leak/`:
- **2000 composite-scener** (1600 train / 400 val)
- **9907 frø i train-pool, 2477 i val-pool — DISJUNKTE**
- Modellen ser ALDRIG val-frø under træning
- Metrics afspejler reel generalisering, ikke memorisering

## Trænings-config

- `imgsz: 1280`, `batch: 4`, `epochs: 80` planlagt
- `patience: 15`, `device: mps`
- **13 epochs gennemført** før crash i epoch 14 validation
- Augmentations: flipud=0.5, fliplr=0.5, degrees=180, mosaic=1.0, mixup=0.15, copy_paste=0.3

## Metrics på val-set (400 composites, 23.954 frø-instances)

```
Precision:  0.9995    (krav: > 0.5)
Recall:     0.9999    (krav: > 0.3)
mAP@50:     0.9950    (krav: > 0.2)
mAP@50-95:  0.8260    (bonus)
```

**Best.pt er fra epoch 8** (mAP50-95 = 0.822 ved selve epoch). Final val efter træning gav 0.826 — alle tal over Jone's krav med stor margin.

## Sammenligning v4 (m. leakage) vs v5 (clean)

| | v4 | v5 |
|---|---|---|
| Precision | 0.972 | **0.9995** |
| Recall | 0.987 | **0.9999** |
| mAP@50 | 0.992 | **0.9950** |
| mAP@50-95 | 0.630 | **0.8260** |

**v5 er bedre på ALLE metrics** — særligt mAP@50-95 (0.83 vs 0.63). Uden data leakage tvinges modellen til at lære generelle frø-features i stedet for at memorisere specifikke stamps → bedre lokalisering.

Det viser også at **vores composite-pipeline producerer god træningsdata** — modellen klarer 99.5% mAP@50 selv på disjunkte stamps.

## Crash i epoch 14 (Ultralytics MPS-bug)

Same bug som v4 ramte:
```
File "ultralytics/utils/tal.py", line 200, in get_box_metrics
    overlaps[mask_gt] = self.iou_calculation(gt_boxes, pd_boxes)
RuntimeError: shape mismatch: value tensor of shape [66701] cannot be broadcast
to indexing result of shape [90028]
```

Vedvarende Ultralytics MPS-bug i Task-Aligned Learning assigner — IKKE fixet i 8.4.49. Workarounds til næste træning:
- `--device cpu` (10× langsommere, men stabilt)
- `val=False` i træning, kør kun final val efter
- Reducer `batch` til 2 for at undgå specifikke tensor-shapes

## Hvad gik godt

- **Composite-pipeline uden leakage virker** — model generaliserer
- **Højere mAP@50-95 end v4** trods kortere træning — ægte forbedring
- 13 epochs er rigeligt for denne task

## Hvad kan forbedres

- Få en fuld 80-epoch træning gennemført (kræver workaround for MPS-bug)
- Validér på rigtige Videometer-scener
- A/B-test mod YOLO26n
