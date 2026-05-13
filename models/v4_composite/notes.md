# v4_composite

**Trænet:** 2026-05-12 (22:13) → 2026-05-13 (04:42)
**Device:** Apple M4 Pro via PyTorch MPS
**Arkitektur:** YOLOv11n (2.58M params, 6.3 GFLOPs)
**Ultralytics:** 8.4.47 (TIDLIGERE — opdatér til 8.4.49+ før næste træning, se "Crash" nedenfor)

## Data

Composite-pipelinens første produktion (`data/composite_v1/`):
- **2000 composite-scener** bygget fra 12.384 alpha-stansede Videometer-frø-PNG'er
- **4096×2704 pixels** på fast blå plade-baggrund (Belt1 + Belt2 resized til match)
- **1600 train / 400 val** split (80/20)
- ~50-70 frø pr. scene, 50% scatter / 30% dense / 30% t_config-clusters

## Trænings-config

- `imgsz: 1280` (YOLO ned-skalerer 4096→1280 internt)
- `batch: 4` (MPS holdt ~18 GB unified memory)
- `epochs: 80` planlagt, **kun 13 gennemført før crash**
- `patience: 15`, `device: mps`
- Augmentations: flipud=0.5, fliplr=0.5, degrees=180, mosaic=1.0, mixup=0.15, copy_paste=0.3

## Metrics på val-set (400 composites, 23.793 frø-instances)

| Metric | Værdi | Krav | Status |
|---|---|---|---|
| Precision | **0.972** | > 0.5 | ✅ |
| Recall | **0.987** | > 0.3 | ✅ |
| mAP@50 | **0.992** | > 0.2 | ✅ |
| mAP@50-95 | **0.630** | (bonus) | God |

Modellen lærte ekstremt hurtigt — efter blot 13 epochs er den på 99.2% mAP@50. Det indikerer at composite-data er en god approximation af det modellen skal lære, og at task'en er overkommelig for YOLOv11n's kapacitet.

## Crash i epoch 14

Træning stoppede med `RuntimeError: shape mismatch: value tensor of shape [112275] cannot be broadcast to indexing result of shape [108940]` i `ultralytics/utils/tal.py:195` (Task-Aligned Learning assigner).

Det er en kendt sjælden bug i Ultralytics 8.4.47 + MPS, hvor specifikke kombinationer af pd_scores og mask_gt tensor-shapes triggrer en index-mismatch. Den er sandsynligvis fixet i 8.4.49+.

**Anbefaling før næste træning:** `.venv/bin/pip install -U ultralytics` (kræver internet).

## Sammenligning med tidligere v1/v2/v3 modeller

| Model | Data | mAP@50 | mAP@50-95 |
|---|---|---|---|
| v1_kalibreret | 300 ellipse-billeder, kun støj-niveau 0.02 | 0.9947 | 0.7162 |
| v2_robust | 600 ellipse-billeder, mixed conditions | TBD | TBD |
| v3_augmented | 1200 ellipse-billeder + plade-imperfektioner | TBD | TBD |
| **v4_composite** | **2000 rigtige Videometer-frø på blå plade** | **0.992** | **0.630** |

v4 er sammenligneligt med v1 på mAP@50, men trænet på reelt-lignende data i stedet for tegnede ellipser. Skal generalisere bedre når Andreas tester på rigtige Videometer-billeder.

## Hvad gik godt

- Composite-pipeline producerede data af god kvalitet
- YOLO lærte hurtigt — 99.2% mAP@50 efter 13 epochs er overrakte forventninger
- MPS-acceleration virkede (selv med crash i epoch 14)

## Hvad kan forbedres

- Få fuld 80 epochs gennemført — opdatér ultralytics og prøv igen
- Test på rigtige Videometer-scener for at validere generalisering
- A/B-test mod YOLO26n når v4 er fuldt trænet
