# v3_augmented

**Trænet:** 2026-05-07, Colab T4 GPU
**Arkitektur:** YOLOv11s (9.4M params, ~19 MB)
**Ultralytics:** 8.4.47

## Data (1200 billeder, 256×256, 80/20 split)

Mixed conditions pr. billede:
- Støj-niveau: random (0.02 / 0.05 / 0.10)
- Frø-størrelse: random (mindre / normal / større)
- Baggrunds-reflektans: uniform i [0.75, 0.95]

Plus **plade-imperfektioner** (nyt i v3):
- 50% af billeder: 3-15 dust-pletter
- 30%: 1-3 ridser
- 70%: tilfældig lys-gradient

## Trænings-augmentations

Spatial: `flipud=0.5`, `fliplr=0.5`, `degrees=180`, `perspective=0.0005`, `shear=5.0`
Mix: `mixup=0.15`, `copy_paste=0.3`, plus Ultralytics defaults (mosaic, HSV, etc.)
Early stopping: `patience=15`, max 80 epochs

## v1 vs v3 — head-to-head på 6 edge cases

| Case | v1 P | **v3 P** | v1 F1 | **v3 F1** |
|---|---|---|---|---|
| Sparsomt (20 frø) | 1.000 | 1.000 | 1.000 | 1.000 |
| Tæt (80 frø) | 0.952 | 0.964 | 0.976 | 0.982 |
| Lille billede (128px) | 0.727 | **1.000** | 0.842 | **1.000** |
| Svær støj | 0.443 | **0.879** | 0.614 | **0.935** |
| Store frø | 0.951 | 0.921 | 0.975 | 0.959 |
| Worst-case (alt værst) | 0.020 | **0.444** | 0.038 | **0.615** |

## Trade-off

Median-afstand er 2-3× højere end v1 på nemme cases (0.025 → 0.055 mm), men
v3 er **22× bedre** på worst-case og **2× bedre** på svær støj. Stadig under
0.1 mm overalt — sub-pixel niveau.

## Demo-strategi

Skift frem og tilbage mellem v1 og v3 for at vise udviklingen i robusthed, især på svære cases (støj, lille billede, worst-case). V3's forbedringer er tydeligst i edge cases, mens v1 og v3 er mere ens på nemme cases.
