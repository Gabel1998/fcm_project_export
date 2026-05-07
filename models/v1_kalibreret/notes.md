# v1_kalibreret

**Trænet:** 2026-05-07, lokalt på Apple M4 Pro (MPS)
**Trænings-tid:** 23 minutter, 48 epochs (early stopping ved epoch 38, patience=10)
**Ultralytics:** 8.4.47

## Trænings-data

- 300 syntetiske billeder, 256×256 pixels
- 17.580 frø totalt (gennemsnitligt 58.6 pr. billede)
- **Ensartede betingelser:**
  - `noise_level=0.02` ("kalibreret") for alle
  - Frø-størrelse: 12-18 px lange ("normal")
  - Mm/pixel: 0.05
- Train/val: 240/60

## Resultater (val-sæt)

| Metric | Værdi |
|---|---|
| Precision | 0.9916 |
| Recall | 0.9988 |
| mAP@50 | 0.9947 |
| mAP@50-95 | 0.7488 |
| Median fejl | 0.028 mm |

## Kendte begrænsninger

Modellen er trænet på én noise-distribution og én frø-størrelse, så den degraderer markant uden for det:

- "realistisk" støj (0.05): Precision falder til ~0.50
- "svært" støj (0.10): Precision falder til ~0.44
- "større" frø: Precision falder til ~0.95 (mindre dramatisk)
- 128×128 billede: Precision falder til ~0.73 (modellen er trænet på 256)

Det er motivationen for v2_robust — træning med mix af betingelser.
