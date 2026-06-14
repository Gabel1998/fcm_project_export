# Fremlæggelse — cue-sheet (10 min)

Ikke et manus. Stikord til at danse rundt i kode/billeder + live demo.
Dækker Jons 6 punkter: demo · hvordan bygget · overvejelser · fravalg · godt/skidt · andet.

**Før du starter (pre-flight):**
- [ ] `make upload-demo` kører ALLEREDE i en terminal (browseren åben) — start aldrig den live foran censor
- [ ] Et **val-composite** kopieret et nemt sted hen (fx Skrivebordet): `data/composite_v1_no_leak/images/val/composite_01600.png` — val-billede = har labels, så **IoU-checkboxen virker** (rigtige majs, IKKE ellipser)
- [ ] `models/active` peger på `v5_no_leak` (`./switch_model.sh v5_no_leak`)
- [ ] Ét billede klar i en fane: `models/v5_no_leak/results.png`
- [ ] IntelliJ åben på `composite_generator/` og `seed_morphology.py`

---

## 0:00–1:00 — Ramme & problem
**ÅBN:** evt. `test_images/composite_00000.png`
**SIG:**
- Videometer scanner frø på en blå plade; opgaven = find hvert frøs **landmark** (centrum).
- Videometer bruger i dag en **fler-steps blob-pipeline**. Mit mål: **ét lært detektions-skridt** der finder alle frø + let efterbehandling der giver samme mål (center-of-mass, areal, orientering).
- **NDA:** rigtige data må ikke vises/uploades → derfor en **syntetisk generator** (det er selve pointen, ikke en nødløsning).

## 1:00–3:30 — LIVE DEMO (hook'et — vis det virker først)
**ÅBN:** browseren med `make upload-demo` → upload `composite_01600.png` (et val-composite med rigtige majs)
**SIG / gør:**
- "Det her er et syntetisk composite af **rigtige majs-frø** klippet ud og sat på den blå Videometer-plade — ikke abstrakte former. Og det er et **val-billede modellen aldrig har trænet på**."
- Kør detektion → modellen finder alle frø som bokse.
- Tænd lagene ét ad gangen: **masker · center-of-mass (+) · orienterings-pil · objectness-heatmap**.
- **Den vigtige sætning:** "Kun **bokse** er fra modellen. Masken, center-of-mass og orienteringen er **klassisk billedbehandling** jeg lægger ovenpå — ét detektions-skridt erstatter Videometers fler-steps pipeline."
- Tænd **"Vis IoU mod ground truth"** → grønne sande bokse vs røde model-bokse + IoU pr. frø. Sig: *"median IoU 0,97; alle består ved 0,5, ~91% ved 0,95 — det ER min mAP@50-95-historie."*
- **Brug IKKE `make demo`** (ellipse-showcasen) — den repræsenterer ikke dit majs-projekt.

## 3:30–5:30 — Hvordan det er bygget
**ÅBN:** `scripts/02_yolo_pipeline/composite_generator/` → så `visualizations/synthetic_data_examples.png`
**SIG (pipeline, modul for modul):**
- `seed_loader.py` — klipper rigtige frø ud som "stamps"
- `background_loader.py` — den blå plade som baggrund
- `packing.py` — placerer frø **uden overlap** (Videometer-constraint)
- `composer.py` — samler scene + skriver YOLO-labels
- `generate_dataset.py` — kører det hele → færdigt YOLO-datasæt

**ÅBN:** `scripts/02_yolo_pipeline/03b_train_composite.py`
- YOLOv11n (2,59M params), `imgsz=1280`, augmentations (flip, 180° rotation, mosaic, mixup, copy-paste).

**ÅBN:** `scripts/03_demo/seed_morphology.py`
- Efterbehandling: HSV-segmentering → split rørende frø via nærmeste detektions-center (marker-watershed) → center-of-mass + orientering via momenter.

## 5:30–7:00 — Overvejelser & fravalg
**SIG (fravalg — vær konkret om HVORFOR):**
- **FCN + PSNR** → fravalgt (Jones råd; 2015-arkitektur). Valgte **detection** i stedet.
- **ConvNeXt** → er classification, ikke detection.
- **Heatmap-som-output** → droppet — MEN beholdt som **interpretability-overlay** (ikke det samme! modellen *trænes ikke* til at lave heatmap).
- **Rigtige data i Colab** → aldrig (NDA + 14 GB).

**Videometer-constraints jeg byggede ind:** blå plade fast · ingen overlap · droppede farve/lys-jitter (unødvendig variation).

## 7:00–8:30 — Resultater + historien
**ÅBN:** `models/v5_no_leak/results.png` + `docs/screenshots/v4_vs_v5_comparison.png`
**SIG:**
- v5: **mAP@50-95 = 0,97**, precision/recall ~1,0.
- **Data leakage-historien (stærkeste kort):** v4 delte de samme frø-stamps mellem train og val → 0,63. v5 har **disjunkte** frø → 0,97. Tallet steg fordi modellen blev tvunget til at **generalisere**, ikke memorere.
- **MPS-bug:** Ultralytics' TAL-assigner crasher på Apple MPS (`tal.py`). `make auto-resume` restartede fra `last.pt` → alle 80 epochs gennemført. Peak fitness epoch 77.
- **mm-præcision:** 0,05 mm/pixel → ~sub-pixel fejl.

## 8:30–9:30 — Hvad gik godt / skidt + andet
**ÅBN:** `docs/screenshots/heatmaps_per_scale.png`
**SIG:**
- **Godt:** pipelinen generaliserer; sub-pixel-præcision; ét detektions-skridt erstatter fler-steps blob-pipeline.
- **Cool:** objectness-heatmap på P3/P4/P5 viser **multi-scale-arkitekturen** (fint grid = små objekter). Det er det eneste ægte model-interne lag i demoen.
- **Skidt (vær ærlig):** MPS-buggen kostede tid; **rigtige Videometer-scener er endnu ikke testet** — næste skridt.

## 9:30–10:00 — Afrunding
**SIG:**
- En sætning: "Lært detektor + klassisk efterbehandling, trænet på syntetiske data jeg selv genererer, fordi rigtige data er under NDA. 0,97 mAP, men reel test på Videometer-scener mangler."
- "Det er hvad jeg har — jeg er klar til spørgsmål."

---

## Hvis du løber tør for tid
Drop i denne rækkefølge: 8:30-blokken (godt/skidt) → 5:30-blokken (fravalg, men nævn FCN-fravalget kort). **Behold ALTID** demo + data-leakage-historien — det er dine to stærkeste kort.

## Hvis demoen fryser
- Sig roligt: "den er live, lad mig give den et øjeblik" — eller skift til `docs/screenshots/streamlit_demo.png` og `demo_predictions.png` og forklar ud fra dem. (Du valgte ingen video-backup; screenshots er din nødplan.)