# FCM Landmark Detection

Eksamensprojekt: Detektion af frø-/kornkerne-positioner på Videometer-plader vha. YOLOv11n.

## Quick start

1. **Læs `PROJECT_SPEC.md`** først (fuld kontekst)
2. **Læs `CLAUDE.md`** hvis du bruger Claude Code
3. **Setup miljø:** se `setup_guide.md`
4. **Test pipeline:** kør scripts i `scripts/02_yolo_pipeline/` i rækkefølge

## Mappestruktur

```
.
├── PROJECT_SPEC.md         # FULD projekt-specifikation - læs først!
├── CLAUDE.md               # instruktioner til Claude Code
├── README.md               # denne fil
├── setup_guide.md          # lokal setup på Mac M-chip
│
├── scripts/
│   ├── 01_initial_fcn_attempt/   # arkiveret - FCN-tilgang vi droppede
│   └── 02_yolo_pipeline/         # ★ den aktive pipeline
│
├── visualizations/         # genererede billeder fra forrige kørsler
│
├── docs/
│   └── exam_qa.md         # forventede eksamensspørgsmål + svar
│
└── notes/
    └── timeline.md         # projekt-tidslinje og beslutninger
```

## Status

- ✅ Pipeline bygget og testet på syntetiske data (P=0.99, R=0.99, mAP@50=0.99)
- ⏳ Venter på rigtige Videometer-data
- ⏳ Skal bygge Excel/HIPS-loader til rigtige data
- ⏳ Skal migrere fra Colab til lokalt Mac-miljø

## Eksamen
- **Dato:** Uge 24 (11.–16. juni 2026)
- **Format:** 10 min præsentation + 10–15 min Q&A
- **Underviser:** Jon Eikholm (jone@ek.dk)
- **Industri-partner:** Videometer A/S
