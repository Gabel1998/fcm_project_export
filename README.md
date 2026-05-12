# FCM Landmark Detection

Eksamensprojekt: Detektion af frø-/kornkerne-positioner på Videometer-plader vha. YOLOv11n.

## Quick start

```bash
make help            # se alle kommandoer
make status          # hvad findes / hvad kører
make gen             # generér 2000-composite dataset (~30 min)
make train           # træn YOLOv11n på det (~1-7 timer afhængig af imgsz)
make watch           # live-monitorering i andet terminal-vindue
make save-model      # arkivér trænet model under models/<NAME>/
```

## Dokumentation

| Fil | Hvad det dækker |
|---|---|
| `PROJECT_SPEC.md` | Fuld projekt-kontekst og beslutninger — læs først |
| `OFFLINE_GUIDE.md` | Selvkørende workflow uden internet (genere + træne + monitorere) |
| `PARAMETERS_GUIDE.md` | Alle parametre i pipelinen — hvad du kan tweake og hvad effekten er |
| `docs/specs/2026-05-11-composite-pipeline-design.md` | Composite-pipeline design |
| `docs/plans/2026-05-11-composite-pipeline.md` | Step-by-step implementeringsplan |
| `models/README.md` | Model-versionering og hvordan switch_model.sh virker |
| `setup_guide.md` | Lokal setup på Mac M-chip |
| `CLAUDE.md` | Instruktioner til Claude Code (hvis du bruger det) |

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

- ✅ Ellipse-pipeline (v0) — arkiveret under `scripts/02a_archived_ellipse_synthetic/`
- ✅ v1/v2/v3 modeller trænet på syntetiske ellipser (P=0.99, R=0.99, mAP@50=0.99)
- ✅ Composite-pipeline bygget — kombinerer 12.384 rigtige Videometer-frø-PNG'er + 2 plade-baggrunde til realistiske scener
- 🟡 v4_composite model under træning (eller venter på fuld dataset)
- ⏳ A/B-test YOLOv11n vs YOLO26n (efter v4)
- ⏳ Demo opdateret med rigtig data (besøg hos Videometer)

## Eksamen
- **Dato:** Uge 24 (11.–16. juni 2026)
- **Format:** 10 min præsentation + 10–15 min Q&A
- **Underviser:** Jon Eikholm (jone@ek.dk)
- **Industri-partner:** Videometer A/S
