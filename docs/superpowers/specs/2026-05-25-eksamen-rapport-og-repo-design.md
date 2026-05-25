# Eksamens-rapport og GitHub-repo (FCM Landmark Detection)

**Dato:** 2026-05-25
**Forfatter:** Andreas Søgård Gabel
**Status:** Design — afventer godkendelse
**Deadline:** 27. maj 2026 (Wiseflow)

---

## 1. Mål

Producér en eksamensaflevering der består af:

1. En **3-5 siders skriftlig rapport** (`REPORT.md` + `REPORT.pdf`) der fortæller projektet som en sammenhængende metodisk fortælling — fra syntetisk-data-koncept til færdig trænet model.
2. Et **offentligt GitHub-repo** der fungerer som reproducerbarheds-pakke: censor skal kunne klone, installere og køre demo'en på under 10 minutter.
3. En **kørbar demonstration** med den endelige model (v5_no_leak) på vedlagte testbilleder, inklusive en objectness-heatmap-visualisering som interpretability-værktøj.

Afleveringen sker til Wiseflow (PDF + GitHub-link). Den mundtlige eksamen i uge 24 2026 bruger samme materiale som talepapir.

## 2. Stil og sprogkrav

Disse krav er bindende for al rapport- og spec-tekst (ikke kode-kommentarer, ikke chat):

- **Sprog:** Dansk.
- **Person:** Aldrig 1. person (intet "vi", "jeg", "vores", "os"). Passiv form eller nominalisering bruges i stedet.
- **Tid:** Førnutid som default ("der er bygget", "modellen er blevet trænet", "leakage er blevet identificeret"). Begrundelse: projektet er afsluttet på skrivetidspunktet, men resultatet er stadig gældende.
- **Kode i rapport:** Vises som markdown code blocks (` ```python `), aldrig som screenshots.
- **Visuelle resultater:** Vises som indlejrede PNG-billeder (træningskurver, predictions, heatmaps, confusion matrix).

## 3. Out of scope

- Akademisk fuld rapport (10+ sider).
- Real-world Videometer-validering (planlagt, men ikke afsluttet før deadline — nævnes som fremtidsudsigt).
- Multi-class detection (kun ét class label: "frø").
- Web-deployment af modellen (Streamlit-demo køres lokalt).
- Inddragelse af FCN-forsøget i selve rapportens hovedfortælling (det blev forkastet tidligt og er ikke relevant for det endelige produkt).
- Slavisk overholdelse af de 5 spec-krav fra opgavebeskrivelsen — de dækkes implicit, men strukturen følger projektets fortælling.

## 4. Rapport-struktur (REPORT.md)

Den kronologiske fortælling, ca. 3-5 sider. Sektionsoverskrifter er indikative og kan justeres under skriving.

### 4.1 Problemstilling (~½ side)

- Kort introduktion til FCM-projektet og samarbejdet med Videometer.
- Den manuelle annotering af frø-centre (center-of-mass) tager tid og skalerer dårligt.
- Mål: automatisk landmark-detektion af frø på Videometers blå plade via en YOLO-baseret model.

### 4.2 Datagrundlag — hvorfor syntetisk data (~½ side)

- Begrænset adgang til ægte Videometer-data (NDA, lille mængde, manuel annotering).
- Konceptet bag syntetisk data: enkeltfrø-billeder kombineres procedurelt på en blå plade-baggrund.
- 7.000 syntetiske 1280×1280 composite-billeder er blevet genereret med fuld kontrol over labels.
- **Asset:** eksempel-billede fra `test_images/composite_00000.png`.

### 4.3 Tidlige tilgange og opdaget data leakage (~1 side)

- Kort kronologi over v1-v4 (én sætning hver).
- v4 viste mistænkeligt høje metrikker → undersøgelse afslørede at samme baggrunde optrådte i både train- og val-split.
- Læringen: realistisk evaluering kræver leakage-fri splits — baggrundsbilleder må ikke deles på tværs.
- **Asset:** (valgfri) sammenligning af v4 vs. v5 predictions på samme billede.

### 4.4 Composite-pipelinen og v5_no_leak (~1 side)

- Skitse af 6-modul composite-generator-arkitektur (uden kode-detaljer).
- Hvordan train/val-splittet er ændret så baggrunde holdes adskilt.
- Træning på Apple Silicon (MPS), 80 epochs, YOLOv11n.
- **Asset:** evt. arkitektur-diagram (kan udelades hvis tiden er knap).
- **Code block:** kort uddrag af composite-konfigurationen (yaml eller python-dict).

### 4.5 Resultater (~½ side)

- Tabel over mAP@50 og mAP@50-95 for v1-v5 (viser leakage-effekten).
- Slutmetrikker for v5_no_leak: **mAP@50 = 0.999, mAP@50-95 = 0.969**.
- Objectness-heatmap som interpretability — viser at modellens interne confidence stemmer overens med bounding box-outputtet.
- **Assets:** `results.png` (træningskurver), `BoxPR_curve.png`, `confusion_matrix.png`, prediction-billede, heatmap-overlay.

### 4.6 Refleksion og fremtidsudsigter (~½ side)

- Real-world validering på ægte Videometer-billeder er næste skridt.
- Mulige forbedringer: mere realistisk lyssætning i syntetisk data, multi-class (forskellige frø-typer), domain adaptation.
- Kort refleksion over hvad data-leakage-opdagelsen har lært om metodisk strenghed.

## 5. GitHub repo-struktur

```
fcm_project_export/
├── README.md              # Setup-guide (lokal Mac + Colab) + projekt-intro
├── REPORT.md              # Rapporten — linket direkte fra Wiseflow
├── REPORT.pdf             # PDF-eksport, afleveret på Wiseflow
├── requirements.txt       # Pinned Python-dependencies
├── .gitignore             # Holder NDA-data, runs/, .venv/, etc. ude
├── models/
│   └── v5_no_leak/
│       └── best.pt        # 5.3 MB — den endelige model
├── test_images/
│   ├── composite_00000.png # 3 syntetiske testbilleder (~36 MB i alt)
│   ├── composite_00001.png
│   └── composite_00002.png
├── scripts/
│   ├── 02_yolo_pipeline/  # Composite generator + træningsscripts
│   ├── 03_demo/           # Streamlit-demo med objectness heatmap
│   ├── 04_colab/          # Colab-notebook (peger på seneste version)
│   └── 05_generate_report_assets.py  # NYT: bygger alle rapport-billeder
└── docs/
    ├── superpowers/specs/ # Design-specs (denne fil)
    └── screenshots/       # Genererede assets brugt i REPORT.md
```

Det fulde syntetiske datasæt (2.5 GB) lægges **ikke** i repoet. I stedet:

- README linker til en Google Drive-mappe hvor `composite_v1_no_leak_1280.tar.gz` kan downloades.
- Composite-generator-koden er tilstrækkelig til at regenerere datasættet fra bunden.

## 6. Asset-strategi (rapport-billeder)

Et nyt script `scripts/05_generate_report_assets.py` orkestrerer al billed-generering:

1. **Eksisterende træningsoutputs kopieres** fra `runs/detect/runs/train/v5_no_leak/` til `docs/screenshots/` med pænere filnavne:
   - `results.png` → `training_curves.png`
   - `BoxPR_curve.png` → `pr_curve.png`
   - `confusion_matrix.png` → `confusion_matrix.png`
   - `val_batch0_pred.jpg` → `val_predictions.jpg`
2. **Prediction-overlay genereres** på `test_images/composite_00000.png` (model loadet fra `models/v5_no_leak/best.pt`, bounding boxes tegnet på).
3. **Objectness heatmap-overlay genereres** på samme testbillede (logikken er genbrugt fra `scripts/03_demo/upload_demo.py`).
4. **(Valgfrit) sammenligningsbillede v4 vs. v5** genereres på samme testbillede hvis tiden tillader det.

Scriptet skal være idempotent og kunne køres med én kommando: `python scripts/05_generate_report_assets.py`.

## 7. README-indhold

`README.md` på rod-niveau skal indeholde:

1. **Projekt-overskrift** + en sætnings beskrivelse.
2. **Setup — lokalt (Mac/MPS):** Python 3.11, venv, `pip install -r requirements.txt`, test af MPS.
3. **Setup — Colab:** link til notebook + kort beskrivelse af GPU-fordelen.
4. **Kør demo'en:** Streamlit-kommando + hvilke testbilleder der findes.
5. **Reproducer rapport-billeder:** `python scripts/05_generate_report_assets.py`.
6. **Reproducer datasæt:** kort note + link til generator.
7. **Reference:** link til REPORT.md.

## 8. Konkrete leverancer (definition of done)

- [ ] `REPORT.md` skrevet med alle 6 sektioner og indlejrede assets.
- [ ] `REPORT.pdf` eksporteret (Pandoc eller VS Code Markdown PDF).
- [ ] `README.md` med lokal + Colab setup-guide testet end-to-end.
- [ ] `scripts/05_generate_report_assets.py` kører rent og producerer alle billeder.
- [ ] `models/v5_no_leak/best.pt` + 3 testbilleder committet i repoet.
- [ ] `.gitignore` verificeret: ingen NDA-data, ingen `runs/`, ingen `.venv/`.
- [ ] `requirements.txt` opdateret og minimal.
- [ ] Repoet er pushet til offentlig GitHub.
- [ ] PDF + GitHub-link afleveret på Wiseflow.

## 9. Tidsplan (2 dage)

| Dag | Hovedopgaver |
|---|---|
| **25/5 aften** | Skriv REPORT.md udkast, byg `05_generate_report_assets.py`, generér alle billeder. |
| **26/5** | Polish rapport, eksport PDF, skriv/testkør README, oprydning, commit + push. |
| **27/5** | Aflever PDF + GitHub-link på Wiseflow. Buffer til småfejl. |

## 10. Risici og buffer

- **Markdown → PDF-eksport kan have layoutproblemer.** Plan B: print til PDF fra browser med GitHub-rendering.
- **Asset-script kan tage længere end forventet.** Plan B: skip valgfri v4-vs-v5 sammenligning.
- **Datasæt-download via Google Drive kan kræve KEA-konto-konfiguration.** Plan B: brug WeTransfer eller link til extern hosting.
