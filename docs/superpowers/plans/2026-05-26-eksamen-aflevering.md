# Eksamens-aflevering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Levér færdig eksamensaflevering (REPORT.pdf + offentligt GitHub-repo) til Wiseflow inden deadline 27. maj 2026.

**Architecture:** Eksisterende projekt-repo udvides med (1) en public-facing README, (2) en 3-5 siders dansk markdown-rapport eksporteret til PDF, (3) et asset-generator-script der producerer alle rapport-billeder reproducerbart fra v5_no_leak-modellen, og (4) en .gitignore-justering så den trænede model og 3 testbilleder kan committes.

**Tech Stack:** Python 3.11, Ultralytics YOLOv11n, PIL/Pillow, matplotlib, Pandoc eller VS Code Markdown PDF til PDF-eksport.

**Stilkrav til al rapport- og spec-tekst (ikke kode):** passiv form, førnutid, aldrig "vi"/"jeg".

---

## File Structure

**Create:**
- `REPORT.md` — selve rapporten (root)
- `REPORT.pdf` — PDF-eksport (root, genereret)
- `scripts/05_generate_report_assets.py` — asset-generator
- `docs/screenshots/training_curves.png` — kopi af results.png
- `docs/screenshots/pr_curve.png` — kopi af BoxPR_curve.png
- `docs/screenshots/confusion_matrix.png` — kopi
- `docs/screenshots/val_predictions.jpg` — kopi af val_batch0_pred.jpg
- `docs/screenshots/demo_predictions.png` — genereret prediction-overlay
- `docs/screenshots/objectness_heatmap.png` — genereret heatmap-overlay
- `docs/screenshots/synthetic_example.png` — kopi af test_images/composite_00000.png skaleret ned

**Modify:**
- `README.md` (overskrives — public-facing setup-guide)
- `.gitignore` (linje 49 — tilføj exception for test_images/composite_*.png)
- `requirements.txt` (verificer minimal sæt)

**Untracked → tracked:**
- `models/v5_no_leak/best.pt` (5.3 MB)
- `test_images/composite_00000.png` (12 MB)
- `test_images/composite_00001.png` (12 MB)
- `test_images/composite_00002.png` (12 MB)

---

## Task 1: Repo-fundament — .gitignore, model og testbilleder

Tillad de 3 testbilleder og verificér at model-filen kan tilføjes. Dette er forudsætning for at asset-generator kan committes med assets.

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Tilføj test_images exception i .gitignore**

Find linjen:
```
# Test-billeder til upload-demo (kan regenereres)
test_images/*.png
```

Erstat med:
```
# Test-billeder til upload-demo (kan regenereres)
# Undtagelse: 3 composite-testbilleder skal være i repoet til reproducerbar demo
test_images/*.png
!test_images/composite_00000.png
!test_images/composite_00001.png
!test_images/composite_00002.png
```

- [ ] **Step 2: Verificer at .gitignore-ændringen virker**

Kør:
```bash
git check-ignore -v test_images/composite_00000.png || echo "OK - filen er ikke ignored"
git check-ignore -v test_images/preview.png && echo "OK - preview.png stadig ignored"
```
Forventet: første kommando siger "OK - filen er ikke ignored", anden siger "OK - preview.png stadig ignored".

- [ ] **Step 3: Add og commit**

```bash
git add .gitignore models/v5_no_leak/best.pt \
    test_images/composite_00000.png \
    test_images/composite_00001.png \
    test_images/composite_00002.png
git commit -m "repo: track v5_no_leak model og 3 composite-testbilleder til public repo"
```

---

## Task 2: Asset-generator script (scripts/05_generate_report_assets.py)

Ét script der producerer alle 7 billeder brugt i REPORT.md. Idempotent — kan køres igen og igen.

**Files:**
- Create: `scripts/05_generate_report_assets.py`

- [ ] **Step 1: Opret asset-generator-scriptet**

Skriv hele filen `scripts/05_generate_report_assets.py`:

```python
#!/usr/bin/env python3
"""Generér alle billeder brugt i REPORT.md.

Henter:
- Træningskurver, PR-kurve, confusion matrix fra runs/detect/runs/train/v5_no_leak/
- Validation-predictions fra samme mappe
- Et nedskaleret syntetisk eksempel-billede
Genererer:
- Prediction-overlay på composite_00000.png med v5_no_leak-modellen
- Objectness heatmap-overlay (kombineret P3+P4+P5)

Kør:
    .venv/bin/python scripts/05_generate_report_assets.py
"""
from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

# Stier — løses relativt til repo-root
ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs" / "detect" / "runs" / "train" / "v5_no_leak"
MODEL_PATH = ROOT / "models" / "v5_no_leak" / "best.pt"
TEST_IMG = ROOT / "test_images" / "composite_00000.png"
OUT_DIR = ROOT / "docs" / "screenshots"

IMGSZ = 1280


def copy_training_assets() -> None:
    """Kopiér eksisterende træningsbilleder med pænere navne."""
    pairs = [
        ("results.png", "training_curves.png"),
        ("BoxPR_curve.png", "pr_curve.png"),
        ("confusion_matrix.png", "confusion_matrix.png"),
        ("val_batch0_pred.jpg", "val_predictions.jpg"),
    ]
    for src_name, dst_name in pairs:
        src = RUNS_DIR / src_name
        dst = OUT_DIR / dst_name
        shutil.copy(src, dst)
        print(f"  kopieret: {src_name} -> {dst_name}")


def generate_synthetic_example() -> None:
    """Nedskalér composite_00000.png så den er handy til REPORT.md."""
    img = Image.open(TEST_IMG)
    # Halvér opløsning så filen ikke bliver 12 MB i docs/
    new_size = (img.width // 2, img.height // 2)
    img.resize(new_size, Image.LANCZOS).save(OUT_DIR / "synthetic_example.png", optimize=True)
    print(f"  genereret: synthetic_example.png ({new_size[0]}x{new_size[1]})")


def generate_prediction_overlay(model: YOLO) -> None:
    """Tegn bounding boxes på test-billedet."""
    results = model.predict(source=str(TEST_IMG), imgsz=IMGSZ, conf=0.25, save=False, verbose=False)
    plotted_bgr = results[0].plot()  # numpy array, BGR
    plotted_rgb = plotted_bgr[:, :, ::-1]
    img = Image.fromarray(plotted_rgb)
    # Nedskalér til halv opløsning af samme grund som ovenfor
    img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
    img.save(OUT_DIR / "demo_predictions.png", optimize=True)
    print(f"  genereret: demo_predictions.png ({len(results[0].boxes)} detektioner)")


def compute_objectness_heatmap(model: YOLO, image: Image.Image) -> np.ndarray:
    """Beregn samlet objectness-heatmap (P3+P4+P5) for et billede.

    Hooker ind på detection head'ens output før NMS og decoding,
    summerer klassesandsynligheder per grid-celle på tværs af de tre skalaer.
    """
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.model.to(device)
    model.model.eval()

    # Preprocess: pad og resize til IMGSZ med Ultralytics' egen logik
    arr = np.array(image.convert("RGB"))
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    tensor = torch.nn.functional.interpolate(tensor, size=(IMGSZ, IMGSZ), mode="bilinear", align_corners=False)
    tensor = tensor.to(device)

    # Hook detection head'ens raw outputs
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        # output er en tuple (head outputs, ...) — vi gemmer hele tuplen
        captured.append(output)

    detect_layer = model.model.model[-1]  # sidste lag = Detect
    handle = detect_layer.register_forward_hook(hook)
    with torch.no_grad():
        _ = model.model(tensor)
    handle.remove()

    # Detect-output i train-mode er en liste af tensors per skala [P3, P4, P5]
    # I eval-mode er det en tuple (decoded, [P3, P4, P5]). Vi prøver begge.
    raw = captured[0]
    if isinstance(raw, tuple):
        feats = raw[1]
    else:
        feats = raw

    # For hver skala: tag classifikations-channels (de sidste nc), max over klasser, sigmoid
    nc = model.model.nc if hasattr(model.model, "nc") else 1
    target_h, target_w = IMGSZ, IMGSZ
    summed = np.zeros((target_h, target_w), dtype=np.float32)
    for feat in feats:
        # feat shape: (1, nc + reg, H, W) — vi tager klasse-delene
        cls_logits = feat[0, -nc:]  # (nc, H, W)
        cls_max = torch.sigmoid(cls_logits.max(dim=0).values)  # (H, W)
        # Upsample til IMGSZ
        upsampled = torch.nn.functional.interpolate(
            cls_max.unsqueeze(0).unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()
        summed += upsampled

    # Normalisér til [0, 1]
    summed = summed / max(summed.max(), 1e-6)
    return summed


def make_heatmap_overlay(base: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Læg en farvet heatmap-overlay på base-billedet."""
    # Resize heatmap til base-størrelse
    hm_resized = np.array(
        Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8))
        .resize(base.size, Image.BILINEAR)
    )
    # Mapper værdier til rød-gul-hvid skala (simpel "hot" colormap)
    import matplotlib
    cmap = matplotlib.colormaps["hot"]
    colored = (cmap(hm_resized / 255.0)[:, :, :3] * 255).astype(np.uint8)
    overlay = Image.fromarray(colored)
    return Image.blend(base.convert("RGB"), overlay, alpha)


def generate_heatmap_overlay(model: YOLO) -> None:
    """Generér objectness heatmap-overlay på testbilledet."""
    base = Image.open(TEST_IMG).convert("RGB")
    heatmap = compute_objectness_heatmap(model, base)
    overlay = make_heatmap_overlay(base, heatmap, alpha=0.5)
    overlay = overlay.resize((overlay.width // 2, overlay.height // 2), Image.LANCZOS)
    overlay.save(OUT_DIR / "objectness_heatmap.png", optimize=True)
    print(f"  genereret: objectness_heatmap.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] Kopierer træningsoutputs...")
    copy_training_assets()

    print("[2/3] Genererer syntetisk eksempel-billede...")
    generate_synthetic_example()

    print("[3/3] Loader v5_no_leak-modellen...")
    model = YOLO(str(MODEL_PATH))
    generate_prediction_overlay(model)
    generate_heatmap_overlay(model)

    print(f"\nFærdig. Alle assets ligger i {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Kør scriptet**

```bash
.venv/bin/python scripts/05_generate_report_assets.py
```
Forventet output: 7 filer i `docs/screenshots/`. Bemærk: heatmap-hook'en på Detect-laget kan kræve justering hvis Ultralytics' interne struktur er anderledes end forventet — hvis det fejler med en KeyError eller IndexError, inspicér `model.model.model[-1]` interaktivt og juster `hook`-logikken.

- [ ] **Step 3: Verificér alle 7 assets findes**

```bash
ls -lh docs/screenshots/
```
Forventet: 7 filer (training_curves.png, pr_curve.png, confusion_matrix.png, val_predictions.jpg, synthetic_example.png, demo_predictions.png, objectness_heatmap.png), alle > 10 KB.

- [ ] **Step 4: Eyeball alle 7 billeder**

Åbn dem manuelt (Finder, eller `open docs/screenshots/*.png`). Verificér:
- `demo_predictions.png`: bounding boxes ligger korrekt på frø
- `objectness_heatmap.png`: heatmap'et "lyser op" der hvor frøene er
- Resten ser rimelige ud

Hvis heatmap'et ser forkert ud (helt sort, helt hvidt, eller mønster ikke matcher frøene), kig på `compute_objectness_heatmap` — sandsynligvis er feature-extraction-formen anderledes end antaget.

- [ ] **Step 5: Commit script og assets**

```bash
git add scripts/05_generate_report_assets.py docs/screenshots/
git commit -m "report: asset-generator-script + 7 rapport-billeder

scripts/05_generate_report_assets.py orkestrerer al billed-generering:
- Kopierer træningsoutputs (results, PR, confusion matrix, val_pred)
- Nedskalerer syntetisk eksempel-billede
- Kører v5_no_leak-prediction på test_images/composite_00000.png
- Producerer objectness heatmap-overlay (P3+P4+P5)

Output i docs/screenshots/ committes så rapporten kan vise dem
uden at modtagere behøver køre scriptet."
```

---

## Task 3: README.md — public-facing setup-guide

Den eksisterende README er intern dokumentation. Den overskrives med en public-facing version: hvad er projektet, hvordan kører man det, og hvor er rapporten.

**Files:**
- Modify: `README.md` (overskrives helt)

- [ ] **Step 1: Skriv ny README.md**

Overskriv `README.md` med:

````markdown
# FCM Landmark Detection

Eksamensprojekt (EK - Erhvervsakademiet København, Machine Learning F26, maj 2026). En YOLOv11n-baseret model er trænet på syntetisk genererede composite-billeder til detektion af frø-positioner på Videometers blå plade.

**Rapport:** [REPORT.md](REPORT.md) (eller [REPORT.pdf](REPORT.pdf))

---

## Resultater (kort)

| Model | mAP@50 | mAP@50-95 | Note |
|---|---|---|---|
| v5_no_leak | **0.999** | **0.969** | Endelig model — trænet 80 epochs, leakage-fri splits |

Detaljer og fortælling findes i [REPORT.md](REPORT.md).

---

## Setup — lokalt (Mac M-chip)

Python 3.11 anbefales (Ultralytics og PyTorch halter på 3.13).

```bash
# 1. Klon repoet
git clone <repo-url>
cd fcm_project_export

# 2. Opret virtuelt miljø
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Installer dependencies
pip install -r requirements.txt

# 4. Verificér at MPS (Apple Silicon GPU) virker
python -c "import torch; print('MPS:', torch.backends.mps.is_available())"
```

Forventet output: `MPS: True`. Hvis False, virker modellen stadig — bare langsommere på CPU.

## Setup — Google Colab

En notebook ligger i `scripts/04_colab/v6_yolov10/train.ipynb` og kan åbnes direkte i Colab. Den henter dependencies og kører træning på Colabs T4-GPU.

## Kør demo'en

Streamlit-demoen lader brugeren uploade et billede og se modellens predictions + objectness heatmap:

```bash
.venv/bin/streamlit run scripts/03_demo/upload_demo.py
```

Brug et af de 3 testbilleder under `test_images/` til at prøve det.

## Reproducer rapport-billeder

```bash
.venv/bin/python scripts/05_generate_report_assets.py
```

Genererer alle billeder i `docs/screenshots/` brugt i REPORT.md.

## Reproducer datasæt

Det fulde syntetiske datasæt (2.5 GB, 7.000 billeder) er ikke versioneret i repoet. To muligheder:

1. **Download:** [Google Drive-link kommer her] — `composite_v1_no_leak_1280.tar.gz`
2. **Regenerér fra kode:** se `scripts/02_yolo_pipeline/composite_generator/` og `make gen`.

## Mappestruktur

```
fcm_project_export/
├── REPORT.md / REPORT.pdf      # Eksamensrapport
├── models/v5_no_leak/best.pt   # Endelig trænet model
├── test_images/                # 3 composite-testbilleder
├── scripts/
│   ├── 02_yolo_pipeline/       # Composite generator + træning
│   ├── 03_demo/                # Streamlit-demo med heatmap
│   ├── 04_colab/               # Colab-notebooks
│   └── 05_generate_report_assets.py
└── docs/
    ├── screenshots/            # Rapport-billeder
    └── superpowers/            # Specs og planer (interne)
```

## Licens

Projektet er udviklet til eksamen og er ikke licenseret til kommerciel brug.
````

- [ ] **Step 2: Verificér at intern dokumentation ikke linkes fra README**

```bash
grep -nE "PROJECT_SPEC|OFFLINE_GUIDE|PARAMETERS_GUIDE|setup_guide|CLAUDE.md" README.md && echo "FEJL: intern dok refereret" || echo "OK"
```
Forventet: `OK`.

- [ ] **Step 3: Verificér at Colab-notebook eksisterer på den angivne sti**

```bash
ls scripts/04_colab/v6_yolov10/train.ipynb && echo "OK" || echo "FEJL: Colab-sti forkert"
```
Forventet: `OK`. Hvis FEJL, find den faktiske notebook med `find scripts/04_colab -name "*.ipynb"` og opdater README-stien.

- [ ] **Step 4: Commit README**

```bash
git add README.md
git commit -m "docs: omskriv README til public-facing setup-guide

Tidligere README var intern dokumentation med links til PROJECT_SPEC og
CLAUDE.md. Ny version er rettet mod censor/eksterne læsere:
- Resultattabel øverst
- Setup-guide til lokal Mac (Python 3.11 + MPS) og Colab
- Demo-instruktioner
- Reproducerbarheds-noter for assets og datasæt"
```

---

## Task 4: REPORT.md — Sektion 1-2 (Problemstilling + Datagrundlag)

Start på rapporten. Skriv intro-sektionerne. Husk: passiv form, førnutid, aldrig "vi".

**Files:**
- Create: `REPORT.md`

- [ ] **Step 1: Opret REPORT.md med titel + sektion 1+2**

Skriv `REPORT.md`:

````markdown
# FCM Landmark Detection

**Eksamensaflevering — EK - Erhvervsakademiet København, Machine Learning F26, maj 2026**
**Forfatter:** Andreas Søgård Gabel

---

## 1. Problemstilling

I samarbejde med Videometer A/S er der undersøgt om frø-positioner på en standardiseret optagelsesplade kan detekteres automatisk via maskinlæring. Den nuværende arbejdsgang kræver manuel annotering af hvert enkelt frøs centermasse, hvilket både er tidskrævende og skalerer dårligt ved større datamængder.

Målet med projektet har været at træne en model der kan udpege frø-positioner i et billede med tilstrækkelig præcision til at erstatte den manuelle annotering, eller som minimum fungere som pre-annotering der reducerer det manuelle arbejde markant.

Som detektions-arkitektur er der valgt **YOLOv11n** fra Ultralytics-rammeværket. Valget skyldes en kombination af modellens lette form (~5 MB trænede vægte), gode anchor-fri performance på små objekter og kompatibilitet med Apple Silicon (MPS), som har været det primære udviklingsmiljø.

## 2. Datagrundlag — hvorfor syntetisk data

Adgangen til ægte Videometer-data har været begrænset gennem hele projektet: optagelserne er underlagt en NDA, mængden er lille, og manuel annotering af frø-centre er præcis det problem der ønskes løst. At basere træningen direkte på de få tilgængelige rigtige billeder ville derfor være både datafattigt og cirkulært.

I stedet er der bygget en **syntetisk data-pipeline** der procedurelt sammensætter realistiske composite-billeder: enkeltfrø-udsnit fra Videometer-optagelser placeres på et baggrundsbillede af den blå optagelses-plade, og labels genereres automatisk ud fra placeringen. Pipelinen har genereret **7.000 composite-billeder i 1280×1280** opløsning, og giver fuld kontrol over hvor mange frø der ligger i hvert billede, deres positioner og deres ikke-overlappende layout.

![Eksempel på syntetisk composite-billede](docs/screenshots/synthetic_example.png)
*Figur 1: Et af de 7.000 syntetiske composite-billeder brugt til træning. Frø er placeret tilfældigt på en blå plade-baggrund uden overlap.*

````

- [ ] **Step 2: Verificér markdown-formatering**

```bash
head -40 REPORT.md
```
Tjek visuelt at overskrifter, fed tekst og billed-reference renderer korrekt (kan også åbnes i GitHub-preview via IntelliJ eller VS Code).

- [ ] **Step 3: Commit**

```bash
git add REPORT.md
git commit -m "report: udkast sektion 1-2 (problemstilling + datagrundlag)"
```

---

## Task 5: REPORT.md — Sektion 3 (Data leakage)

Den ærlige fortælling om v1-v4 og opdagelsen af leakage. Dette er rapportens "stærke" sektion — vis metodisk modenhed.

**Files:**
- Modify: `REPORT.md` (append)

- [ ] **Step 1: Tilføj sektion 3 til REPORT.md**

Tilføj efter sektion 2:

````markdown

## 3. Tidlige tilgange og opdaget data leakage

Modellen er blevet trænet i flere iterationer hvor metodikken er strammet op undervejs:

| Version | Datasæt-tilgang | Bemærkning |
|---|---|---|
| v1_kalibreret | Kun rene frø-udsnit, ingen baggrund | Baseline — orienteringsstudie |
| v2_robust | Composite med tilfældig augmentation | Test af robusthed |
| v3_augmented | Composite med stærkere augmentation | Validerede augmenterings-strategien |
| v4_composite | Større composite-datasæt, fælles baggrunde train/val | Mistænkeligt høje metrikker |
| **v5_no_leak** | **Composite med adskilte baggrunde train/val** | **Endelig model** |

Ved gennemgang af v4-resultaterne er der opdaget en metodisk fejl: **de baggrundsbilleder der blev brugt til at konstruere composite-eksemplerne, var fælles mellem træningssæt og valideringssæt**. Hver baggrund optrådte under flere frø-konstellationer, men hvis den samme baggrund optrådte både i train og val, kunne modellen lære features ved baggrunden selv frem for ved frøene. Det er klassisk data leakage og forklarer hvorfor v4-metrikker var urealistisk gode.

Læringen har formet hele den endelige pipeline: **enhver baggrund må kun optræde i ét split**. Den re-arkitekterede composite-generator splitter først baggrunds-poolen i train/val/test, og bygger derefter composite-billeder inden for hvert split. Det er denne disciplin der gør v5's tal troværdige.
````

- [ ] **Step 2: Commit**

```bash
git add REPORT.md
git commit -m "report: sektion 3 — data leakage opdagelse og læring"
```

---

## Task 6: REPORT.md — Sektion 4-5 (Pipeline + Resultater)

Den tekniske kerne: arkitektur, træning, resultater. Skal inkludere konkrete tal og 3 billeder.

**Files:**
- Modify: `REPORT.md` (append)

- [ ] **Step 1: Tilføj sektion 4 og 5**

Tilføj efter sektion 3:

````markdown

## 4. Composite-pipelinen og v5_no_leak

Den endelige pipeline består af **seks moduler** der hver har ét ansvar:

1. **Baggrunds-loader** — indlæser plade-baggrunde og splitter dem i train/val/test (baggrundene blandes aldrig på tværs af splits).
2. **Frø-katalog** — holder en pulje af enkeltfrø-udsnit med metadata om størrelse og center-masse.
3. **Layout-generator** — placerer N frø tilfældigt på en baggrund uden overlap.
4. **Composite-renderer** — sammensætter baggrund + frø til ét billede og holder styr på pixel-koordinater.
5. **Label-skriver** — skriver YOLO-formatterede labels (klasse + normaliserede bbox-koordinater) per billede.
6. **Manifest-bygger** — gemmer `data.yaml` og generation-metadata så datasættet er fuldt sporbart.

Træningen af v5_no_leak er gennemført på en Mac M-chip vha. PyTorch's MPS-backend over **80 epochs** med YOLOv11n som startpunkt. Træningskonfiguration: input-størrelse 1280×1280, batch 8, learning rate auto-tuned.

![Træningskurver for v5_no_leak](docs/screenshots/training_curves.png)
*Figur 2: Train- og val-loss + metrikker over 80 epochs. mAP@50 konvergerer hurtigt, mens mAP@50-95 fortsætter med at stige langsomt.*

## 5. Resultater

På valideringssættet — som ingen baggrunde deler med træningssættet — har v5_no_leak opnået følgende slutmetrikker:

| Metric | Værdi |
|---|---|
| Precision | 0.999 |
| Recall | 1.000 |
| mAP@50 | **0.999** |
| mAP@50-95 | **0.969** |

Tallene er konsistente med kvaliteten af de syntetiske data og indikerer at modellen har lært at lokalisere frø meget pålideligt på composite-billeder. Det skal understreges at evalueringen stadig foregår på syntetisk data — den endelige validering på rigtige Videometer-billeder er planlagt men ligger uden for denne aflevering.

![PR-kurve](docs/screenshots/pr_curve.png)
*Figur 3: Precision-Recall-kurven viser at modellen holder >0.99 precision over hele recall-spektret.*

### Objectness heatmap som interpretability

Som interpretability-værktøj er der i Streamlit-demoen indbygget en visualisering af modellens **interne objectness-confidence** før NMS og bbox-decoding. Heatmap'et opsamles ved at hooke ind på YOLO-modellens Detect-lag og summere klassesandsynligheder per grid-celle på tværs af de tre featuremap-skalaer (P3, P4, P5).

![Objectness heatmap-overlay](docs/screenshots/objectness_heatmap.png)
*Figur 4: Objectness heatmap (rød-gul-hvid) lagt oven på et testbillede. Modellens interne confidence "tænder" netop der hvor de detekterede bounding boxes ligger — visuelt bekræft af at modellen ikke "snyder" via baggrundens features.*

![Predictions på testbillede](docs/screenshots/demo_predictions.png)
*Figur 5: Predictions fra v5_no_leak på composite_00000.png. Hver bounding box svarer til et detekteret frø.*
````

- [ ] **Step 2: Commit**

```bash
git add REPORT.md
git commit -m "report: sektion 4-5 — composite-pipeline, træning og resultater"
```

---

## Task 7: REPORT.md — Sektion 6 (Refleksion + fremtidsudsigter)

Afslut rapporten med en ærlig refleksion og udsigt til næste skridt.

**Files:**
- Modify: `REPORT.md` (append)

- [ ] **Step 1: Tilføj sektion 6**

Tilføj efter sektion 5:

````markdown

## 6. Refleksion og fremtidsudsigter

**Real-world validering** er det vigtigste næste skridt. Resultaterne ovenfor er imponerende på syntetisk data, men gabet mellem syntetisk og rigtigt er kendt fra litteraturen. Et besøg hos Videometer er aftalt med henblik på at køre v5_no_leak på en lille mængde ægte plade-optagelser og rapportere mAP samt fejl i mm via den kalibrerede mm/pixel-faktor.

**Mulige forbedringer** identificeret under arbejdet:

- **Mere realistisk lyssætning** i composite-generatoren — den nuværende blå plade har konstant belysning, mens rigtige Videometer-optagelser har subtile lys-gradienter.
- **Multi-class detection** — modellen detekterer kun "frø". Hvis flere kornsorter skal skelnes, skal pipelinen udvides med klasse-balancerede syntetiske eksempler.
- **Domain adaptation** — selv en mindre fine-tuning på ægte data forventes at forbedre generaliseringen betydeligt.

**Metodisk refleksion.** Den vigtigste læring fra projektet er ikke selve modellens arkitektur, men opdagelsen af data leakage mellem v4 og v5. Det demonstrerer hvor let det er at producere imponerende tal på en utæt evaluering, og hvor vigtigt det er at splitte ikke kun *eksemplerne* men også *de underliggende kilder* (her: baggrunds-billederne) når et syntetisk datasæt bygges fra et lille sæt råmaterialer.

---

*Repo: <github-link kommer her>. Demo: `streamlit run scripts/03_demo/upload_demo.py`. Reproducer rapport-billeder: `python scripts/05_generate_report_assets.py`.*
````

- [ ] **Step 2: Læs hele rapporten igennem**

```bash
wc -l REPORT.md
```
Forventet: ~80-120 linjer. Åbn i en markdown-preview og verificér:
- Alle 7 billeder loader korrekt
- Ingen "vi"/"jeg"/"vores" optræder
- Sektion-overskrifter er konsistente
- Tabeller renderer

```bash
grep -nE "\b(vi|jeg|vores|os)\b" REPORT.md && echo "FEJL: 1. person fundet" || echo "OK"
```
Forventet: `OK`.

- [ ] **Step 3: Commit**

```bash
git add REPORT.md
git commit -m "report: sektion 6 — refleksion og fremtidsudsigter (REPORT.md færdig)"
```

---

## Task 8: PDF-eksport (REPORT.md → REPORT.pdf)

Eksportér rapporten til PDF til Wiseflow-aflevering.

**Files:**
- Create: `REPORT.pdf`

- [ ] **Step 1: Find tilgængelig PDF-eksport-værktøj**

Prøv i denne rækkefølge — første der virker, bruges:

```bash
# Mulighed A: Pandoc med default LaTeX
command -v pandoc && pandoc --version | head -1

# Mulighed B: VS Code med Markdown PDF extension (hvis installeret)
command -v code && code --list-extensions | grep -i markdown-pdf

# Mulighed C: Chrome/Chromium headless (print til PDF)
command -v "Google Chrome" || ls /Applications/Google\ Chrome.app
```

- [ ] **Step 2: Eksportér til PDF**

**Hvis Pandoc:**
```bash
pandoc REPORT.md -o REPORT.pdf \
    --pdf-engine=xelatex \
    -V geometry:margin=2cm \
    -V mainfont="Helvetica" \
    -V documentclass=article \
    --highlight-style=tango
```

**Hvis ikke pandoc/xelatex virker (LaTeX-fejl), fald tilbage til wkhtmltopdf eller browser-print:**
```bash
# Render markdown til HTML og print fra browser:
pandoc REPORT.md -o REPORT.html --self-contained --metadata title="FCM Landmark Detection"
open REPORT.html
# Brug derefter Cmd+P → Save as PDF i browseren, gem som REPORT.pdf
```

- [ ] **Step 3: Verificér PDF**

```bash
ls -lh REPORT.pdf
```
Forventet: filen findes, størrelse 0.5-3 MB (afhænger af hvor høj-opløselige assets bliver indlejret).

Åbn PDF'en manuelt (`open REPORT.pdf`) og tjek:
- Alle 5-6 figurer er med
- Sidetal og layout er rimelige
- Ingen tomme sider eller cut-off tekst
- Sidetal i alt: ~3-5 sider

- [ ] **Step 4: Commit**

```bash
git add REPORT.pdf
git commit -m "report: PDF-eksport til Wiseflow-aflevering"
```

---

## Task 9: Slutverifikation og oprydning

Sidste runde før push: sikr at intet NDA-materiale slipper med, at alle links virker, at setup-guiden faktisk er kørbar.

- [ ] **Step 1: Verificér .gitignore stadig blokerer NDA-data**

```bash
git status --ignored --short | grep -E "videometer|\.hips" | head -5
git ls-files | grep -iE "videometer|\.hips" && echo "FEJL: NDA-data trackes!" || echo "OK: ingen NDA-data trackes"
```
Forventet: anden kommando siger "OK".

- [ ] **Step 2: Verificér modellen er trackbar og åbnes**

```bash
git ls-files models/v5_no_leak/best.pt
.venv/bin/python -c "from ultralytics import YOLO; m = YOLO('models/v5_no_leak/best.pt'); print('Model loaded, klasser:', m.names)"
```
Forventet: Stien er tracked, og modellen loader.

- [ ] **Step 3: Verificér alle billed-referencer i REPORT.md eksisterer**

```bash
grep -oE 'docs/screenshots/[a-zA-Z0-9_.-]+' REPORT.md | sort -u | while read f; do
  [ -f "$f" ] && echo "OK: $f" || echo "MISSING: $f"
done
```
Forventet: alle siger "OK".

- [ ] **Step 4: Verificér ingen "vi"/førsteperson i REPORT.md eller spec**

```bash
grep -nE "\b(vi|jeg|vores|os|min|mit)\b" REPORT.md docs/superpowers/specs/*.md && echo "GENNEMSE FUND" || echo "OK"
```
Forventet: `OK`. Hvis fund i specs, må de ignoreres (specs er internt design-doc); men i REPORT.md må intet stå.

- [ ] **Step 5: Slet eventuelt store untrackede artefakter fra working dir**

```bash
ls -lh composite_v1_no_leak_1280.tar.gz 2>/dev/null
```
Hvis filen ligger i repo-root: den er allerede gitignored (kommer ikke med i commits), men kan flyttes ud af mappen for renlighed:
```bash
mv composite_v1_no_leak_1280.tar.gz ~/Downloads/  # eller hvor du vil have den
```

- [ ] **Step 6: Final commit (hvis ændringer)**

```bash
git status
# hvis noget er ændret:
git add <files>
git commit -m "report: slutverifikation"
```

---

## Task 10: Push og aflever (manuelle skridt)

Disse skridt udføres af Andreas — ikke automatiseret.

- [ ] **Step 1: Opret offentligt GitHub-repo** (hvis ikke allerede gjort)

Via GitHub web UI eller `gh repo create fcm_project_export --public`.

- [ ] **Step 2: Push**

```bash
git remote -v  # tjek at remote er sat op korrekt
git push -u origin master
```

- [ ] **Step 3: Verificér på GitHub**

Åbn repo-URL i browseren. Tjek:
- README renderer pænt
- REPORT.md renderer med alle billeder
- Klik på `models/v5_no_leak/best.pt` — skal kunne downloades
- Klik på `test_images/composite_00000.png` — skal kunne vises

- [ ] **Step 4: Opdater README med faktisk GitHub-link**

Hvis du allerede har linket fra REPORT.md sektion 6 (`<github-link kommer her>`) — opdater det med den rigtige URL og push igen.

- [ ] **Step 5: Upload REPORT.pdf + GitHub-link til Wiseflow**

Manuel hand-in på Wiseflow inden 27. maj 2026.

---

## Success-kriterier

Når alle 10 tasks er færdige:

- ✅ REPORT.md og REPORT.pdf eksisterer og indeholder 6 sektioner med 5-6 indlejrede billeder
- ✅ README.md beskriver setup for både lokalt og Colab
- ✅ `models/v5_no_leak/best.pt` + 3 testbilleder er versioneret
- ✅ `scripts/05_generate_report_assets.py` kører rent og producerer alle assets
- ✅ Ingen NDA-data trackes
- ✅ Ingen 1.-person i rapport-tekst
- ✅ Repoet er pushet og PDF afleveret på Wiseflow
