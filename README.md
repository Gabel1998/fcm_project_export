# FCM Landmark Detection

Eksamensprojekt (KEA, Maskinlæring, maj 2026). En YOLOv11n-baseret model er trænet på syntetisk genererede composite-billeder til detektion af frø-positioner på Videometers blå plade.

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
