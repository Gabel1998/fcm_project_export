# Instruktioner til Claude Code

## Læs først
1. `PROJECT_SPEC.md` — fuld projekt-kontekst
2. Denne fil

## Tone og stil
- **Dansk er primært sprog** med brugeren. Engelsk er OK i kodekommentarer.
- **Brug "du"-form**, ikke "De".
- **Vær ærlig.** Fortæl brugeren hvis noget ikke virker, hvis en plan har risici, eller hvis vedkommende har misforstået noget. Ikke smiger.
- **Skub tilbage** hvis brugeren beder om noget der ikke giver mening for projektet.

## Arbejdsstil
- **Kør ting selv** når du kan. Brugeren foretrækker at se resultater frem for instruktioner.
- **Test din kode** før du leverer den. Ikke alt skal "tilsyneladende virke" — vis at det faktisk virker.
- **Forklar hver linje på dansk** i kode-kommentarer. Brugeren skal kunne forklare alt til eksamen.

## Tekniske præferencer
- **Python 3.10 eller 3.11** (ikke nyere — Ultralytics og PyTorch halter på 3.13)
- **PyTorch med MPS** for Apple Silicon GPU acceleration: `device='mps'`
- **Ultralytics YOLOv11n** som hovedmodel
- **Pandas** til Excel-læsning
- **IntelliJ IDEA** som IDE (configure som Python-projekt)

## Projekt-struktur (forslag)
```
fcm_landmark_detection/
├── data/
│   ├── synthetic/        # syntetisk genererede data
│   ├── videometer_raw/   # de rå HIPS-filer (gitignored!)
│   └── videometer_processed/  # konverterede PNG + YOLO labels
├── scripts/
│   ├── 01_generate_synthetic.py
│   ├── 02_convert_to_yolo.py
│   ├── 03_train_yolo.py
│   ├── 04_evaluate_landmarks.py
│   └── 05_load_videometer_data.py    # NY - skal bygges
├── models/                # gemte .pt og .onnx filer
├── results/               # plots, metrics, demo-billeder
├── notebooks/             # eventuelt jupyter til eksperimenter
├── requirements.txt
├── README.md              # for brugerens GitHub
└── .gitignore             # vigtig: ekskluder data/videometer_*
```

## .gitignore (vigtigt for NDA!)
```
data/videometer_raw/
data/videometer_processed/
*.hips
runs/
__pycache__/
.idea/
.venv/
```

## Vigtige opgaver i prioritetsorden

### 1. Setup lokalt miljø (gør først)
- Conda eller venv til Python 3.11
- Installer: ultralytics, pandas, openpyxl, pillow, matplotlib, numpy
- Test at MPS virker: `torch.backends.mps.is_available()` skal være True
- Setup IntelliJ: Configure Project SDK til det rette Python-miljø

### 2. Migrér eksisterende scripts (let)
- Tag scripts/02_yolo_pipeline/ filerne
- Test at de kører lokalt (på syntetiske data først)
- Sammenlign performance Mac MPS vs. Colab GPU

### 3. Byg Videometer-data-loader (når data kommer)
**Inputs:**
- Excel-fil med kolonner: Blob ID, Y Center mass, X Center mass, Source Image ID
- Mappe med billeder (forhåbentlig PNG/TIFF — alternativt HIPS)

**Output:**
- YOLO-format dataset (samme struktur som syntetisk)

**Pseudo-kode:**
```python
import pandas as pd
df = pd.read_excel("Videometer_data.xlsx")

# Group by image - hvert billede har flere frø
for image_id, group in df.groupby("Source Image ID"):
    image_path = group["File Path"].iloc[0]
    # Load image (HIPS eller PNG)
    img = load_image(image_path)
    H, W = img.shape[:2]

    # Skriv YOLO label
    label_lines = []
    for _, row in group.iterrows():
        cx = row["X Center mass"] / W
        cy = row["Y Center mass"] / H
        bbox_size = 18 / W  # tilpas til frø-størrelse
        label_lines.append(f"0 {cx} {cy} {bbox_size} {bbox_size}")

    save_yolo_label(image_id, label_lines)
    save_image_as_png(image_id, img)
```

### 4. Træn på rigtige data
Samme pipeline som syntetisk — bare peg på den nye `data.yaml`.

### 5. Evaluering med metric units
**VIGTIGT:** Videometer leverer mm/pixel pr. billede. Brug det til at rapportere fejl i mm, ikke kun pixels. Det er guld til eksamen.

## Hvad der IKKE skal gøres

- ❌ **Ikke heatmaps eller PSNR** — vi har bevidst valgt dem fra
- ❌ **Ikke FCN-arkitektur** — Jone har eksplicit frarådet det
- ❌ **Ikke ConvNeXt** — det er classification, ikke detection
- ❌ **Ikke upload data til Colab** — NDA og 14GB
- ❌ **Ikke gøre koden "smart"** hvis det går ud over forklaringsbarheden

## Kommunikation under arbejdet

Når du beder brugeren om input:
- Stil **én ting ad gangen**
- Tilbyd **konkrete valg** (ikke "hvad vil du gøre?")
- Forklar **hvorfor** du spørger

Når du leverer kode:
- **Vis test-resultatet** før brugeren skal køre
- **Forklar det vigtigste** i en kort tekst
- **Frag brugeren** om noget skal justeres før vi går videre

## Eksamens-perspektivet altid

Hver beslutning skal vurderes ud fra:
1. **Kan brugeren forklare det til eksamen?**
2. **Er det forankret i kursets undervisning?**
3. **Tilføjer det reel værdi til projektet?**

Hvis svaret er nej til to af de tre — drop det.
