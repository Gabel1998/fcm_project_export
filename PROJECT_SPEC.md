# FCM Landmark Detection - Project Specification

> **Til Claude Code:** Læs dette dokument først. Det indeholder hele projektets kontekst, beslutninger, og status. Spørg brugeren før du foretager større ændringer i retning.

---

## 1. Projekt-overblik

**Hvad:** Detect frø/kornkerner på plader fra Videometer's spektrale imaging-system, og finde center-koordinater for hvert frø (landmark detection).

**Hvorfor:** Eksamensprojekt på EK - Erhvervsakademiet København Machine Learning Elective, forår 2026. Samtidig en reel industri-case fra Videometer A/S (dansk virksomhed der laver spektral imaging til frø/korn-analyse).

**Underviser:** Jone Eikholm (jone@ek.dk)
**Industri-partner:** Videometer A/S (videometer.dk)
**Eksamen:** Uge 24 (11.–16. juni 2026), 10 min præsentation + 10–15 min spørgsmål

---

## 2. Kursus-kontekst og krav

### Eksamenskrav (fra kursusplanen)
1. Briefly describe the problem you want to solve
2. Find a dataset (e.g., from kaggle.com)
3. Prepare the data
4. Train a model
5. Evaluate the quality of the model

**Extras:**
- Publish the model to a webserver (Java Spring + ONNX)
- Demonstrate in Colab or any Python editor

**Tilladte metoder:** DNN, CNN, GPT finetuning. Andre metoder kræver godkendelse fra Jone.

### Performance-krav (fra YOLO-tutorial)
- Precision (P) > 0.5
- Recall (R) > 0.3
- mAP@50 > 0.2

### Kerneprincip for eksamen
Brugeren skal kunne **forklare hver linje af sin kode**. Det er det vigtigste evalueringskriterium efter projektets faglige indhold.

### DNN-fundament der altid testes
Forward propagation, backpropagation, gradient, learning rate, momentum, ReLU, sigmoid, softmax, MSE, categorical_crossentropy, train/test split, overfitting, regularization.

---

## 3. Vigtige beslutninger og deres begrundelser

### Beslutning 1: YOLOv11 i stedet for FCN
**Kontekst:** FCM-dokumentet fra Videometer beskriver løsningen som FCN (Fully Convolutional Network).

**Jone's feedback (citat):** *"Dokumentet virker underligt: de anbefaler FCN som stammer fra 2015. Vi har i undervisningen brugt YOLOv11n eller ConvNeXtTiny, som er bedre modeller."*

**Beslutning:** Skift fra FCN til YOLOv11n.

**Begrundelse:**
- Følger underviserens råd
- YOLOv11 er state-of-the-art (2024) til object/keypoint detection
- ConvNeXt er primært classification — ikke optimal til detection af mange objekter
- YOLOv11 producerer bbox + confidence, vi konverterer center-of-bbox = landmark

### Beslutning 2: Syntetiske data først, derefter rigtige Videometer-data
**Begrundelse:**
- Industri-data tager tid at få (NDA, formater, etc.)
- Validering af pipeline på data hvor vi kender ground truth
- Sikkerhedsline hvis Videometer-data forsinkes
- Demonstrerer professionel ML-praksis til eksamen
- Den samme pipeline genbruges — kun data-loaderen ændres

### Beslutning 3: Drop heatmap + PSNR
**Kontekst:** FCM-dokumentet nævner heatmap-output og PSNR som evalueringsmetric.

**Beslutning:** Brug ikke heatmaps eller PSNR.

**Begrundelse:**
- Heatmaps er FCN-specifikt (ikke YOLO-naturligt)
- PSNR måler heatmap-kvalitet — vi har ingen heatmaps
- Vi har bedre metrics: P/R/F1 på landmark-niveau + center-præcision i mm
- FCM-dokumentet siger "average distance OR PSNR" — vi har valgt det første

### Beslutning 4: Lokalt projekt på Mac (ikke Colab)
**Begrundelse:**
- 14 GB Videometer-data (NDA-overvejelser)
- Videometer-toolbox kræver Windows (men vi får sandsynligvis konverterede PNG/TIFF)
- Stabil arbejdsgang uden session-disconnects
- Apple Silicon (M-chip) understøtter PyTorch via MPS

---

## 4. Brugerens setup

- **OS:** Mac med Apple Silicon (M1/M2/M3)
- **GPU:** Apple Silicon (skal bruge PyTorch MPS-acceleration)
- **IDE:** IntelliJ IDEA
- **Python:** Anbefal 3.10 eller 3.11 (3.13 er for nyt)

---

## 5. Data-spec (fra Videometer)

### Hvad brugeren får
- **14 GB total**
- **Fulde billeder** i HIPS-format (multispektrale)
- **Blobs** (allerede ekstraherede individuelle frø)
- **Excel-fil** med ground truth: `Blob ID`, `Y Center mass`, `X Center mass`, `Source Image ID`, `File Path`, `Predicted` osv.

### Tekniske detaljer
- **Format:** HIPS (Videometer's eget format)
- **Enhed:** Reflektans (kalibreret, [0, 1])
- **Skala:** mm/pixel-faktor følger med hvert billede (metric units!)
- **Multispektral:** Op til 19 bånd fra 365–970 nm

### Frø-karakteristika (bekræftet af brugeren)
- **Type:** Aflange (ris/hvede-lignende)
- **Antal pr. plade:** Tætpakkede, men minimalt overlap (frø rystet ud)
- **Belysning:** Top og bund (ingen skygger)

### HIPS-toolbox issue
- Videometer's Python-toolbox kræver Windows
- **Workaround:** Bed Videometer eksportere som PNG/TIFF, eller konverter på Windows-maskine på EK - Erhvervsakademiet København
- **Excel-filen kan læses direkte uden toolbox** — så al ground truth er tilgængelig på Mac

---

## 6. Hvad vi har bygget hidtil

### Pipeline-status
| Step | Beskrivelse | Status |
|------|-------------|--------|
| 1 | Simpelt syntetisk datasæt (cirkler) | ✅ Færdigt — kasseret efter beslutning 1 |
| 1B | Realistisk syntetisk datasæt (aflange frø) | ✅ Færdigt |
| 2 | YOLO-format konvertering | ✅ Færdigt |
| 3 | YOLOv11n træning | ✅ Færdigt — meget gode resultater |
| 4 | Predict + landmark-evaluering | ✅ Færdigt (men brugeren har ikke kørt det endnu pga. path-fejl) |
| 5 | HIPS/Excel data-loader | ⏳ Mangler — venter på data |
| 6 | Træning på rigtige data | ⏳ Mangler |
| 7 | ONNX-eksport | ⏳ Mangler (extra) |
| 8 | Java Spring webserver | ⏳ Mangler (extra) |

### Resultater på syntetiske data (YOLOv11n)
| Metric | Værdi | Krav | Status |
|--------|-------|------|--------|
| Precision | 0.9932 | > 0.5 | ✅ |
| Recall | 0.9992 | > 0.3 | ✅ |
| mAP@50 | 0.9947 | > 0.2 | ✅ |
| mAP@50-95 | 0.7162 | (bonus) | God |

Træningstid: 0.034 timer (~2 min på Colab T4 GPU) for 39 epochs (early stopping ved epoch 29).

### Datasæt-statistik
- 300 syntetiske billeder (256×256 pixels)
- 17.580 frø totalt
- ~58 frø pr. billede gennemsnitligt
- 0.05 mm/pixel skala (matcher typisk VideometerLab)
- Train/val split: 240/60

---

## 7. Næste skridt

### Umiddelbart (denne uge)
1. **Setup lokalt projekt på Mac** med IntelliJ + Python miljø
2. **Test YOLO med MPS** (Apple Silicon GPU)
3. **Migrér eksisterende scripts** fra Colab til lokalt
4. **Vente på data** fra Videometer / Jone-møde

### Når data lander
1. **Læs Excel-fil** med pandas → uddrag (image_id, x, y) tuples
2. **Konvertér HIPS → standard format** (eller få det leveret som PNG/TIFF)
3. **Skriv data-loader** der laver YOLO-format fra Excel-koordinater
4. **Træn YOLOv11n** på rigtige data
5. **Evaluer** med samme pipeline som på syntetiske data
6. **Rapport metrics i mm** (ikke kun pixels) — brug mm/pixel fra Videometer-headers

### Til eksamen
1. **Slides** med fortælling: problem → datasæt → metode → resultater → demo
2. **Live demo** — predict på et frisk billede
3. **Sammenligning syntetisk vs. rigtige data** (stærk fortælling)
4. **Forberede DNN-grundspørgsmål**

### Extras (hvis tid)
- ONNX-eksport
- Java Spring webserver med ONNX

---

## 8. Vigtige tekniske detaljer

### YOLO-format
```
dataset/
├── images/
│   ├── train/   ← .png billeder
│   └── val/
├── labels/
│   ├── train/   ← .txt filer (én per billede)
│   └── val/
└── data.yaml
```

Hver `.txt` label-fil har formatet:
```
class_id x_center y_center width height
```
Hvor alle koordinater er **normaliseret** til [0, 1] (relative til billed-størrelse).

For frø-detection: `class_id = 0` (kun "seed"), bbox-størrelse 18×18 pixels rundt om hvert center.

### Custom loss funktion (kun relevant for FCN — historisk)
Hvis FCN-rute genoptages, bruges weighted MSE for at håndtere sparse heatmaps:
```python
def weighted_mse(y_true, y_pred):
    weights = tf.where(y_true > 0.1, 100.0, 1.0)
    return tf.reduce_mean(tf.square(y_true - y_pred) * weights)
```

### YOLOv11 konfiguration brugt
```python
EPOCHS = 50
IMG_SIZE = 256
BATCH_SIZE = 16
PATIENCE = 10  # early stopping
flipud = 0.5
fliplr = 0.5
degrees = 180.0  # frø kan ligge i alle retninger
```

---

## 9. Kommunikations-historik

### Med Jone
- Spurgt om FCM Landmark Detection-emnet og fået "fortsætter du" som svar
- Aftale om GitHub-link på ITSLearning (privat repo OK indtil videre)
- Jone har sat 50 min af til at hjælpe med data
- Jone har eksplicit sagt: *"FCN er fra 2015. Vi har brugt YOLOv11n eller ConvNeXtTiny, som er bedre modeller."*

### Med Videometer
- Bekræftet:
  - Billeder er HIPS-format, præprocesseret, reflektans-enheder
  - mm/pixel skaleringsfaktor følger med
  - Privat repo OK indtil efterår, hvor det skal til Videometers GitHub
  - 14 GB delt: blobs + fulde billeder + Excel
  - Excel indeholder: Blob ID, Y Center mass, X Center mass, Source Image ID
  - "Mange løsninger" — brugeren bestemmer selv tilgang

---

## 10. Til Claude Code: Sådan tænker den tidligere Claude

Disse principper har styret beslutningerne hidtil — overvej at fortsætte i samme ånd:

1. **Pragmatisme over perfektion.** Plan A altid. Plan B som fallback. Drop "nice-to-have" der ikke flytter projektet fremad.

2. **Forklaringsbarhed først.** Hver linje kode skal kunne forklares til eksamen. Hvis kode er smart men uforklarligt, omskriv det enklere.

3. **Bygg på det brugeren allerede har lært.** Kursus-materiale > smarte tricks. Brugeren har set Sequential, Conv2D, Adam, MSE, train_test_split — brug det vokabular.

4. **Vær ærlig om risici og kompromiser.** Ikke alt skal være "fantastisk" — sig hvis noget er "fint", "OK med forbehold", eller "potentielt et problem".

5. **Skub tilbage når noget ikke giver mening.** Brugeren spurgte om at bygge heatmap+PSNR — Claude argumenterede for at droppe dem. Det viste sig at være rigtigt.

6. **Dansk er primært, engelsk OK i kode-kommentarer.** Brugeren skifter naturligt mellem sprogene.
