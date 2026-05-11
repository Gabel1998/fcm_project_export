# Design: Composite-pipeline til majs-detection

**Dato:** 2026-05-11
**Eksamen:** Uge 24 (juni 2026)
**Foranledning:** Demo til Videometer 2026-05-08 afdækkede at det reelle data-setup er enkelt-frø PNG'er + faste plade-baggrunde, ikke færdige scener. Pipelinen skal omdesignes så vi træner på composites bygget af rigtige Videometer-frø-stamps i stedet for tegnede ellipser.
**Status:** Klar til implementation efter brugerens godkendelse af denne spec.

---

## 1. Kontekst og formål

### 1.1 Hvad der ændrer sig

Den nuværende pipeline (`scripts/02_yolo_pipeline/01b_realistic_synthetic_data.py`) genererer 300 syntetiske scener med tegnede aflange ellipser på ensartet grå baggrund. YOLOv11n-modellen v1/v2/v3 er trænet på dette og rammer P/R/F1 ≈ 0.99 på syntetiske data.

Efter Videometer-demoen ved vi at det rigtige setup består af:

- **12.384 enkelt-frø PNG'er** (RGBA, alpha-stansede majs på transparent baggrund) i `~/Development/dataset/Corn/images/`
- **2 faste plade-baggrunde** (`Belt1.png`, `Belt2.png`) — blå pyramide-teksturerede måtter med mørk brun kant
- **`metadata.csv`** med 12.384 rækker, format `Id, Labels, ImagePath` (USDA-kvalitetslabels — ikke relevante for landmark-detection, parkeres til evt. senere klassifikations-projekt)

Den nuværende ellipse-pipeline træner derfor modellen på en falsk verden. Composite-pipelinen er bygget til at producere træningsdata der ligner det rigtige Videometer-setup så meget som muligt, så modellen generaliserer når den ser ægte scener.

### 1.2 Formål

Bygge en data-genererings-pipeline der:

1. Tager enkelt-frø PNG'er + plade-baggrunde + en konfiguration.
2. Producerer composite-scener med 40-80 frø placeret realistisk på pladen, inklusive T- og L-konstellationer (Videometers kerne-udfordring).
3. Skriver YOLO-format dataset (`images/{train,val}/`, `labels/{train,val}/`, `data.yaml`) klar til træning med eksisterende `03_train_yolo.py` uden ændringer.

## 2. Mål og ikke-mål

**Mål:**

- Producere et træningsdataset (default 2000 composites, 80/20 train/val split) der visuelt og distributionelt ligner rigtige Videometer-scener.
- Eksplicit håndtere T/L-konstellationer i pakkings-strategien — ikke håbe at random scatter producerer dem.
- Bevare den eksisterende YOLO-træningskode (`03_train_yolo.py`) og evaluering (`04_predict_and_evaluate_landmarks.py`) uændrede.
- Bevare den gamle ellipse-pipeline som arkiveret baseline til eksamens-fortælling.

**Ikke-mål:**

- Ingen kvalitets-klassifikation (god/skadet majs). CSV'ens USDA-labels parkeres til evt. senere projekt.
- Ingen havre, hvede eller andre frø-typer. Kun majs i denne iteration. Pipelinen behøver ikke være eksplicit "udvidelig" — YAGNI; vi udvider hvis havre kommer.
- Ingen fysisk-baseret rendering (lys-direction-beregning, ambient occlusion). Vi bruger syntetisk dropshadow + edge-feathering, hvilket er "godt nok"-niveau.
- Ingen automatisk perceptual quality metric (SSIM, FID). Kvalitet vurderes via `visualize.py` med øjnene.
- Ingen YOLO-træning som del af denne pipeline. Pipelinen producerer kun data.

## 3. Centrale beslutninger og deres begrundelser

### 3.1 Tilgang B (realistiske composites) — ikke A (minimum viable) eller C (fysisk-baseret)

Tre tilgange overvejet:

- **A — Paste & go:** alpha-stanset frø direkte på baggrund, ingen skygge, hård kant.
- **B — Realistiske composites:** A + syntetisk dropshadow + edge-feathering + eksplicit T/L-pakning.
- **C — Fysisk-baseret:** B + lys-direction-skygger + ambient occlusion + baggrund-aware farve-tilpasning.

**Valgt: B.** Begrundelse: A risikerer "klistret" look hvor modellen lærer alpha-kant-artifakter i stedet for at lære frø. C er for kompleks at forklare til eksamen og marginalvinsten kan ikke verificeres uden ægte test-scener.

### 3.2 Held-out composites alene som validering (ingen rigtig ground truth)

Modellen valideres på 20% holdout af de composites vi selv bygger. **Reel risiko:** modellen kan overfitte til composite-artifakter (skygge-mønster, alpha-blending-stil) og fejle på rigtige Videometer-scener uden vi opdager det kvantitativt.

**Mitigering:** Aggressive train-time augmentations (YOLOs egne `degrees`, `fliplr`, `mosaic`, `mixup`) + den planlagte besøg hos Videometer hvor Andreas selv tager test-billeder. Et lille kvalitativt eval bygges på de billeder bagefter.

**Ærlig formulering til eksamen:** "Mine kvantitative metrics gælder kun composites. Generaliserings-claim er kvalitativ — verificeret ved visuel inspektion af forudsigelser på rigtige scener."

### 3.3 Fuld plade-størrelse composites (~2000×1500), ikke 256×256 eller 640×640

Vi genererer i fuld plade-størrelse for at matche rigtige Videometer-scener direkte. YOLO håndterer scaling internt under træning (`imgsz` parameter). Tradeoff: hver composite er stor på disk (~5-10 MB PNG), så 2000 composites ≈ 10-20 GB. Det er acceptabelt på Mac med ekstern disk.

### 3.4 YOLOv11n nu, A/B-test mod YOLO26n efter v4 er trænet

Composite-pipelinen er model-agnostisk — den producerer YOLO-format labels der virker med både versioner. Vi forfølger YOLO26 som et A/B-test efter v4-modellen er trænet på composites. Hvis YOLO26n giver målbart bedre resultater, bliver det "v5".

### 3.5 Variabel bbox-størrelse pr. frø (`tight`), ikke fast kvadrat

Hvert frø-stamps bbox beregnes fra dets alpha-bounding-box (efter rotation). YOLO lærer dermed at relatere bbox-form til visuel frø-form — bedre signal end den faste 18×18 fra v1-v3.

## 4. Arkitektur og position i projektet

```
┌─────────────────────────────────────────────────────────────────────┐
│ ~/Development/dataset/Corn/                                         │
│ ├── images/ (12.384 alpha-stansede maj-PNG'er)                      │
│ ├── Belt1.png, Belt2.png (plade-baggrunde)                          │
│ └── metadata.csv (USDA-labels — parkes)                             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ scripts/02_yolo_pipeline/composite_generator/                       │
│ ├── seed_loader.py         (læs + preprocess frø-PNG'er)            │
│ ├── background_loader.py   (læs + sample plade-baggrunde)           │
│ ├── packing.py             (scatter / dense / T-config strategier)  │
│ ├── composer.py            (rendering: skygge + paste + GT)         │
│ ├── generate_dataset.py    (driver: N composites → YOLO-dataset)    │
│ ├── visualize.py           (sanity-check: 8 samples + GT-overlay)   │
│ ├── config.yaml            (alle hyper-parametre)                   │
│ └── tests/                 (pytest-tests pr. modul)                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│ data/composite_v1/                                                  │
│ ├── images/{train,val}/composite_*.png                              │
│ ├── labels/{train,val}/composite_*.txt                              │
│ ├── data.yaml                                                       │
│ ├── preview.png                  (auto-genereret efter kørsel)      │
│ └── generation_metadata.json     (reproducerbarhed)                 │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
                    03_train_yolo.py    (uændret — peger på ny data.yaml)
                           │
                           ▼
                    04_predict_and_evaluate_landmarks.py   (uændret)
```

**Arkivering:** `scripts/02_yolo_pipeline/01b_realistic_synthetic_data.py` og dens output-filer flyttes til `scripts/02a_archived_ellipse_synthetic/`. Den eksisterende `02_convert_to_yolo_format.py` bliver overflødig (composite-pipelinen producerer YOLO-format direkte) og kan også arkiveres samme sted.

## 5. Komponenter

### 5.1 `seed_loader.py`

**Ansvar:** Læs og preprocess enkelt-frø PNG'er fra disk.

**Interface:**
- `scan(seed_dir: Path) -> list[Path]` — find alle PNG'er
- `load(path: Path) -> SeedStamp | None` — læs én, returnér None hvis ugyldig
- `SeedStamp` dataclass: `rgba: np.ndarray (H,W,4)`, `source_id: str`, `bbox_after_crop: tuple`

**Logik:**
1. Åbn PNG via PIL. Verificer mode == RGBA, ellers skip + log.
2. Find ikke-transparent bounding box via `alpha > 0`.
3. Crop til den bounding box (fjern transparent margin).
4. Hvis cropped frø er < 30×30 pixels efter crop, log warning men brug det.
5. Returnér `SeedStamp`.

**Caching:** LRU-cache på 500 stamps i RAM. Cache-hit-raten er lav ved random sampling (~4% når pool er 12.384), så cachens primære værdi er at undgå redundant disk-I/O når samme frø tilfældigt vælges to gange tæt på hinanden. Hvis profilering viser at disk-I/O er flaskehalsen, skifter vi til pre-load-all (~2-3 GB RAM for hele poolen).

### 5.2 `background_loader.py`

**Ansvar:** Læs plade-baggrunde og sample crops fra blå zone.

**Interface:**
- `load(paths: list[Path]) -> BackgroundPool` — load + analysér én gang
- `BackgroundPool.sample(size: tuple[int,int]) -> np.ndarray` — random crop fra blå zone

**Logik:**
1. Load begge Belt-PNG'er.
2. Identificér blå zone vs brun kant via HSV-farve-tærskel: pixels hvor `H ∈ [180-220°]` og `S > 0.4` er blå zone.
3. Beregn en boolsk maske pr. baggrund og en liste over gyldige crop-anker-punkter (hvor target-size crop kun rammer blå zone).
4. Ved `sample()`: vælg random baggrund, random crop-anker, evt. random 90/180/270° rotation + flip, returnér crop.

**Hvis fail:** Hvis ingen gyldige crop-anker findes for requested size, hård fail med besked. Vi reducerer size eller giver bedre baggrunde.

### 5.3 `packing.py`

**Ansvar:** Bestem hvor og hvordan frø placeres på en canvas.

**Interface:**
- `compute(canvas_size, seed_stamps, strategy_mix, gap_pixels, max_attempts) -> list[Placement]`
- `Placement` dataclass: `center_x: int`, `center_y: int`, `rotation_deg: float`, `seed_index: int`

**Strategier:**

- **Scatter** (default 50% af frø): random `(x, y, rotation)` med rejection sampling — ingen overlap, respekterer `gap_pixels`. Genbrug af logikken fra ellipse-pipelinen.
- **Dense** (default 30%): som scatter men med `gap_pixels = 0` eller endda lille overlap-tolerance. Producerer tæt pakkede områder.
- **T-config** (default 20%): eksplicit sample 2-3 frø placeret i T- eller L-form. Algoritme: vælg første frø, sample en perpendikulær placering for andet frø der rør første frø, evt. tredje frø oven på krydset. Det er Videometers kerne-udfordring, så vi træner direkte på det.

**Strategi-mix:** En composite indeholder typisk frø fra alle tre strategier blandet. Random sampling af strategi pr. frø-placering, vægtet efter `strategy_mix`.

**Fail-tolerance:** Hvis ingen plads findes efter `max_attempts` forsøg på et frø, accepter færre frø i scenen og fortsæt. Matcher v1-v3-adfærd.

### 5.4 `composer.py`

**Ansvar:** Tag baggrund + frø + placeringer → render ét composite-billede med ground truth.

**Interface:**
- `render(background: np.ndarray, seed_stamps: list[SeedStamp], placements: list[Placement], shadow_config, feather_px, scale_jitter) -> (composite: np.ndarray, centers: list[tuple[int,int]], bbox_sizes: list[tuple[int,int]])`

**Rendering-pipeline pr. frø:**
1. Skalér frø-stamp efter en uniform random faktor i `seed_scale_jitter` (default ±15%).
2. Rotér frø-stamp efter `placement.rotation_deg`.
3. Beregn dropshadow: kopi af stampens alpha-kanal, gaussian-blurret med `shadow.blur_sigma`, forskudt med `shadow.offset_px`, multipliceret med `shadow.opacity`, paste'et som mørkning af baggrunden.
4. Feather alpha-kanten: gaussian-blur af alpha-masken med `edge_feather_px` så hård kant blødes ud.
5. Paste frøets RGB oven på baggrunden med den feathered alpha som blend-vægt.
6. Registrer centrum (placement.center_x, center_y) og beregn akse-aligneret bbox fra den skalerede + roterede stamp's alpha-bounding-box.

**Returnerer:**
- `composite`: (H, W, 3) RGB uint8
- `centers`: liste af (cx, cy) pixel-koordinater
- `bbox_sizes`: liste af (w, h) pr. frø (akse-aligneret efter rotation). YOLO-label kombinerer `centers[i]` + `bbox_sizes[i]` til den normaliserede `cx cy w h`-linje i `generate_dataset.py`.

### 5.5 `generate_dataset.py`

**Ansvar:** Driver-script. Læs config, kald composer N gange, skriv YOLO-dataset.

**Flow:**
1. Parse `config.yaml`.
2. Setup random seed.
3. Init `BackgroundPool` og `seed_pool` (path-liste).
4. For `i in range(num_composites)`:
   a. Tjek om `composite_{i:05d}.png` allerede findes (resume-feature) — hvis ja, skip.
   b. Sample baggrund, vælg `seed_count` frø, kør packing, kør composer.
   c. Bestem split: `train` hvis `i < 0.8 * N`, ellers `val`.
   d. Skriv atomic: composite → `.tmp` → rename. Label-fil samme måde.
   e. Hver 50. composite: skriv `progress.json` med count + tid.
5. Skriv `data.yaml` (YOLO-format config).
6. Skriv `generation_metadata.json` (config-snapshot, random seed, kørsel-tid).
7. Kald `visualize.py` for at producere `preview.png`.

**Label-format (YOLO):** Hver linje: `0 cx_norm cy_norm w_norm h_norm` hvor alt er normaliseret til canvas-størrelse. Class 0 = "seed".

### 5.6 `visualize.py`

**Ansvar:** Sanity-check af genereret dataset.

**Brug:**
- **Auto:** kaldes af `generate_dataset.py` som sidste step, producerer `preview.png` med 8 random samples + GT-markører.
- **Standalone:** `python visualize.py --dataset data/composite_v1 --n 20 --strategy t_config --output custom_preview.png`. Filtrer på strategi (kræver at metadata-felt logges pr. composite).

**Output:** Grid-figur med composite + grønne GT-markører overlaid. Ingen quantitative metrics — ren visuel inspektion.

## 6. Konfiguration (`config.yaml`)

```yaml
# === Paths ===
seed_image_dir: ~/Development/dataset/Corn/images
background_paths:
  - ~/Development/dataset/Corn/Belt1.png
  - ~/Development/dataset/Corn/Belt2.png
output_dir: ~/Development/school/fcm_project_export/data/composite_v1

# === Dataset-skala ===
num_composites: 2000
train_val_split: 0.80
random_seed: 42

# === Composite-størrelse ===
canvas_size: [2000, 1500]   # (width, height)

# === Frø pr. scene ===
seeds_per_composite:
  min: 40
  max: 80

# === Pakkings-strategi ===
packing_strategy_mix:
  scatter: 0.50
  dense: 0.30
  t_config: 0.20
packing_gap_pixels: 4
max_placement_attempts: 200

# === Frø-transformation ===
seed_rotation_range: [0, 360]
seed_scale_jitter: [0.85, 1.15]

# === Rendering ===
shadow:
  enabled: true
  offset_px: [3, 3]
  blur_sigma: 4.0
  opacity: 0.35
edge_feather_px: 1.5

# === Baggrund-augmentation ===
background_random_rotation: true
background_color_jitter: 0.05
background_brightness_jitter: 0.10

# === Output ===
image_format: png
bbox_strategy: tight

# === Validering ===
visualize_n_samples: 8
log_seed_metadata: true
```

## 7. Dataflow (komplet kald)

```
STARTUP (én gang)
├─ Læs config.yaml
├─ BackgroundPool.load([Belt1.png, Belt2.png])
└─ seed_pool = scan(seed_image_dir)  # 12.384 paths

PR. COMPOSITE (gentages num_composites gange)
├─ Tjek om output allerede findes → skip hvis ja (resume)
├─ background = pool.sample(canvas_size)
├─ seed_count = randint(40, 80)
├─ seed_paths = sample(seed_pool, seed_count)
├─ seed_stamps = [seed_loader.load(p) for p in seed_paths]
├─ placements = packing.compute(canvas_size, seed_stamps, ...)
├─ composite, centers, bboxes = composer.render(background, seed_stamps, placements, ...)
├─ split = "train" hvis i < 0.8*N else "val"
└─ atomic write: composite.png + label.txt

AFSLUTNING
├─ Skriv data.yaml
├─ Skriv generation_metadata.json
└─ Kald visualize.py → preview.png
```

## 8. Fejlhåndtering

**Hard fail (stop):**
- Baggrund-fil kan ikke loades → `FileNotFoundError`.
- Output-mappe kan ikke oprettes → `PermissionError`.
- `config.yaml` mangler påkrævede felter → `KeyError` med klar besked.

**Skip + log (fortsæt):**
- Frø-PNG mangler alpha eller er korrupt → log til `skipped_seeds.log`, spring over.
- Frø-PNG har ingen ikke-transparente pixels → skip.
- Packing giver op for et frø → accepter færre frø i scenen.

**Soft tolerance:**
- Frø < 30×30 pixels efter crop → log warning, brug det.
- Baggrund-sample rammer brun zone → re-sample (op til 5 forsøg, så hård fail).

**Resilience:**
- **Atomic write:** `composite_00042.png.tmp` → rename efter både image og label er skrevet.
- **Resume:** `generate_dataset.py` skipper eksisterende output-filer.
- **Progress-log:** `progress.json` hver 50. composite.

## 9. Test-strategi

### 9.1 Unit tests (pytest)

```
tests/
├── test_seed_loader.py     (~5 tests)
├── test_packing.py         (~6 tests)
├── test_composer.py        (~4 tests)
└── test_generate_dataset.py (~3 integration tests)
```

**Strikt TDD for:**
- `packing.py` — rene deterministiske funktioner med klar input/output. Test først, så implementering.
- Label-format-funktioner i `generate_dataset.py` — YOLO-format er regelret.

**TDD-lite for:**
- `seed_loader.py` — fokus på edge cases (corrupt PNG, no alpha) først.

**Visualize-driven for:**
- `composer.py` — rendering-kvalitet kan ikke unittestes på "ser det godt ud?". Basis-invarianter testes (output-størrelse, antal centre, ingen out-of-bounds), kvalitet vurderes via `visualize.py`.

### 9.2 Manuelle preview-gates (to checkpoints)

**Vigtigt:** Vi har to manuelle gates før den fulde 2000-composite kørsel. Implementeringsplanen bygger dem ind eksplicit.

**Gate 1 — efter MVP (sektion 11 trin 1):**
Lille kørsel (N=20) med kun scatter-pakning og uden skygge. Verificerer at grundlaget virker.

- [ ] GT-markører ligger på frø-centre, ikke ved siden af
- [ ] Ingen frø placeret på brun kant (background_loader virker)
- [ ] Frø-rotation virker — der er frø i flere orientering
- [ ] Bbox-størrelser varierer pr. frø (ikke alle ens)
- [ ] Composites ser overordnet rimelige ud (ingen åbenlys bug)

**Gate 2 — efter realisme-features (sektion 11 trin 2):**
Lille kørsel (N=20) med fuld pipeline. Verificerer at realisme-features virker som forventet.

- [ ] Skygger ser naturlige ud (ikke for hårde, ikke usynlige)
- [ ] Ingen synlige hårde alpha-kanter (edge-feather virker)
- [ ] T/L-konfigurationer er reelt til stede (ikke kun random scatter)
- [ ] Dense-pakning giver tæt-pakkede områder
- [ ] Strategi-mix matcher config (ca. 50/30/20)

Hvis et eller flere punkter fejler: justér config eller fix bug, kør igen. Først ved godkendt Gate 2 går vi til den fulde 2000-composite kørsel.

## 10. Begrænsninger og kendte risici

| Risiko | Sandsynlighed | Mitigation |
|---|---|---|
| Modellen overfitter til composite-artifakter | Middel | Aggressive YOLO-augmentations + kvalitativ eval hos Videometer |
| 2000 composites × ~5 MB = 10 GB fylder for meget | Lav | Disk plads tjekkes før kørsel; reducer N hvis nødvendigt |
| Belt1/Belt2 er for ens — modellen memoriserer baggrund | Middel | Aggressive background augmentation (rotation, jitter); ved problem: bed Videometer om flere baggrunde |
| `tight` bbox-strategi giver YOLO problemer med små frø | Lav | Fallback til `square` via config-flag |
| Packing er langsom for tæt-pakkede scener (mange retries) | Lav | `max_placement_attempts` capper; accept færre frø |
| LRU-cache på 500 stamps er for stor til RAM | Lav | Reducer til 100; eller drop cache, læs fra disk hver gang |

## 11. Implementerings-fasering (overordnet)

Detaljeret implementeringsplan laves separat. Højniveau-rækkefølge:

1. **MVP** — `seed_loader.py` + `background_loader.py` + minimal `packing.py` (kun scatter) + minimal `composer.py` (uden skygge) + `generate_dataset.py` der kører N=20. **Gate:** Andreas reviewer preview.
2. **Realisme-features** — dropshadow + edge-feather + dense/T-config-pakning. **Gate:** Andreas reviewer preview igen.
3. **Test-suite** — fuld pytest-coverage pr. modul.
4. **Fuld kørsel** — N=2000, generér produktions-dataset.
5. **Arkivering** — flyt gammel ellipse-pipeline til `scripts/02a_archived_ellipse_synthetic/`.
6. **Træning** — `03_train_yolo.py` peger på det nye `data/composite_v1/data.yaml`, træn v4-model.
7. **A/B-test (senere)** — YOLO26n vs YOLOv11n på samme dataset, byg v5.

## 12. Det vi *ikke* bygger nu

For at undgå scope-creep:

- Ingen ONNX-eksport (extras-mål)
- Ingen Java Spring webserver (extras-mål)
- Ingen multi-frø-type-support (kun majs)
- Ingen kvalitets-klassifikation fra USDA-labels
- Ingen automatisk perceptual quality metric
- Ingen pretrained backbone-fine-tuning ud over Ultralytics' default
- Ingen mosaik-augmentation i pipelinen (YOLO håndterer det)
