# Parameter-guide: hvad du kan ændre og hvad effekten er

Komplet reference over alle parametre i pipelinen, hvor de bor, hvad de gør,
og typiske værdier at prøve.

## Indhold

1. [Data-generering (composite-pipeline)](#1-data-generering)
2. [Træning (YOLO)](#2-træning)
3. [Model-arkivering og versionering](#3-model-arkivering)
4. [Kode-niveau parametre (sjældent ændret)](#4-kode-niveau-parametre)

---

## 1. Data-generering

Konfigureres i `scripts/02_yolo_pipeline/composite_generator/config.yaml`
(eller en variant under `configs/`).

### Dataset-skala

| Parameter | Default | Hvad det gør | Hvornår ændre |
|---|---|---|---|
| `num_composites` | `2000` | Antal composite-scener at generere | Op for mere træningsdata (langsommere). Ned til 100-500 for hurtig iteration |
| `train_val_split` | `0.80` | 80% til train, 20% til val | Sjældent ændret. 0.90 hvis du har for lidt val-data |
| `random_seed` | `42` | Reproducerbarhed | Skift hvis du vil have anderledes data med samme config |

### Composite-størrelse

| Parameter | Default | Hvad det gør | Hvornår ændre |
|---|---|---|---|
| `canvas_size` | `[4096, 2704]` | Output-billede-størrelse (width, height) | Behold matcher Belt1's blå zone. Halvér til `[2048, 1352]` for hurtigere træning |

### Frø pr. scene

| Parameter | Default | Hvad det gør | Hvornår ændre |
|---|---|---|---|
| `seeds_per_composite.min` | `40` | Minimum frø pr. scene | Op for tættere scener |
| `seeds_per_composite.max` | `80` | Maximum frø pr. scene | Op til 120 for stress-test |

### Pakkings-strategi (den vigtigste!)

| Parameter | Default | Hvad det gør | Hvornår ændre |
|---|---|---|---|
| `packing_strategy_mix.scatter` | `0.40` | 40% af frø placeres tilfældigt med god margin | Op for sparsommere scener |
| `packing_strategy_mix.dense` | `0.30` | 30% af frø placeres tæt (gap=0) | Op for tættere klynger |
| `packing_strategy_mix.t_config` | `0.30` | 30% af frø placeres i T/L-konstellationer (touching) | Op til 0.50 for mere fokus på Videometers udfordring |
| `packing_gap_pixels` | `4` | Minimum px mellem scattered frø-kanter | Op for mere luft, ned for tættere |
| `max_placement_attempts` | `200` | Hvor mange forsøg pr. frø før vi giver op | Op hvis pipeline returnerer for få frø |

**De 3 weights skal summe til 1.0** (ellers normaliserer compute() dem).

### Frø-transformation

| Parameter | Default | Hvad det gør | Hvornår ændre |
|---|---|---|---|
| `seed_rotation_range` | `[0, 360]` | Hver frø-stamp roteres random i dette range | Sjældent ændret |
| `seed_scale_jitter` | `[1.0, 1.0]` | Skalering — pt. droppet (inkompatibel med packing) | Lad være med at ændre uden at læse spec 3.6 |

### Rendering

| Parameter | Default | Hvad det gør | Hvornår ændre |
|---|---|---|---|
| `shadow.enabled` | `true` | Tilføj dropshadow under hvert frø | Sæt `false` for at simulere top-belysning uden skygge |
| `shadow.offset_px` | `[3, 3]` | Skygge-retning (x, y) | Ned til `[1, 1]` for mindre dramatisk skygge |
| `shadow.blur_sigma` | `4.0` | Skygge-blødhed (gaussian blur) | Op til 6-8 for soft shadow |
| `shadow.opacity` | `0.35` | Hvor mørk skyggen er (0-1) | Ned til 0.2 for subtil, op til 0.5 for hård |
| `edge_feather_px` | `1.5` | Blødhed af alpha-kanten | Op til 3.0 hvis du ser hårde kanter, ned til 0 for skarpe |

### Baggrund-augmentation

| Parameter | Default | Hvad det gør | Hvornår ændre |
|---|---|---|---|
| `background_random_rotation` | `true` | Random 90/180/270° rotation af baggrund | Sæt `false` hvis det forvirrer modellen |
| `background_color_jitter` | `0.0` | Farve-jitter — droppet pr. Videometer-constraint | Lad det stå |
| `background_brightness_jitter` | `0.0` | Lys-jitter — droppet pr. Videometer-constraint | Lad det stå |

### Output

| Parameter | Default | Hvad det gør | Hvornår ændre |
|---|---|---|---|
| `image_format` | `png` | Output-format | `jpg` for ~5× mindre disk men kvalitetstab |
| `bbox_strategy` | `tight` | Bbox passer hver frøs faktiske dimensioner | Sjældent ændret |
| `visualize_n_samples` | `8` | Antal samples i preview.png | Op til 16 for større oversigt |

---

## 2. Træning

CLI-args til `scripts/02_yolo_pipeline/03b_train_composite.py` (eller via `make train EPOCHS=... IMGSZ=...`).

### Vigtige træningsparametre

| Parameter | Default | Hvad det gør | Hvornår ændre |
|---|---|---|---|
| `--epochs` | `80` | Antal trænings-runder | Ned til 40 for hurtig iteration. Op til 150 for mere finjustering |
| `--imgsz` | `1280` | YOLO ned-skalerer composites til denne størrelse | 640 = hurtigt, 1920 = mere detalje, 2560 = max kvalitet |
| `--batch` | `4` | Antal billeder pr. weight-opdatering | Op til 8/16 hvis MPS har RAM. Ned til 2 hvis MPS swapper |
| `--patience` | `15` | Stop tidligt hvis ingen forbedring i N epochs | Op til 25 hvis early-stop er for aggressiv |
| `--device` | `mps` | Hardware: mps (Mac), cuda, cpu | `cpu` hvis MPS crasher |

### Trænings-tid (estimater på Mac MPS)

| imgsz | batch | Tid pr. epoch | 80 epochs total |
|---|---|---|---|
| 640 | 8 | ~1 min | ~80 min |
| 1280 | 4 | ~5 min | ~6-7 timer |
| 1920 | 2 | ~10 min | ~13 timer |

**Tip:** Start med `imgsz=640` for hurtig sanity-check. Hvis det virker, kør samme dataset igen med `imgsz=1280` for produktion.

### YOLO augmentations (i `03b_train_composite.py` direkte)

Disse er hard-coded i scriptet. Ændr ved at editere `model.train(...)`-kaldet:

| Parameter | Default | Effekt |
|---|---|---|
| `flipud` | `0.5` | Vertikalt flip 50% af tiden |
| `fliplr` | `0.5` | Horisontalt flip 50% af tiden |
| `degrees` | `180.0` | Random rotation ±180° |
| `mosaic` | `1.0` | 4-billede mosaic-augmentation 100% af tiden |
| `mixup` | `0.15` | Bland 2 billeder med 15% sandsynlighed |
| `copy_paste` | `0.3` | Klip-og-indsæt objekter 30% af tiden |

Sæt en til 0 for at slukke. YOLO docs: <https://docs.ultralytics.com/usage/cfg/#augmentation-settings>

---

## 3. Model-arkivering

Når en træning er færdig og du vil bevare resultatet:

```bash
make save-model NAME=v5_t_heavy MODEL_DIR=v5_t_heavy
```

Det laver:
- `models/v5_t_heavy/best.pt` — modellens vægte
- `models/v5_t_heavy/results.png` — trænings-kurver
- `models/v5_t_heavy/results.csv` — rå metrics pr. epoch
- `models/v5_t_heavy/notes.md` — skabelon du udfylder

**Udfyld `notes.md`** med:
- Hvilket dataset modellen er trænet på
- Endelige metrics (kør `make eval` for at få dem)
- Observationer om hvad gik godt/skidt

**Skift aktiv model** (det demo-appen bruger):

```bash
make switch-model MODEL_DIR=v5_t_heavy
# Eller direkte:
./switch_model.sh v5_t_heavy
```

**Sammenlign flere modeller:**

```bash
make status                          # se alle versionerede modeller
make eval NAME=v4_composite          # eval seneste træning
make eval NAME=v5_t_heavy            # eval anden træning
```

### Versionerings-konvention

Følg eksisterende mønster:
- `v1_kalibreret` — første rene træning
- `v2_robust` — med robusthed-augmentations
- `v3_augmented` — med yderligere augmentations
- `v4_composite` — første på rigtige Videometer-data (composite_v1)
- `v5_t_heavy` — composite med 50% T-clusters
- `v6_dense_only` — composite med 100% dense

Holder en logisk historie til eksamen om hvordan modellen udvikler sig.

---

## 4. Kode-niveau parametre

Disse ændrer du direkte i koden — ikke via config.

### `scripts/02_yolo_pipeline/composite_generator/packing.py`

| Sted | Variabel | Effekt |
|---|---|---|
| `_bounding_radius()` | Half-diagonal-formula | Ændrer hvor konservativt vi tjekker overlap |
| `t_config()` linje 162 | `+90.0 + rng.uniform(-15, 15)` | Random vinkel-jitter på frø 2's perpendicular rotation. Mere jitter = mere variation i L-form |

### `scripts/02_yolo_pipeline/composite_generator/background_loader.py`

| Sted | Variabel | Effekt |
|---|---|---|
| Linje 13-16 | `BLUE_HUE_MIN/MAX/SAT/VAL` | HSV-tærskler for at detektere blå zone. Ændr hvis nyt Belt-billede med anden blå-tone |
| `load()` morphology kernel `(21, 21)` | Closing kernel-størrelse | Op for at fylde større huller i tekstur, ned for mere præcis maske |

### `scripts/02_yolo_pipeline/03b_train_composite.py`

| Sted | Variabel | Effekt |
|---|---|---|
| `parse_args()` defaults | Standard CLI-værdier | Ændr default `--epochs`, `--imgsz` etc her |
| `model.train(...)` kald | Alle YOLO-args | Hele YOLO's træning-config — se Ultralytics docs |

---

## Eksperimenter at prøve

Ideer til at se hvad der virker:

1. **Sammenlign t_config-vægtning:**
   ```bash
   make gen DATASET=v_balanced                     # 30% t_config (default)
   make gen-config CONFIG=t_heavy DATASET=v_heavy  # 50% t_config
   # Træn begge, eval, sammenlign mAP
   ```

2. **Test sværere edge-cases via tæthed:**
   ```bash
   make gen-config CONFIG=dense_only DATASET=v_dense   # alle frø tætpakkede
   make train DATASET=v_dense NAME=v_dense_model
   make eval NAME=v_dense_model
   ```

3. **Hurtigere træning ved lavere opløsning:**
   ```bash
   make train DATASET=composite_v1 NAME=v_fast IMGSZ=640 BATCH=16 EPOCHS=40
   # 8x hurtigere end default — god til at iterere over konfigurations-ideer
   ```

4. **Test om skygge hjælper:**
   - Kopiér `config.yaml` til `configs/no_shadow.yaml`, sæt `shadow.enabled: false`
   - Generér + træn + sammenlign
