# Offline guide — selvkørende pipeline

Hvad du kan gøre uden internet (alle data er lokale + YOLO11n weights er allerede downloaded).

## TL;DR

```bash
make help          # se alle kommandoer
make status        # hvad findes / hvad kører lige nu

# Generér og træn på en variant:
make gen-config CONFIG=t_heavy DATASET=v2_t_heavy
make train DATASET=v2_t_heavy NAME=v5_t_heavy
make watch NAME=v5_t_heavy       # i et andet terminal-vindue mens den træner
make eval NAME=v5_t_heavy         # når den er færdig
```

## Workflows

### 1. Generér flere datasæt (~30-40 min hver)

Du har 4 forskellige konfigurations-varianter klar:

| Variant | Hvad det er | Hvornår |
|---|---|---|
| (default `config.yaml`) | 40% scatter / 30% dense / 30% t_config | Balanceret baseline |
| `t_heavy` | 30% / 20% / 50% — mange T/L-clusters | Hvis modellen fejler på touching |
| `dense_only` | 0% / 100% / 0% — ekstrem tæthed | Stress-test |
| `sparse` | 100% / 0% / 0% — kun løse scener | Sanity-baseline / kontrolforsøg |

```bash
make gen                                       # default config → data/composite_v1/
make gen-config CONFIG=t_heavy DATASET=v2_t_heavy
make gen-config CONFIG=dense_only DATASET=v3_dense_only
make gen-config CONFIG=sparse DATASET=v4_sparse
```

Hver kørsel producerer `data/<NAME>/images/{train,val}/`, `labels/{train,val}/`, `data.yaml`, `generation_metadata.json`, `preview.png`.

**Vil du have en helt ny variant?** Kopiér en config:

```bash
cp scripts/02_yolo_pipeline/composite_generator/configs/t_heavy.yaml \
   scripts/02_yolo_pipeline/composite_generator/configs/min_variant.yaml
# Rediger min_variant.yaml manuelt
make gen-config CONFIG=min_variant DATASET=v5_eksperiment
```

### 2. Træn YOLO på et dataset (~1-3 timer på Mac MPS)

```bash
make train DATASET=v2_t_heavy NAME=v5_t_heavy
```

**Default parametre:**
- `EPOCHS=80` — antal trænings-epochs
- `IMGSZ=1280` — YOLO input-størrelse (kompromis: detalje vs. hastighed)
- `BATCH=4` — batch-størrelse (op til 8 hvis MPS har RAM nok)

Override:

```bash
make train DATASET=v2 NAME=v5 EPOCHS=40 IMGSZ=640 BATCH=8   # hurtigere
```

Mac-tips:
- Hvis MPS løber tør for RAM: reducér `BATCH` eller `IMGSZ`
- Hvis du har masser af tid: `IMGSZ=1920` giver bedre præcision på små frø

### 3. Se om træningen går godt — `make watch`

I et separat terminal-vindue mens træning kører:

```bash
make watch NAME=v5_t_heavy
```

Du ser et live-tabel der opdateres hvert 5. sekund:

```
=== Live træning: v5_t_heavy ===
21:34:52

Sidste 8 epochs (af results.csv):
epoch   train/box_loss  metrics/precision  metrics/recall  metrics/mAP50  ...
12      0.0432          0.91               0.88            0.93
13      0.0398          0.93               0.90            0.95
...

Peak mAP@50 indtil videre: 0.951
```

**Hvad du leder efter:**
- ✅ `train/box_loss` falder over tid → modellen lærer
- ✅ `metrics/mAP50` stiger → bedre detection
- ⚠️ Hvis `train/box_loss` plateauer tidligt og mAP stagnerer → modellen er stuck. Prøv større imgsz eller flere epochs
- ❌ Hvis loss bliver `NaN` → noget er skidt. Stop og tjek input-data (preview.png)

**Eller åbn plot-versionen:**

```bash
open runs/train/v5_t_heavy/results.png    # live plots, opdateres pr. epoch
```

### 4. Evaluering bagefter

```bash
make eval NAME=v5_t_heavy
```

Printer Precision / Recall / mAP@50 / mAP@50-95 på val-set:

```json
{
  "precision": 0.946,
  "recall": 0.972,
  "mAP50": 0.985,
  "mAP50-95": 0.724
}
```

**Acceptkriterier fra Jone (kursus-min-krav):**
- Precision > 0.5
- Recall > 0.3
- mAP@50 > 0.2

Vores models v1/v2/v3 ramte alle ~0.99 på syntetisk. Forventer composites ramler omkring 0.85-0.95 (sværere data).

### 5. Visual sanity-check

```bash
make predict NAME=v5_t_heavy
# eller direkte:
.venv/bin/python -c "from ultralytics import YOLO; \
  YOLO('runs/train/v5_t_heavy/weights/best.pt').predict('data/composite_v2_t_heavy/images/val/composite_00000.png', save=True, imgsz=1280, device='mps')"
```

Resultat: `runs/detect/predict/composite_00000.png` med predikteret bbox'er.

## Anbefalet rækkefølge over 2 dage offline

**Dag 1 morgen:** Træn v4 på default (composite_v1 — kører lige nu)
```bash
make train                          # default dataset + navn
```

**Dag 1 eftermiddag:** Generér 2 varianter parallelt
```bash
make gen-config CONFIG=t_heavy DATASET=v2_t_heavy &
make gen-config CONFIG=dense_only DATASET=v3_dense_only &
wait
```

**Dag 1 aften:** Træn på t_heavy
```bash
make train DATASET=v2_t_heavy NAME=v5_t_heavy
```

**Dag 2 morgen:** Sammenlign + vælg den bedste
```bash
make status                         # se hvad du har
make eval NAME=v4_composite
make eval NAME=v5_t_heavy
```

**Dag 2 eftermiddag:** Træn på dense_only eller sparse for at få en sammenligning
```bash
make train DATASET=v3_dense_only NAME=v6_dense_only
```

## Hvis noget går galt

| Problem | Fix |
|---|---|
| `Mangler data/.../data.yaml` | Kør `make gen` først for det dataset |
| MPS error / out of memory | `make train BATCH=2 IMGSZ=640` |
| Træning ekstremt langsom | Tjek `make status` — kører flere processer? |
| Preview ser galt ud | Tjek `preview.png` for det dataset; juster config og kør gen igen |
| `pretrained yolo11n.pt mangler` | Filen ligger i `scripts/02_yolo_pipeline/yolo11n.pt` — flyt til projekt-root eller juster `--pretrained` arg |

## Filer der ligger lokalt (ingen internet kræves)

- `~/Development/dataset/Corn/images/` — 12.384 alpha-stansede majs-PNG'er
- `~/Development/dataset/Corn/Belt{1,2}.png` — plade-baggrunde
- `scripts/02_yolo_pipeline/yolo11n.pt` — pretrained YOLOv11n weights
- `.venv/` — alle Python-dependencies
