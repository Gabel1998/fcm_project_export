# Colab-træning

Træning på Colab T4 GPU er ~10× hurtigere end lokal MPS, og er især nyttig
når man eksperimenterer med data-mix eller hyperparametre.

**Vigtigt:** Kun **syntetiske data** her. Rigtige Videometer-data må aldrig
op til Colab pga. NDA — det er præcis derfor vores syntetiske generator
findes.

## Notebooks

Hver notebook svarer 1:1 til en model under `models/`. Det giver fuld
sporbarhed: når Jone eller Videometer spørger "hvordan blev `v2_robust`
trænet?", er notebooken det fulde svar.

| Notebook | Model | Hvad er anderledes |
|---|---|---|
| `train_v2.ipynb` | `models/v2_robust/` | Mixed conditions (støj, frø-størrelse, baggrund). Standard augmentations under træning. |
| `train_v3.ipynb` | `models/v3_augmented/` | v2 + plade-imperfektioner (dust, ridser, gradient) + aggressive train-augmentations (mixup, copy-paste, perspective, shear). |

**Note:** `v1_kalibreret` blev trænet **lokalt** på Mac MPS via
`scripts/02_yolo_pipeline/03_train_yolo.py` — derfor ingen v1-notebook her.

## Sådan kører du en notebook

1. Åbn [colab.research.google.com](https://colab.research.google.com)
2. **File → Open notebook → Upload** → vælg fx `train_v3.ipynb`
3. **Runtime → Change runtime type → T4 GPU** (gratis)
4. **Runtime → Run all** (Cmd/Ctrl + F9)
5. Vent 5-15 min — sidste celle downloader `best.pt`
6. På din Mac:
   ```bash
   mkdir -p models/v3_augmented      # tilpas til notebookens output-navn
   mv ~/Downloads/best.pt models/v3_augmented/
   fcm-switch v3_augmented
   fcm-demo
   ```

## Tilføj en ny version

1. Kopier den nyeste notebook: `cp train_v3.ipynb train_v4.ipynb`
2. Ændr `name='v3_augmented'` i træningscellen til fx `name='v4_<beskrivelse>'`
3. Ændr de eksperimenter du vil afprøve (augmentations, hyperparametre, data-mix)
4. Kør på Colab, download, læg under `models/v4_<beskrivelse>/`
5. Tilføj en `notes.md` der beskriver hvad der er anderledes fra forrige version

## Versions-match

Alle notebooks pinner `ultralytics==8.4.47` så `best.pt` er kompatibel med
din lokale installation. Hvis du opgraderer Ultralytics lokalt, opdater
også versionsnummeret i notebookens første kode-celle.
