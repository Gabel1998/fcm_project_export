# Colab-træning

Træning på Colab T4 GPU er ~10× hurtigere end lokal MPS, og er især nyttig
når man eksperimenterer med data-mix eller hyperparametre.

**Vigtigt:** Kun **syntetiske data** her. Rigtige Videometer-data må aldrig
op til Colab pga. NDA — det er præcis derfor vores syntetiske generator
findes.

## Layout

Hver mappe matcher 1:1 en mappe under `models/`. Når Jone eller Videometer
spørger "hvordan blev `v2_robust` trænet?", er notebooken det fulde svar.

```
scripts/04_colab/
├── README.md
├── v2_robust/
│   └── train.ipynb     →   models/v2_robust/best.pt
└── v3_augmented/
    └── train.ipynb     →   models/v3_augmented/best.pt
```

| Version | Hvad er anderledes |
|---|---|
| `v2_robust` | Mixed conditions (støj, frø-størrelse, baggrund). Standard augmentations under træning. |
| `v3_augmented` | v2 + plade-imperfektioner (dust, ridser, gradient) + aggressive train-augmentations (mixup, copy-paste, perspective, shear). |

**Note:** `v1_kalibreret` blev trænet **lokalt** på Mac MPS via
`scripts/02_yolo_pipeline/03_train_yolo.py` — derfor ingen v1-mappe her.

## Sådan kører du en notebook

1. Åbn [colab.research.google.com](https://colab.research.google.com)
2. **File → Open notebook → Upload** → vælg fx `v3_augmented/train.ipynb`
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

1. Kopier den nyeste mappe: `cp -r v3_augmented v4_<beskrivelse>`
2. Ændr `name='v3_augmented'` i træningscellen til fx `name='v4_<beskrivelse>'`
3. Justér de eksperimenter du vil afprøve (augmentations, hyperparametre, data-mix)
4. Kør på Colab, download, læg under `models/v4_<beskrivelse>/`
5. Tilføj en `notes.md` under `models/v4_<beskrivelse>/` der beskriver hvad der er anderledes

## Versions-match

Alle notebooks pinner `ultralytics==8.4.47` så `best.pt` er kompatibel med
din lokale installation. Hvis du opgraderer Ultralytics lokalt, opdater
også versionsnummeret i notebookens første kode-celle.
