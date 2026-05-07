# Modeller

Versionerede YOLOv11n-modeller til FCM Landmark Detection.
Hver model bor i sin egen mappe så vi kan A/B-sammenligne uden at miste tidligere træninger.

## Layout

```
models/
├── README.md                        # denne fil
├── active -> <version>              # symlink til den model demo'en bruger
├── v1_kalibreret/
│   ├── best.pt                      # 5.4 MB
│   └── notes.md                     # hvad denne version blev trænet på
└── v2_robust/                       # kommer når Colab-notebooken er kørt
    ├── best.pt
    └── notes.md
```

## Skift aktiv model

Brug helper-scriptet i projektroden:

```bash
./switch_model.sh v2_robust
```

Eller manuelt med symlink:

```bash
cd models
ln -sfn v2_robust active
```

Streamlit-appen finder altid `models/active/best.pt` først — caching opdateres ved app-restart.

## Tilføj ny version

1. Opret mappen: `mkdir models/v3_<beskrivelse>`
2. Læg `best.pt` derind (fx fra Colab-download eller lokal træning)
3. Skriv en kort `notes.md` med trænings-config og resultater
4. (Valgfri) skift til den: `./switch_model.sh v3_<beskrivelse>`

## Versions-historik

| Version | Trænet på | P / R / F1 | Note |
|---|---|---|---|
| v1_kalibreret | 300 syntetiske billeder, kun `noise_level=0.02` | 0.99 / 1.00 / 0.99 | Brittel uden for trænings-fordeling |
| v2_robust | 600 syntetiske billeder, mix af `kalibreret`/`realistisk`/`svært` + variabel frø-størrelse | TBD | Trænes via `scripts/04_colab/train_robust.ipynb` |
