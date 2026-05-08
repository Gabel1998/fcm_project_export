# Design: Streamlit-demo til FCM Landmark Detection

**Dato:** 2026-05-07
**Eksamen:** Uge 24 (juni 2026) — denne demo er en ekstern fremvisning til Videometer dagen efter (2026-05-08).
**Status:** Klar til implementation efter brugerens godkendelse.

---

## 1. Formål

Bygge et Streamlit-baseret UI-lag oven på den eksisterende, trænede YOLOv11n-model, så Videometer-folk kan se modellen i aktion på en polished måde.

Demoen bruges som en showcase med interaktiv backup: brugeren (Andreas) styrer fortællingen, men gæsterne kan også selv trykke på parametre hvis de vil.

## 2. Mål og ikke-mål

**Mål:**
- Vise side-by-side: ground truth (genereret) vs YOLOv11n-prediction.
- Lade gæsten justere fire parametre der genererer et nyt billede + kører prediction live.
- Forklare alle tekniske termer via små info-tooltips (ⓘ-ikoner) på dansk.
- Se professionelt ud uden at have en designer involveret — Streamlit-default + lidt CSS.

**Ikke-mål:**
- Ingen træning under demoen. Modellen er allerede trænet (`best.pt`).
- Ingen upload af eget billede.
- Ingen brugerlogins, persistens, deployment.
- Ingen rigtig Videometer-data — kun syntetisk (rigtig data er ikke ankommet endnu).
- Ingen mobile-tilpasning. Køres på en Mac under præsentationen.

## 3. Brugerrejse

1. Andreas starter app: `streamlit run scripts/03_demo/app.py`.
2. Default-skærm vises: et default-genereret billede (58 frø, 256×256, kalibreret støj, normal frø-størrelse) med ground truth til venstre og prediction til højre, samt metrics nederst.
3. Andreas (eller gæst) ændrer en parameter i sidebar.
4. Streamlit re-renderer: ny generering → ny prediction → opdaterede billeder + metrics.
5. Gæster kan klikke ⓘ ved enhver term for en kort dansk forklaring.

Forventet brug pr. interaktion: ~1-2 sekunder fra parameter-ændring til opdateret skærm (genereing er hurtig, MPS-prediction er ~200 ms pr. billede).

## 4. UI-struktur

```
┌─ Sidebar (240 px) ──────────┬─ Main ──────────────────────────────┐
│ ⚙️ Generér nyt billede       │ FCM Landmark Detection              │
│                              │ (tagline)                           │
│ Antal frø ⓘ                  │                                     │
│ [─ slider ─] 58              │ ┌─ Ground Truth ┐ ┌─ Prediction ┐  │
│                              │ │ billede + grøn │ │ billede + rød│  │
│ Billed-størrelse ⓘ           │ │ centre         │ │ bbox + centre│  │
│ [256 × 256 ▾]                │ └────────────────┘ └──────────────┘  │
│                              │                                     │
│ Støj-niveau ⓘ                │ ┌─ Metrics-bar ──────────────────┐  │
│ [Kalibreret][Real.][Svært]   │ │  P     R     F1    Median fejl │  │
│                              │ │ 0.99  1.00  0.99   0.028 mm    │  │
│ Frø-størrelse ⓘ              │ └────────────────────────────────┘  │
│ [Mindre][Normal][Større]     │                                     │
│                              │ ℹ️ Hvad ser du? (forklarings-blok)   │
│ [⟳ Generér + predict]        │                                     │
│                              │                                     │
│ ─────                        │                                     │
│ Model: YOLOv11n              │                                     │
│ Trænet: 48 epochs, MPS       │                                     │
│ Skala: 0.05 mm/pixel         │                                     │
└──────────────────────────────┴─────────────────────────────────────┘
```

## 5. Parametre

| Parameter | Type | Værdier | Default | Streamlit-widget |
|---|---|---|---|---|
| Antal frø | int | 20–80 | 58 | `st.slider` |
| Billed-størrelse | int | 128, 256 | 256 | `st.selectbox` |
| Støj-niveau | preset | kalibreret / realistisk / svært | kalibreret | `st.radio` (horizontal) |
| Frø-størrelse | preset | mindre / normal / større | normal | `st.radio` (horizontal) |

**Mapping fra preset til generator-værdier:**

```python
NOISE_PRESETS = {
    "kalibreret": 0.02,    # som model blev trænet på
    "realistisk": 0.05,
    "svært":      0.10,
}
SEED_SIZE_PRESETS = {
    "mindre":  (length=(8, 12),  width=(3, 5)),
    "normal":  (length=(12, 18), width=(5, 7)),  # som model blev trænet på
    "større":  (length=(18, 24), width=(7, 9)),
}
```

## 6. Datafl ow

```
[parameter-ændring]
        ↓
generate_image(num_seeds, img_size, noise, seed_size_preset)
        ↓ returnerer: (img_uint8, ground_truth_centers, mm_per_pixel)
predict(img_uint8) via cached YOLO-model
        ↓ returnerer: (pred_centers, pred_confidences, inference_ms)
match_landmarks(gt_centers, pred_centers, tolerance=5px)
        ↓ returnerer: (P, R, F1, distances)
render_side_by_side(img, gt_centers, pred_centers)
        ↓ to PIL-billeder med overlays
st.image(...) + st.metric(...)
```

**Caching:**
- `@st.cache_resource` på model-loaderen — ladet kun én gang.
- Ingen caching på `generate_image` (vi vil have ny tilfældig generering hver gang).

## 7. Forklaringstekster (tooltips)

Alle ⓘ-ikoner åbner en kort dansk forklaring (max 2 sætninger). Tekster samles i `info_texts.py` som ét dict, så de er nemme at redigere.

| Term | Forklaring |
|---|---|
| Antal frø | Hvor mange frø generatoren forsøger at placere på pladen. Tætpakkede billeder er sværere at detektere. |
| Billed-størrelse | Modellen er trænet på 256×256. Mindre billeder gør prediction lettere men giver færre frø. |
| Støj-niveau | Tilfældige reflektans-variationer i pladen. Kalibreret = som model blev trænet på. |
| Frø-størrelse | Modellen er trænet på "normal" størrelse (12-18 px lange). Mindre/større frø udfordrer modellen. |
| Ground Truth | Den kendte sandhed fra generatoren — hvor frøene faktisk er placeret. Vores facit. |
| YOLOv11n Prediction | Modellens output: bounding boxes omkring hvert frø. Centrum af hver box = predicted landmark. |
| Precision | Andelen af modellens forudsigelser der rammer rigtigt. 1.0 = ingen falske detektioner. |
| Recall | Andelen af rigtige frø som modellen faktisk fandt. 1.0 = ingen missede frø. |
| F1 Score | Harmonisk gennemsnit af Precision og Recall — én score der balancerer begge. |
| Median fejl | Typisk afstand fra predicted center til rigtigt center. Rapporteret i mm via Videometers skala. |

## 8. Filstruktur

Ny mappe: `scripts/03_demo/`

```
scripts/03_demo/
├── app.py             # Streamlit-app — entry point
├── synthetic.py       # generate_image(num, size, noise, seed_size_preset)
│                      # → refaktoreret fra 01b_realistic_synthetic_data.py
├── landmarks.py            # match_landmarks(gt, pred, tolerance) + metric-beregning
│                      # → refaktoreret fra 04_predict_and_evaluate_landmarks.py
├── info_texts.py      # dict med alle danske tooltip-tekster
└── styles.css         # ~50 linjer custom CSS for at hæve over Streamlit-default
```

**Eksisterende scripts ændres ikke.** Demo-koden er en ny mappe ved siden af de andre.

**Model-sti:** Findes via `glob.glob("runs/**/best.pt", recursive=True)` og tager nyeste — samme fallback som vi tilføjede til script 4.

## 9. Dependencies

Tilføjes til `requirements.txt`:
```
streamlit>=1.30.0
```

Alt andet (numpy, PIL, ultralytics, torch) er allerede installeret.

## 10. Test-strategi

Manuel test, ingen automatiserede tests:
1. **Smoke test:** `streamlit run scripts/03_demo/app.py` — appen åbner uden fejl.
2. **Default-rendering:** Default-parametre giver et billede med ~58 frø og høj precision/recall.
3. **Parameter-respons:** Hver af de fire parametre udløser ny generering + prediction.
4. **Edge cases:** 20 frø (sparse), 80 frø (tæt), 128×128 (lille billede), "svært" støj — alle skal stadig rendere uden crash.
5. **Tooltip-test:** Klik hver ⓘ — forklaring skal komme på dansk og være forståelig.
6. **MPS-fallback:** Hvis MPS ikke er tilgængelig (fx på en anden maskine), skal app falde tilbage til CPU uden crash.

Demo-killer-scenario at undgå: model-loader fejler ved appstart — håndteres med tydelig fejlbesked + tjekliste ("har du kørt step 3 først?").

## 11. Risici og åbne spørgsmål

**Risiko: NMS er langsom på MPS.** Vi så 100+ ms NMS pr. billede under træning. Hvis prediction tager 2-3 sekunder pr. parameter-ændring, føles demoen sløv. **Mitigering:** Vis en `st.spinner` så det er åbenlyst at noget arbejdes på.

**Risiko: 128×128 + få frø giver tomme detektioner.** YOLOv11n kan ramme 0 detektioner. **Mitigering:** Håndter `pred_centers == []` pænt — vis "0 detektioner" i stedet for at crashe på division.

**Risiko: Frø-størrelse "større" får frø til at rage ud over billed-kanten.** Eksisterende margin-logik i generatoren håndterer det allerede, men færre frø passer.

**Risiko: Custom CSS er Streamlit-version-følsom.** Hvis CSS-selectors ikke virker, falder vi tilbage til Streamlit-default — stadig præsentabelt.

**Åbent spørgsmål:** Skal "Generér + predict"-knappen være eksplicit, eller automatisk på hver parameter-ændring? **Beslutning:** Automatisk på hver ændring (Streamlit's default-pattern). Knappen er kun en explicit "ny tilfældig" hvor parametre er uændrede.

## 12. Tidsestimat

- Refaktorering af synthetic.py + landmarks.py: 60 min
- Streamlit app.py (layout, parametre, billed-rendering): 90 min
- Tooltips + info_texts.py: 30 min
- Custom CSS: 30 min
- Manuel test + finpudsning: 30 min

**Total: ~4 timer**, realistisk på en dag.

## 13. Hvad sker der efter demoen

Demo-koden bliver i `scripts/03_demo/` og kan bruges igen til:
- Eksamenspræsentationen i juni (samme demo, opdateret med rigtige Videometer-data hvis de er ankommet).
- Test af modellen på rigtige data — bare tilføj en "upload billede"-tilstand senere.

Intet i designet låser vejen for det.
