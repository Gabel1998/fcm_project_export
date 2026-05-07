# Projekt-tidslinje

## Beslutninger truffet undervejs

### 5. maj 2026
- **Initial vurdering:** FCN Landmark Detection som beskrevet i Videometer's case-dokument
- **Rute valgt:** Først syntetiske data, så rigtige Videometer-data når de kommer
- **Bekræftet:** Jone har givet grønt lys for emnet
- **Bekræftet:** Videometer er aktiv samarbejdspartner (case fra dem)

- **Pivot 1:** Jone skubbede tilbage på FCN — vi skifter til YOLOv11
  - Begrundelse: FCN er fra 2015, kursus bruger YOLOv11n / ConvNeXtTiny
  - Vi havde allerede bygget FCN-pipeline (F1=0.96 på syntetisk data)
  - FCN-arbejdet bevares som arkiv i scripts/01_initial_fcn_attempt/

- **Beslutning:** Ikke heatmaps + PSNR
  - Begrundelse: De er FCN-specifikke, ikke YOLO-naturlige
  - Vores landmark-metrics i mm er bedre/mere informative

- **Bekræftet datasæt-format fra Videometer:**
  - 14 GB total
  - HIPS-format (kalibreret reflektans, 0-1)
  - mm/pixel skaleringsfaktor i hver header
  - Excel-fil med ground truth (Blob ID, X/Y center mass, Source Image ID)
  - Aflange frø (ris/hvede-lignende), tætpakkede uden overlap
  - Belyst top + bund (ingen skygger)

- **Pivot 2:** Lokalt projekt på Mac M-chip i stedet for Colab
  - Begrundelse: NDA-overvejelser, 14GB upload, stabil arbejdsgang
  - Udfordring: Videometer-toolbox kræver Windows (men Excel kan læses uden)
  - Workaround: Bed Videometer/Jone om PNG/TIFF-eksport

## YOLOv11 træningsresultater (syntetiske data)

```
DATASAET:
  300 billeder paa 256x256 pixels
  17.580 frø totalt (~58 pr. billede)
  Skala: 0.05 mm/pixel
  Train/val: 240/60

YOLO BBOX METRICS (Jons krav i parentes):
  Precision:  0.9932    (krav: > 0.5)  
  Recall:     0.9992    (krav: > 0.3)  
  mAP@50:     0.9947    (krav: > 0.2)  
  mAP@50-95:  0.7162    (bonus)

TRAINING:
  Epochs trained: 39 (early stopping at 29)
  Time on Colab T4: 0.034 hours (~2 min)
  Model: YOLOv11n (2.58M params, 5.4MB)
```

## Vigtige links og ressourcer

- **Kursusplan:** Machine Learning Elective spring 2026
- **YOLO-tutorial:** Train_Yolov11_model_on_custom_dataset.pdf
- **FCN-dokument (case fra Videometer):** FCM Landmark detection.docx (vi følger ikke dets metode-anbefalinger)
- **Jons DNN Guide:** Jons Deep Neural Network Guide.pdf

## Næste milepæle

- [ ] Skaffe PNG/TIFF-versioner af billederne
- [ ] Lokalt setup på Mac (brug setup_guide.md)
- [ ] Migrér og test pipeline lokalt
- [ ] Bygge Excel/Videometer data-loader
- [ ] Træne på rigtige data
- [ ] Forberede slides til præsentation
- [ ] Forberede DNN-grundspørgsmål
- [ ] (Extra) ONNX-eksport
- [ ] (Extra) Java Spring webserver

## Eksamensdato
**Uge 24 (11.–16. juni 2026)** — ca. 5-6 uger fra nu
