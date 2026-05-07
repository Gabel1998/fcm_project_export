# Forventede eksamensspørgsmål + svar-skabeloner

## Tekniske spørgsmål om dit projekt

### "Hvorfor YOLOv11 og ikke FCN som dokumentet beskriver?"
*Du anbefalede selv YOLOv11 fordi FCN er fra 2015 og I underviser i moderne arkitekturer. YOLOv11 har en moderne convolutional backbone, dedikerede detection heads, og er state-of-the-art til object detection. FCN var mere klassisk dense-prediction.*

### "Hvorfor ikke ConvNeXt?"
*ConvNeXt er primært en classification backbone — den siger "hvad er på dette billede" med ét svar. Min opgave kræver detection af mange frø samtidigt. YOLOv11 har en moderne conv-backbone der ligner ConvNeXt's design plus dedikerede detection heads.*

### "Hvorfor syntetiske data først?"
*Validering. Jeg ville være sikker på at min pipeline virkede på data hvor jeg kendte ground truth, før jeg brugte tid på rigtige Videometer-data. Plus det gav mig en sikkerhedsline hvis dataene blev forsinkede. Det er sådan ML-engineers arbejder professionelt med industri-kunder.*

### "Hvad er forskellen på bbox detection og landmark detection?"
*Bbox detection finder rektangler omkring objekter. Landmark detection finder punkter (x,y koordinater). Jeg bruger YOLO's bbox detection og konverterer center-of-bbox til landmark. Det er en lille bro der gør YOLO til en landmark detector.*

### "Hvad er mAP@50?"
*Mean Average Precision ved IoU≥50%. IoU = Intersection over Union — hvor meget overlapper predicted bbox med ground truth bbox. Hvis ≥50% overlap → korrekt detektion. mAP er gennemsnittet af precision på tværs af forskellige confidence-niveauer.*

### "Hvorfor pre-trænet model?"
*YOLOv11n er allerede trænet på COCO (millioner billeder, 80 klasser). Den ved hvordan man finder objekter generelt. Jeg fine-tuner den til kun at finde "frø". Det er hurtigere og bedre end at træne fra bunden på begrænsede data.*

### "Forklar din loss-funktion"
*YOLO bruger en sammensat loss af tre komponenter: CIoU loss for bbox-regression (hvor nøjagtigt rammer vi objektet?), BCE loss for klassifikation (er det et frø?), og DFL (Distribution Focal Loss) for fine-grained bbox-positioning. Jeg har ikke selv skrevet loss — jeg bruger Ultralytics' standard.*

### "Hvad er early stopping?"
*Træningen stopper automatisk hvis validation loss ikke forbedres i N epochs (jeg brugte patience=10). Det forhindrer overfitting og sparer tid. Min model stoppede ved epoch 29 og fik gemt best.pt fra det punkt.*

### "Hvorfor confidence threshold 0.25?"
*YOLO returnerer alle predictions med en confidence-score (0-1). Lav threshold = flere detektioner men flere falske. Høj threshold = færre falske men risiko for at misse rigtige. 0.25 er Ultralytics' default og giver god balance.*

### "Hvorfor evaluerer du både YOLO-metrics og landmark-metrics?"
*YOLO's metrics (mAP, IoU) er for bounding boxes — hvor god er overlappet. Min opgave er landmark detection — hvor præcist finder jeg centre. Jeg laver en separat evaluering der specifikt måler center-præcision i pixels og mm.*

---

## DNN-grundspørgsmål (kommer altid)

### "Forklar forward propagation"
*Inputtet ganges med vægte, summes, går gennem aktiveringsfunktion. Det resultat går til næste lag. Sådan rejser data fra input til output gennem netværket.*

### "Forklar backpropagation"
*Efter forward pass beregnes loss (hvor langt fra rigtigt svar). Gradienten af loss ift. hver vægt beregnes med kædereglen, lag for lag baglæns. Optimizeren bruger så gradienten til at opdatere vægtene.*

### "Hvad er gradient descent?"
*En algoritme der minimerer loss ved at gå i den retning hvor gradienten er negativ. Skridtstørrelsen kaldes learning rate.*

### "Hvad er learning rate?"
*Hvor stort et skridt vi tager når vi opdaterer vægte. For stor → vi springer over minimum. For lille → vi konvergerer aldrig. Adam tilpasser learning rate automatisk.*

### "Hvad er momentum?"
*En slags "memory" af tidligere gradienter. Hjælper med at smutte over flade områder og dæmpe oscillationer. Standard i Adam.*

### "Hvad er ReLU?"
*Rectified Linear Unit. f(x) = max(0, x). Nuller alle negative værdier ud. Fordel: simpel, ingen vanishing gradient. YOLOv11 bruger SiLU (en blød ReLU-variant).*

### "Hvad er overfitting?"
*Når modellen lærer træningsdata for godt og ikke generaliserer til ny data. Tegn: train loss falder mens val loss stiger. Mit projekt bruger early stopping og augmentation for at undgå det.*

### "Hvad er train/test split?"
*Vi deler data: model træner kun på train-sættet, og evalueres på val/test som den aldrig har set. Det viser om modellen kan generalisere. Jeg brugte 80/20 split.*

---

## Spørgsmål om datasæt

### "Hvor mange billeder havde du?"
*På syntetiske data: 300 billeder, 17.580 frø totalt, ~58 frø per billede. Train/val: 240/60. På rigtige Videometer-data: [vil afhænge af hvad de leverer].*

### "Hvordan annoterede du data?"
*Syntetiske data: jeg genererede både billeder OG ground truth samtidigt — så annotationer var perfekte. Rigtige data: følger med fra Videometer i et Excel-dokument med center-koordinater pr. frø.*

### "Hvordan håndterer du data augmentation?"
*YOLO laver det automatisk under træning: flip op/ned, flip venstre/højre, og rotation 0-180°. Frø kan ligge i alle retninger så rotation giver mening. Det forhindrer overfitting.*

---

## Spørgsmål om resultater

### "Tallene ser meget gode ud — er det troværdigt?"
*På syntetiske data, ja. Men det er konsistente data hvor frø-formen er ens og baggrunden er ren. På rigtige Videometer-data forventer jeg lavere tal — typisk er rigtige data sværere. Det er derfor vi tester på syntetisk først, og kun bruger det som baseline.*

### "Hvad er mAP@50-95?"
*Gennemsnit af mAP over IoU-tærskler fra 50% til 95% i 5%-skridt. Strengere end mAP@50. Min mAP@50-95 er 0.72 — ikke perfekt, men det betyder primært at bbox-præcisionen ikke altid er pixel-perfekt. For landmark-detection er det OK fordi vi bruger center-of-bbox.*

### "Hvad er din landmark-præcision i mm?"
*[Indsæt tal fra evaluation]. Det er muligt fordi Videometer leverer mm/pixel skaleringsfaktor i hver billed-header.*

---

## Spørgsmål om implementation

### "Hvad er dine vigtigste hyperparametre?"
*Image size 256, batch size 16, 50 epochs (med early stopping), learning rate styres af AdamW (auto). Augmentation: flip 50%, rotation 180°.*

### "Hvor lang tid tog træningen?"
*På Colab T4 GPU: ~2 minutter for 39 epochs (early stopping). På Mac MPS: lidt længere men stadig under 10 min.*

### "Hvor stor er modellen?"
*best.pt er 5.4 MB. YOLOv11n har 2.58 millioner parametre — langt mindre end fx YOLOv8x med 68 millioner. Trade-off: hastighed vs. kapacitet.*
