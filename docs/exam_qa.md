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
*Træningen stopper automatisk hvis fitness ikke forbedres i N epochs (jeg brugte patience=15). Hos mig blev den aldrig udløst — modellen fandt et nyt peak inden for 15 epochs hele vejen, så alle 80 epochs kørte. best.pt er gemt fra epoch 77 (højeste fitness). Mekanismen forhindrer overfitting hvis modellen holder op med at forbedre sig.*

### "Hvorfor confidence threshold 0.25?"
*YOLO returnerer alle predictions med en confidence-score (0-1). Lav threshold = flere detektioner men flere falske. Høj threshold = færre falske men risiko for at misse rigtige. 0.25 er Ultralytics' default og giver god balance.*

### "Hvorfor evaluerer du både YOLO-metrics og landmark-metrics?"
*YOLO's metrics (mAP, IoU) er for bounding boxes — hvor god er overlappet. Min opgave er landmark detection — hvor præcist finder jeg centre. Jeg laver en separat evaluering der specifikt måler center-præcision i pixels og mm.*

---

## Detektions-kæden i YOLOv11 — fra billede til frø (terp denne)

**Hele flowet i ét diagram:**
```
Billede
 └─ CNN-backbone (convolution + pooling) udtrækker features
     └─ 3 grids:  P3 160×160 (8px celler) · P4 80×80 (16px) · P5 40×40 (32px)
         └─ hver celle = ét fast anchor-punkt  (~33.600 i alt)
             └─ hvert anchor forudsiger SAMTIDIG (to parallelle grene):
                  • cv3 → klasse-score  ("er der et majs?")   ← dine heatmaps
                  • cv2 → 4 kant-afstande via DFL  (boxen)
                     └─ confidence-tærskel:  smid de SVAGE væk
                        └─ NMS:  smid DUBLETTER væk
                           └─ endelige detektioner → 69 frø
```

### "Forklar hele YOLO-pipelinen fra billede til detektion"
*CNN-backbonen udtrækker features og laver tre grids i forskellige opløsninger (P3 fint, P5 groft). Hver grid-celle har ét fast anchor-punkt, og hvert punkt forudsiger samtidig en klasse-score (er det et majs?) og en box (fire kant-afstande via DFL) — i ét forward-pass, deraf "You Only Look Once". Bagefter fjerner confidence-tærsklen de svage detektioner, og NMS fjerner dubletter. Det der overlever er de endelige frø.*

### "Hvad er et anchor-punkt, og hvor kommer det fra?"
*Et fast gitter-punkt, ét pr. grid-celle, lagt ud på forhånd uafhængigt af billedet (~33.600 i alt). Det er udgangspunktet boksens kanter måles fra. Modellen vælger dem ikke — de er der altid; den lærer bare hvilke der skal fyre. Anchor-punktet er IKKE boksens center.*

### "Hvorfor tre grids (P3/P4/P5)?"
*Multi-scale. P3 er fint (160×160, 8px-celler) og fanger små/præcise frø; P5 er groft (40×40, 32px-celler) og fanger store. Det er FPN/PAN-necket. De tre klasse-score-grids er præcis de objectness-heatmaps jeg viser. (160 = antal celler pr. side, ikke pixels.)*

### "Forklar de tre tab (box_loss, cls_loss, dfl_loss)"
*Samlet tab = vægtet sum (7.5·box + 0.5·cls + 1.5·dfl). **cls_loss** (BCE) = "er det et majs?" — let her, kun én klasse. **box_loss** (CIoU) = sidder hele den færdige kasse rigtigt? IoU + center + form. **dfl_loss** = sidder hver af de fire kanter præcist? — arbejder på kant-fordelingerne. dfl gør kanterne skarpe, box_loss sikrer den samlede geometri. cls er helt adskilt.*

### "Hvad er DFL helt konkret?"
*I stedet for at gætte ét tal pr. box-kant fordeler modellen sandsynlighed over 16 bins (mulige afstande, reg_max=16) for hver af de 4 kanter. Den endelige afstand er fordelingens tyngdepunkt (forventningsværdien Σ i·p(i)) — derfor kan den lande mellem hele pixels (sub-pixel-præcision) og vise hvor sikker modellen er. 4 kanter × 16 bins = 64 tal fra cv2-grenen.*

### "Forudsiger modellen center + bredde + højde direkte?"
*Nej. Den forudsiger fire kant-afstande fra anchor-punktet (l/t/r/b via DFL), som afkodes til hjørner (xyxy). Center (cx,cy) regnes til sidst som midten af kassen og bruges som landmark — det forudsiges aldrig direkte.*

### "Hvad er IoU?"
*Intersection over Union — overlap divideret med forening mellem to bokse. 0 = intet overlap, 1 = perfekt. Bruges til at afgøre om en detektion er korrekt (mAP@50 = IoU≥0.5) OG til at finde dubletter i NMS. På mit val-sæt: median IoU 0.97.*

### "Hvad er confidence?"
*Modellens sikkerhed på at en box indeholder et majs (0-1). I YOLOv11 ER det klasse-scoren — der er ingen separat objectness-gren som i v3-v5. Bruges til filtrering (tærskel), til at prioritere i NMS, og til at sortere ved mAP. Siger intet om hvor præcist boksen sidder — det er IoU's job.*

### "Hvad er NMS?"
*Non-Maximum Suppression. Flere anchors laver overlappende bokse om samme frø; NMS sorterer efter confidence, beholder den højeste og fjerner alle med høj IoU til den (dubletter), gentaget til ingen er tilbage. Confidence afgør hvem der vinder, IoU hvem der er dublet. Standard-NMS kan fejle på rørende frø — derfor har jeg en center-afstands-variant i demoen.*

### "Confidence-tærskel vs NMS — forskellen?"
*Confidence-tærskel fjerner de SVAGE bokse (lav score). NMS fjerner DUBLETTER (overlappende bokse om samme frø). To forskellige skridt, i den rækkefølge.*

### "One-stage vs two-stage detektor?"
*To-stage (R-CNN) foreslår først regioner og klassificerer dem bagefter — langsomt. YOLO er one-stage: klasse + box for alle celler i ét pass. Derfor hurtig nok til realtid.*

---

## CNN'et — convolution, pooling, stride, padding (terp denne)

**Hele CNN-flowet (jf. CNN_Explained.png):**
```
Billede
 → [Convolution: filter glider hen over billedet]  → Feature map
 → [Max pooling: krymp, behold stærkeste]           → mindre feature map
 → (gentag conv + pool nogle gange)
 → Flatten → fully-connected → ét klasse-svar       (classification-CNN)
   ELLER: behold de rumlige grids → detection        (YOLO)
```

### "Hvorfor CNN i stedet for et almindeligt (fully-connected) net til billeder?"
*Et fully-connected net ville have en vægt for hver pixel × hver neuron → millioner af parametre, og det ignorerer at nabopixels hænger sammen. CNN deler vægte (ét lille filter glider over hele billedet) → få parametre, fanger lokale mønstre, og er translations-invariant — et frø genkendes uanset hvor det er.*

### "Hvad er convolution / et filter?"
*Et filter er et lille gitter af vægte (fx 3×3) der glider hen over billedet. Ved hver position: gang overlap sammen + læg sammen → ét tal i feature map'et. Filteret detekterer et mønster (kant, tekstur), og vægtene læres under træning. Mit eksempel-filter havde midterkolonnen = 0,33 (≈⅓) → en lodret streg-/gennemsnits-detektor: 3 ettaller i en kolonne giver 3×0,33 ≈ 1.*

### "Hvorfor en aktiveringsfunktion mellem conv-lag?"
*Uden en ikke-linearitet (ReLU/SiLU) ville stablede conv-lag kollapse til én enkelt lineær operation. ReLU = max(0,x) nuller negative værdier ud. YOLOv11 bruger SiLU (en blød ReLU).*

### "Hvad er et feature map, og hvorfor er det mindre end billedet?"
*Feature map'et er filterets svar — det lyser op hvor mønstret er. Det er mindre fordi filteret skal ligge helt inde i billedet, så det taber en kant-stribe: output = input − filter + 1. Et 7-bredt billede med 3×3-filter → 7−3+1 = 5. (Med mindre man bruger padding.)*

### "Hvad er stride?"
*Hvor mange pixels filteret hopper pr. skridt. Stride 1 bevarer opløsningen; stride 2 halverer den (nedskalering). I YOLO giver de akkumulerede strides 8/16/32 mine grids P3/P4/P5 — en P3-celle dækker 8px fordi backbonen har nedskaleret 8×.*

### "Hvad er padding?"
*En kant af 0'er rundt om billedet før convolution. "Valid" = ingen padding → output krymper (7→5). "Same" = pad så output = input (7→7). Padding lader filteret nå helt ud til kanten og bevarer størrelsen.*

### "Output-størrelse-formlen"
*output = ⌊(W − K + 2P) / S⌋ + 1, hvor W=input, K=filter, P=padding, S=stride. Samme formel for både convolution og pooling — kun K/P/S ændrer sig.*

### "Hvad er max pooling?"
*Tag fx 2×2-vinduer og behold kun det største tal, stride 2 → halver opløsningen. Med min lodrette streg: blokken med stregen giver max=1, så stregen overlever i halv opløsning. Modsat average pooling (gennemsnit), der ville udvande 1-tallet til 0,5. Max pooling: reducerer data, beholder de stærkeste features, giver translations-robusthed, forstørrer receptive field.*

### "Convolution-krympning vs pooling-krympning?"
*Convolution krymper kun en lille kant (filter−1, fx 7→5) fordi filteret skal være helt inde i billedet. Pooling halverer bevidst (÷2, fx 6→3). To forskellige grunde til at billedet bliver mindre.*

### "Padding ved pooling vs convolution?"
*Convolution bruger normalt zero-padding (rigtige 0'er) for at bevare størrelsen. Pooling padder normalt IKKE — den dropper det der ikke fylder et helt vindue (floor); padder den, bruger max-pooling −∞/ignorér, ikke 0.*

### "Forskellen på classification-CNN og YOLO (detection)?"
*En classification-CNN ender med Flatten → fully-connected → ét svar ("hvad er på billedet"). YOLO flattener ikke — den beholder de rumlige grids, så den kan sige både hvad OG hvor (mange bokse med position).*

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
*v5 blev trænet på composite_v1_no_leak: 2000 composite-billeder (1600 train / 400 val), bygget af en pulje på 9907 disjunkte train-frø og 2477 val-frø (ingen data leakage). ~24.000 frø-instanser i val-sættet. På rigtige Videometer-data: afhænger af hvad de leverer.*

### "Hvordan annoterede du data?"
*Syntetiske data: jeg genererede både billeder OG ground truth samtidigt — så annotationer var perfekte. Rigtige data: følger med fra Videometer i et Excel-dokument med center-koordinater pr. frø.*

### "Hvordan håndterer du data augmentation?"
*YOLO laver det automatisk under træning: flip op/ned, flip venstre/højre, og rotation 0-180°. Frø kan ligge i alle retninger så rotation giver mening. Det forhindrer overfitting.*

---

## Spørgsmål om resultater

### "Tallene ser meget gode ud — er det troværdigt?"
*På syntetiske data, ja. Men det er konsistente data hvor frø-formen er ens og baggrunden er ren. På rigtige Videometer-data forventer jeg lavere tal — typisk er rigtige data sværere. Det er derfor vi tester på syntetisk først, og kun bruger det som baseline.*

### "Hvad er mAP@50-95?"
*Gennemsnit af mAP over IoU-tærskler fra 50% til 95% i 5%-skridt. Strengere end mAP@50. Min mAP@50-95 er 0.97 (0.9714) på val-sættet — højt, fordi boksene sidder præcist selv ved strenge IoU-tærskler (det matcher min median-IoU på 0.97). Det er dog på syntetiske composites; rigtige Videometer-scener er endnu ikke testet.*

### "Hvad er din landmark-præcision i mm?"
*[Indsæt tal fra evaluation]. Det er muligt fordi Videometer leverer mm/pixel skaleringsfaktor i hver billed-header.*

---

## Spørgsmål om implementation

### "Hvad er dine vigtigste hyperparametre?"
*Image size 1280, batch 4, 80 epochs (patience 15), optimizer=auto (lr0=0.01, momentum=0.937). Augmentation: flip op/ned + venstre/højre (50%), rotation 180°, mosaic, mixup, copy-paste. Imgsz er stor (1280) fordi frøene er små på en stor plade.*

### "Hvor lang tid tog træningen?"
*På Mac M4 Pro (MPS): ~3,6 timer for alle 80 epochs, spredt over 2 sessioner pga. auto-resume (Ultralytics' MPS-bug crashede undervejs, og scripts/auto_resume.sh genstartede fra last.pt). Colab GPU ville være ~10× hurtigere, men rigtige data må ikke uploades (NDA).*

### "Hvor stor er modellen?"
*best.pt er 5.3 MB. YOLOv11n har 2.59 millioner parametre — langt mindre end fx YOLOv8x med 68 millioner. Trade-off: hastighed vs. kapacitet.*

---

## Bredt pensum (Jons liste — uden for projektet)

### "Hvad er et beslutningstræ (decision tree)?"
*Et flowchart af ja/nej-spørgsmål på features. Ved hver knude vælges det split der reducerer urenhed mest — Gini/entropi ved klassifikation, varians/MSE ved regression. Bladet giver svaret (flertal-klasse eller gennemsnit). Let at fortolke, men ét dybt træ overfitter og er ustabilt → man bruger ensembler.*

### "Random Forest vs XGBoost?"
*Random Forest = bagging: mange uafhængige træer på tilfældige data-/feature-udsnit, stemmer midles → reducerer varians. XGBoost = boosting: træer i kæde, hvor hvert nyt træ retter forgængernes fejl via gradienten → reducerer bias. Bagging er parallelt, boosting er sekventielt.*

### "Hvad er XGBoost?"
*eXtreme Gradient Boosting — regulariseret gradient boosting. Mange svage træer bygges sekventielt, hvert trænet på de fejl (residualer) ensemblet stadig laver. "Extreme" = L1/L2-regularisering, 2.-ordens optimering, shrinkage og column subsampling. State-of-the-art på tabulær data — slår ofte deep learning der.*

### "Hvorfor brugte du ikke XGBoost / decision trees?"
*Mit input er billeder, ikke tabulære features. XGBoost er konge på tabeldata (som Videometers Excel-ark med X/Y center mass osv.), men til at lokalisere objekter i et billede bruger man en CNN. Forskellige værktøjer til forskellige data-typer.*

### "Supervised vs unsupervised learning?"
*Supervised = lær med facit/labels (min YOLO med ground-truth-bokse). Unsupervised = find struktur uden labels. Uden labels måles kvalitet anderledes (silhouette score, reconstruction error).*

### "Nævn unsupervised metoder"
*Clustering (k-means — vælg K, tildel til nærmeste centroid, gentag; DBSCAN — tæthedsbaseret, håndterer støj). Dimensionsreduktion (PCA — lineær, max varians; autoencoder — neural komprimering). Anomali-detektion (find outliers).*

### "Hænger unsupervised sammen med dit projekt?"
*Min opgave er supervised, men min efterbehandling `assign_pixels_to_seeds` tildeler hver pixel til nærmeste detektions-center — det er reelt ét skridt af k-means/clustering (marker-styret). Clustering kunne også gruppere frø efter morfologi, og en sten i et billede er en anomali.*

### "Hvad er transfer learning / model transfer?"
*At genbruge en model pre-trænet på et stort datasæt og fine-tune den til sin egen opgave. De tidlige lag har lært generelle features (kanter, teksturer) der passer på enhver billed-opgave; man tilpasser kun de høje lag. Feature extraction = frys backbonen, træn kun head; fine-tuning = træn alt videre med lille learning rate. Gevinst: mindre data, hurtigere, bedre på små datasæt.*

### "Hvordan bruger DU transfer learning?"
*`model = YOLO('yolo11n.pt')` loader COCO-pre-trænede vægte (millioner billeder, 80 klasser), og `model.train(...)` fine-tuner på mine frø-composites. Så jeg starter med generelle billed-features i stedet for fra bunden — derfor klarer jeg mig med relativt få data. Samme princip som GPT/BERT i NLP.*

### "Hvad er GPT / en transformer?"
*GPT = Generative Pre-trained Transformer: en decoder-only, autoregressiv model der forudsiger næste token. Den bygger på self-attention, hvor hvert ord vægter alle andre ord for at danne kontekst — det erstattede RNN/LSTM fordi det kører parallelt og fanger lange afhængigheder. Pre-trænes self-supervised (gæt næste ord) og fine-tunes bagefter — samme transfer-learning-idé som min YOLO, bare på tekst. CNN = billeder (convolution), Transformer = sekvenser (attention).*

### "Hvad er self-attention?"
*Hver token beregner hvor meget den skal "fokusere på" hver anden token (query/key/value) og bygger sin repræsentation som en vægtet sum af de andre. Det giver kontekst — fx at "det" refererer til "dyret" i en sætning.*

### "Hvad er RAG?"
*Retrieval-Augmented Generation. Dokumenter gemmes som embeddings (vektorer) i en vektor-database; et spørgsmål embeddes og de mest semantisk lignende bidder hentes via nærmeste-nabo-søgning og lægges ind i LLM'ens prompt som kontekst. LLM'en svarer forankret i rigtige kilder → opdateret viden, mindre hallucination, bruger private data — uden at gentræne modellen.*

### "Hvordan hænger RAG sammen med resten?"
*En embedding er en lært vektor-repræsentation (som autoencoderens flaskehals). Semantisk søgning = nærmeste-nabo i vektor-rum — samme afstands-princip som k-means og som min pixel-til-nærmeste-center i efterbehandlingen.*

### "Forklar data preparation i dit projekt"
*Trinene: indsamling (syntetisk generator med rigtige frø-stamps på blå plade), labeling (automatisk → perfekte labels), split (train/val UDEN leakage), formatering (YOLO-labels), normalisering (letterbox til 1280) og augmentation (flip, rotation, mosaic, mixup, copy-paste).*

### "Hvad er data leakage, og hvordan undgik du det?" (dit stærkeste data-prep-kort)
*Leakage = når information fra val/test siver ind i træningen. I v4 delte train og val de SAMME frø-stamps, så modellen kunne memorere dem → mAP@50-95 var kun 0.63 og evalueringen misvisende. I v5 gjorde jeg frø-puljerne disjunkte, så val var ægte uset → modellen tvinges til at generalisere → 0.97. Et rent split gav både ærlige metrics OG en bedre model.*

### "Fordele ved syntetisk data?"
*Perfekte labels uden manuel annotering, fuld kontrol over variation, og NDA-sikkert (ingen rigtige data uploades). Jeg validerede pipelinen på syntetisk data hvor jeg kendte facit, før de rigtige Videometer-data kom.*
