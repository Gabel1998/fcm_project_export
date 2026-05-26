# FCM Landmark Detection

**Eksamensaflevering — KEA, Maskinlæring, maj 2026**
**Forfatter:** Andreas Søgård Gabel

---

## 1. Problemstilling

I samarbejde med Videometer A/S er der undersøgt om frø-positioner på en standardiseret optagelsesplade kan detekteres automatisk via maskinlæring. Den nuværende arbejdsgang kræver manuel annotering af hvert enkelt frøs centermasse, hvilket både er tidskrævende og skalerer dårligt ved større datamængder.

Målet med projektet har været at træne en model der kan udpege frø-positioner i et billede med tilstrækkelig præcision til at erstatte den manuelle annotering, eller som minimum fungere som pre-annotering der reducerer det manuelle arbejde markant.

Som detektions-arkitektur er der valgt **YOLOv11n** fra Ultralytics-rammeværket. Valget skyldes en kombination af modellens lette form (~5 MB trænede vægte), gode anchor-fri performance på små objekter og kompatibilitet med Apple Silicon (MPS), som har været det primære udviklingsmiljø.

## 2. Datagrundlag — hvorfor syntetisk data

Adgangen til ægte Videometer-data har været begrænset gennem hele projektet: optagelserne er underlagt en NDA, mængden er lille, og manuel annotering af frø-centre er præcis det problem der ønskes løst. At basere træningen direkte på de få tilgængelige rigtige billeder ville derfor være både datafattigt og cirkulært.

I stedet er der bygget en **syntetisk data-pipeline** der procedurelt sammensætter realistiske composite-billeder: enkeltfrø-udsnit fra Videometer-optagelser placeres på et baggrundsbillede af den blå optagelses-plade, og labels genereres automatisk ud fra placeringen. Pipelinen har genereret **7.000 composite-billeder i 1280×1280** opløsning, og giver fuld kontrol over hvor mange frø der ligger i hvert billede, deres positioner og deres ikke-overlappende layout.

![Eksempel på syntetisk composite-billede](docs/screenshots/synthetic_example.png)
*Figur 1: Et af de 7.000 syntetiske composite-billeder brugt til træning. Frø er placeret tilfældigt på en blå plade-baggrund uden overlap.*

## 3. Tidlige tilgange og opdaget data leakage

Modellen er blevet trænet i flere iterationer hvor metodikken er strammet op undervejs:

| Version | Datasæt-tilgang | Bemærkning |
|---|---|---|
| v1_kalibreret | Kun rene frø-udsnit, ingen baggrund | Baseline — orienteringsstudie |
| v2_robust | Composite med tilfældig augmentation | Test af robusthed |
| v3_augmented | Composite med stærkere augmentation | Validerede augmenterings-strategien |
| v4_composite | Større composite-datasæt, fælles baggrunde train/val | Mistænkeligt høje metrikker |
| **v5_no_leak** | **Composite med adskilte baggrunde train/val** | **Endelig model** |

Ved gennemgang af v4-resultaterne er der opdaget en metodisk fejl: **de baggrundsbilleder der blev brugt til at konstruere composite-eksemplerne, var fælles mellem træningssæt og valideringssæt**. Hver baggrund optrådte under flere frø-konstellationer, men hvis den samme baggrund optrådte både i train og val, kunne modellen lære features ved baggrunden selv frem for ved frøene. Det er klassisk data leakage og forklarer hvorfor v4-metrikker var urealistisk gode.

Læringen har formet hele den endelige pipeline: **enhver baggrund må kun optræde i ét split**. Den re-arkitekterede composite-generator splitter først baggrunds-poolen i train/val/test, og bygger derefter composite-billeder inden for hvert split. Det er denne disciplin der gør v5's tal troværdige.
