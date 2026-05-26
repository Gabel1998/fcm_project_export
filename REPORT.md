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
