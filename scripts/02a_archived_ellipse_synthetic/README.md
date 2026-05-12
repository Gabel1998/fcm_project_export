# Arkiveret: Ellipse-baseret syntetisk pipeline

Denne mappe indeholder den oprindelige syntetiske data-pipeline (v0)
som tegnede aflange ellipser på ensartet grå baggrund. Vi har erstattet
den med `scripts/02_yolo_pipeline/composite_generator/` som bygger
composites fra rigtige Videometer-frø-PNG'er på faste blå plade-baggrunde.

## Hvorfor bevaret?

- **Eksamens-fortælling:** progression fra "ellipser når der ikke var
  rigtige data" → "composites når frø-PNG'er kom" → "rigtige Videometer-
  scener efter besøg på fabrik". Den narrative bro er værd at vise.
- **Baseline-sammenligning:** v1/v2/v3-modellerne er trænet på ellipse-
  data. Hvis composite-trænede modeller klarer sig dårligere på en
  metric, kan vi diagnosticere ved at sammenligne mod denne pipeline.

## Filer

- `01b_realistic_synthetic_data.py` — generér 300 256×256 syntetiske
  billeder med aflange ellipser på ensartet grå baggrund (reflektans 0.85)
- `02_convert_to_yolo_format.py` — konvertér output til YOLO-format dataset

## Ikke aktivt brugt

Scripts er ikke testet efter flyt og er ikke en del af nuværende
pipeline. Hvis du genaktiverer dem, vær opmærksom på at output-stier
kan kollidere med composite_generator-output.

## Hvad blev erstattet med hvad

| Gammel (ellipse) | Ny (composite) |
|---|---|
| Tegnede ellipser via matplotlib | Rigtige alpha-stansede majs-PNG'er fra Videometer |
| Grå baggrund (reflektans=0.85, konstant) | Blå pyramide-tekstureret plade (Belt1/Belt2) |
| 300 billeder, 256×256 | 2000 composites, 4096×2704 |
| Fast bbox 18×18 pr. frø | Bbox beregnet fra hver stamps faktiske dimensioner |
| Ingen pakkings-strategier | scatter / dense / t_config blandet |
| Ingen skygger/feather | Syntetisk dropshadow + edge-feather |

Se `docs/specs/2026-05-11-composite-pipeline-design.md` for design-detaljer.
