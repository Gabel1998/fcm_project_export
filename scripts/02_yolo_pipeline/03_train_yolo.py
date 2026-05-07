"""
=====================================================================
 Step 3 (YOLO version): Træn YOLOv11n på vores syntetiske datasæt
=====================================================================

 Forudsætning:
   Du har kørt step 1B (data) og step 2 (YOLO-format).
   Mappen ./dataset/ med data.yaml skal eksistere.

 Hvad scriptet laver:
   1. Henter en pre-trænet YOLOv11n model (lille version - hurtig at træne)
   2. Træner den på vores frø-datasæt via fine-tuning
   3. Viser metrics: P (precision), R (recall), mAP50

 Hvorfor pre-trænet model og ikke fra bunden?
   YOLOv11n er allerede trænet på COCO-datasættet (80 klasser, millioner billeder).
   Den ved allerede hvordan man finder objekter generelt. Vi finetuner den til
   bare at finde "frø". Det er meget hurtigere og giver bedre resultater end
   at træne fra bunden på kun 240 billeder.

 Jons krav (fra Train Yolov11 model on custom dataset.pdf):
   P > 0.5 for all
   R > 0.3 for all
   mAP50 > 0.2 for all
   Vi sigter selvfølgelig på langt højere :)
"""

# =====================================================================
# IMPORT
# =====================================================================
# Ultralytics er biblioteket der leverer YOLOv11. En enkelt import:
from ultralytics import YOLO

import os


# =====================================================================
# KONFIGURATION
# =====================================================================
# Sti til vores YOLO-format datasæt (lavet i step 2)
DATA_YAML = "dataset/data.yaml"

# YOLOv11n = "nano" version - mindste/hurtigste. Andre: s, m, l, x.
# 'n' er perfekt til Colab og krævende ikke meget GPU-memory.
MODEL_NAME = "yolo11n.pt"   # .pt = pretrained weights (vi finetuner)

# Trænings-parametre
EPOCHS = 50            # antal gange modellen ser hele datasættet
IMG_SIZE_YOLO = 256    # samme som vores billed-størrelse
BATCH_SIZE = 16        # antal billeder pr. opdatering
PATIENCE = 10          # stop tidligt hvis ingen forbedring i 10 epochs (early stopping)

# Hvor resultaterne gemmes
PROJECT_NAME = "runs/seed_detection"
RUN_NAME = "yolov11n_synthetic_v1"


# =====================================================================
# STEP A: VERIFICÉR AT DATASÆTTET EKSISTERER
# =====================================================================
if not os.path.exists(DATA_YAML):
    raise FileNotFoundError(
        f"Kan ikke finde {DATA_YAML}. Kør step 2 først!"
    )

print(f"Datasæt fundet: {DATA_YAML}\n")


# =====================================================================
# STEP B: LOAD PRE-TRAINET YOLOv11n
# =====================================================================
# Første gang denne køres downloader Ultralytics yolo11n.pt automatisk
print(f"Loader pre-trænet model: {MODEL_NAME}")
model = YOLO(MODEL_NAME)
print("Model loadet\n")


# =====================================================================
# STEP C: TRÆN MODELLEN (fine-tuning på vores datasæt)
# =====================================================================
# model.train() gør alt arbejdet:
#   - læser data.yaml
#   - finder train/val billeder
#   - kører træning med YOLO's loss function
#   - gemmer best.pt løbende
#   - producerer plots og metrics i runs/...
print("=" * 60)
print("STARTER TRÆNING")
print("=" * 60)

results = model.train(
    data=DATA_YAML,        # vores data.yaml fil
    epochs=EPOCHS,
    imgsz=IMG_SIZE_YOLO,
    batch=BATCH_SIZE,
    patience=PATIENCE,     # early stopping
    project=PROJECT_NAME,
    name=RUN_NAME,
    exist_ok=True,         # overskriv hvis mappen eksisterer
    verbose=True,
    device="mps",          # Apple Silicon GPU-acceleration (lokalt på Mac)
    # Augmentations (Ultralytics default er gode, men vi tunner lidt):
    flipud=0.5,            # flip op/ned 50% af tiden (vores frø har ingen retning)
    fliplr=0.5,            # flip venstre/højre
    degrees=180.0,         # tilfældig rotation - vores frø kan ligge i alle retninger
    # Bemærk: vi har sort-hvid billeder, men YOLO konverterer dem internt
)

print("\n" + "=" * 60)
print("TRÆNING FÆRDIG")
print("=" * 60)


# =====================================================================
# STEP D: PRINT RESULTATER (det Jone skal se)
# =====================================================================
# results.box er metrics for bounding box detection
# Disse er præcis dem Jone bad om: P, R, mAP50
print("\nFINAL METRICS:")
print(f"  Precision (P):  {results.box.mp:.4f}    (Jons krav: > 0.5)")
print(f"  Recall (R):     {results.box.mr:.4f}    (Jons krav: > 0.3)")
print(f"  mAP@50:         {results.box.map50:.4f}    (Jons krav: > 0.2)")
print(f"  mAP@50-95:      {results.box.map:.4f}    (bonus metric)")

# =====================================================================
# STEP E: VALIDÉR PÅ VAL-SÆT (eksplicit)
# =====================================================================
print("\nValiderer på val-sæt for at bekræfte resultater...")
val_results = model.val(data=DATA_YAML, imgsz=IMG_SIZE_YOLO, verbose=False, device="mps")
print(f"  Val Precision: {val_results.box.mp:.4f}")
print(f"  Val Recall:    {val_results.box.mr:.4f}")
print(f"  Val mAP@50:    {val_results.box.map50:.4f}")


# =====================================================================
# STEP F: HVOR FILERNE LIGGER
# =====================================================================
output_dir = f"{PROJECT_NAME}/{RUN_NAME}"
print(f"\nResultater og artefakter er gemt i: {output_dir}/")
print("  best.pt          - bedste model (brug denne til predictions)")
print("  last.pt          - sidste model (ikke nødvendigvis bedst)")
print("  results.png      - trænings-kurver")
print("  confusion_matrix.png")
print("  val_batch*.jpg   - eksempel-predictions på val-sæt")

print("\n=== Step 3 (YOLO) færdig! ===")
print("Næste skridt: lav predictions og evaluer kvalitativt")
