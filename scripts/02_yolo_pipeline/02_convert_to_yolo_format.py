"""
=====================================================================
 Step 2 (YOLO version): Konvertér syntetiske data til YOLO-format
=====================================================================

 Forudsætning:
   Du har kørt step 1B og har filerne:
     X_realistic.npy
     centers_realistic.npz
     metadata_realistic.json

 Hvad scriptet laver:
   1. Loader vores syntetiske billeder og frø-centre
   2. Konverterer hvert frø-center til en bounding box (lille box rundt om centrum)
   3. Splitter i train/val sæt (80/20)
   4. Gemmer i YOLO's mappestruktur:
        dataset/images/train/   <- .png billeder
        dataset/images/val/
        dataset/labels/train/   <- .txt filer med bbox-koordinater
        dataset/labels/val/
        data.yaml               <- YOLO konfiguration
   5. Visualiserer et eksempel så vi kan tjekke at boxes sidder korrekt

 YOLO label-format (én linje pr. fro):
   class_id x_center y_center width height
   (alle værdier normaliseret til 0-1)

 Hvorfor bounding boxes når vi gerne vil have center-koordinater?
   - YOLOv11 er optimeret til bounding box detection
   - Center af en bbox = vores landmark (det er hvad vi vil have)
   - Ved evaluering konverterer vi simpelthen: center = (x_min + x_max)/2, ...
"""

# =====================================================================
# IMPORTS
# =====================================================================
import os
import shutil       # til at slette gamle datasæt-mapper
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches  # til at tegne bbox-rektangler
from PIL import Image  # til at gemme billeder som PNG


# =====================================================================
# KONFIGURATION
# =====================================================================
DATASET_DIR = "dataset"      # rod-mappe for det YOLO-formaterede datasæt
TRAIN_SPLIT = 0.8            # 80% til træning, 20% til validering
RANDOM_SEED = 42

# Bounding box størrelse omkring hvert fro-center (i pixels)
# Vi laver en lille box - YOLO vil så lære at finde præcist det center
BBOX_SIZE_PIXELS = 18        # box på 18x18 pixels rundt om hvert fro-center
                             # (vores fro er ~12-18 pixels lange, så det dækker)

np.random.seed(RANDOM_SEED)


# =====================================================================
# STEP A: LOAD DATA FRA STEP 1B
# =====================================================================
print("Loader syntetiske data...")
X = np.load("X_realistic.npy")
centers_data = np.load("centers_realistic.npz")
with open("metadata_realistic.json") as f:
    metadata = json.load(f)

NUM_IMAGES = X.shape[0]
IMG_SIZE = X.shape[1]

print(f"  {NUM_IMAGES} billeder på {IMG_SIZE}x{IMG_SIZE}")
print(f"  {metadata['mm_per_pixel']} mm/pixel")


# =====================================================================
# STEP B: OPRET YOLO-MAPPESTRUKTUR
# =====================================================================
# Slet eventuelt gammelt dataset for at undgå forvirring
if os.path.exists(DATASET_DIR):
    shutil.rmtree(DATASET_DIR)

# Opret de fire nødvendige under-mapper
for split in ["train", "val"]:
    os.makedirs(f"{DATASET_DIR}/images/{split}", exist_ok=True)
    os.makedirs(f"{DATASET_DIR}/labels/{split}", exist_ok=True)

print(f"Mappestruktur klar i ./{DATASET_DIR}/")


# =====================================================================
# STEP C: SPLIT INDEKS I TRAIN/VAL
# =====================================================================
# Bland alle billed-indeks så train og val har samme fordeling
indices = np.arange(NUM_IMAGES)
np.random.shuffle(indices)

split_point = int(NUM_IMAGES * TRAIN_SPLIT)
train_indices = indices[:split_point]
val_indices = indices[split_point:]

print(f"\nTræning: {len(train_indices)} billeder")
print(f"Validering: {len(val_indices)} billeder")


# =====================================================================
# HJÆLPEFUNKTION: konvertér frø-center til YOLO bbox-format
# =====================================================================
def center_to_yolo_bbox(center_x, center_y, bbox_size, img_size):
    """
    Konverterer (center_x, center_y) center-koordinat til YOLO-format bbox.

    YOLO format: class x_center y_center width height
                 (alle normaliseret til 0-1)

    YOLO forventer center-baseret bbox - perfekt for os!
    Centrum af bboxen = vores frø-centrum direkte.
    """
    # Normaliser til [0, 1]
    x_norm = center_x / img_size
    y_norm = center_y / img_size
    w_norm = bbox_size / img_size
    h_norm = bbox_size / img_size

    # Sikre at bboxen ikke går ud over billedet (clamp)
    # YOLO accepterer bboxes der er centreret ved kanten,
    # men vi sikrer for en sikkerheds skyld
    x_norm = np.clip(x_norm, 0.0, 1.0)
    y_norm = np.clip(y_norm, 0.0, 1.0)

    return x_norm, y_norm, w_norm, h_norm


# =====================================================================
# STEP D: GEM HVERT BILLEDE + LABEL-FIL
# =====================================================================
def save_split(indices, split_name):
    """Gemmer alle billeder og labels for ét split (train eller val)."""
    print(f"\nGemmer {split_name}-sæt...")
    total_seeds = 0

    for image_index in indices:
        # Hent billede og fjern kanal-dimension (bare for PIL)
        img = X[image_index, :, :, 0]

        # Konvertér float [0,1] til uint8 [0,255] så det kan gemmes som PNG
        img_uint8 = (img * 255).astype(np.uint8)

        # Gem billedet som PNG
        img_filename = f"{DATASET_DIR}/images/{split_name}/img_{image_index:04d}.png"
        Image.fromarray(img_uint8).save(img_filename)

        # Hent fro-centre for dette billede
        centers = centers_data[f"img_{image_index:04d}"]

        # Lav label-fil i YOLO-format
        # Én linje pr. frø: "class_id x_center y_center width height"
        label_filename = f"{DATASET_DIR}/labels/{split_name}/img_{image_index:04d}.txt"
        with open(label_filename, "w") as f:
            for (center_x, center_y) in centers:
                x, y, w, h = center_to_yolo_bbox(center_x, center_y, BBOX_SIZE_PIXELS, IMG_SIZE)
                # class_id 0 = "seed" (vi har kun én klasse)
                f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")

        total_seeds += len(centers)

    print(f"  {len(indices)} billeder + {total_seeds} frø gemt")


save_split(train_indices, "train")
save_split(val_indices, "val")


# =====================================================================
# STEP E: LAV data.yaml KONFIGURATIONSFIL
# =====================================================================
# YOLOv11 har brug for en yaml-fil der fortæller den hvor data ligger
# og hvilke klasser der er.
yaml_content = f"""# YOLOv11 datas\u00e6t-konfiguration for fr\u00f8-detektion
# Genereret fra syntetiske data (Videometer-kompatibel)

# Stier til train/val billeder (relativ til denne yaml-fil)
train: images/train
val: images/val

# Antal klasser
nc: 1

# Klassenavne
names:
  0: seed

# Metadata
mm_per_pixel: {metadata['mm_per_pixel']}
"""

yaml_path = f"{DATASET_DIR}/data.yaml"
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"\nYAML-konfiguration gemt: {yaml_path}")


# =====================================================================
# STEP F: VISUALISER ET PAR EKSEMPLER MED BBOX-OVERLAY
# =====================================================================
# Sanity check at vores bboxes sidder korrekt på frøene
print("\nLaver visualisering af bboxes...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Vælg 3 tilfældige train-billeder
sample_indices = np.random.choice(train_indices, size=3, replace=False)

for col, image_index in enumerate(sample_indices):
    img = X[image_index, :, :, 0]
    centers = centers_data[f"img_{image_index:04d}"]

    # Vis billedet
    axes[col].imshow(img, cmap='gray', vmin=0, vmax=1)

    # Tegn bbox rundt om hvert frø
    for (center_x, center_y) in centers:
        # Beregn bbox-corners (top-left for matplotlib Rectangle)
        x = center_x - BBOX_SIZE_PIXELS / 2
        y = center_y - BBOX_SIZE_PIXELS / 2

        rect = patches.Rectangle(
            (x, y), BBOX_SIZE_PIXELS, BBOX_SIZE_PIXELS,
            linewidth=1.0, edgecolor='lime', facecolor='none'
        )
        axes[col].add_patch(rect)

        # Lille prik i centrum
        axes[col].plot(center_x, center_y, 'r.', markersize=3)

    axes[col].set_title(f"Billede {image_index}: {len(centers)} frø med bboxes")
    axes[col].axis('off')

plt.suptitle(f"YOLO-format datasæt - bboxes på {BBOX_SIZE_PIXELS}x{BBOX_SIZE_PIXELS} pixels rundt om hvert fro",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("yolo_format_examples.png", dpi=100, bbox_inches='tight')
plt.show()

print("Visualisering gemt: yolo_format_examples.png")


# =====================================================================
# OPSUMMERING
# =====================================================================
print("\n" + "=" * 60)
print("YOLO DATASÆT KLAR!")
print("=" * 60)
print(f"\nMappestruktur:")
print(f"  {DATASET_DIR}/")
print(f"    images/train/   ({len(train_indices)} billeder)")
print(f"    images/val/     ({len(val_indices)} billeder)")
print(f"    labels/train/   ({len(train_indices)} .txt-filer)")
print(f"    labels/val/     ({len(val_indices)} .txt-filer)")
print(f"    data.yaml       (konfiguration)")
print(f"\nNæste skridt:")
print(f"  Step 3: pip install ultralytics og træn YOLOv11n")
