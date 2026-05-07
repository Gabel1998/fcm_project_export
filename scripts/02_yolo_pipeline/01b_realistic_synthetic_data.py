"""
=====================================================================
 FCN Landmark Detection - Step 1B: Realistisk syntetisk data
=====================================================================

 Formål:
   Generere syntetiske data der matcher Videometer's setup:
   - Aflange frø (ris/hvede-lignende) med tilfældig orientering
   - Tætpakket layout (frø rystet ud på plade) men minimalt overlap
   - Reflektans-værdier i [0, 1]
   - Output i samme format som Videometer-data:
       * billede: float32 array (H, W, 1)
       * centre: liste af (x, y) koordinater
       * mm/pixel: skaleringsfaktor

 Hvorfor det her er vigtigt:
   Det her datasæt er det "endelige bro" mellem syntetiske og rigtige data.
   Når vores model klarer dette godt, ved vi at den er klar til Videometer-data.

 Output-filer:
   X_realistic.npy        - billeder
   centers_realistic.npz  - center-koordinater (én liste pr. billede)
   metadata_realistic.json - mm/pixel og andre metadata
"""

# =====================================================================
# IMPORTS
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse  # til at tegne aflange frø
import json


# =====================================================================
# KONFIGURATION
# =====================================================================
IMG_SIZE = 256              # større end før fordi vi nu har mange flere frø
NUM_IMAGES = 300

# Fro-parametre - lavet til at ligne hvede-/ris-frø
SEED_LENGTH_MIN = 12        # frø-længde i pixels (langeste akse)
SEED_LENGTH_MAX = 18
SEED_WIDTH_MIN = 5          # frø-bredde i pixels (korte akse)
SEED_WIDTH_MAX = 7
SEEDS_PER_IMAGE_MIN = 40    # tætpakket: mange frø pr. billede
SEEDS_PER_IMAGE_MAX = 80

# Placering - hvor tæt må de pakkes
PACKING_GAP_PIXELS = 2      # minimum afstand mellem fro-kanter (i pixels)
                            # Lille = tæt pakning. 0 = de rører.
MAX_PLACEMENT_ATTEMPTS = 200  # giv op efter så mange fejlede forsøg pr. frø

# Reflektans-værdier (matcher Videometer's kalibrerede data)
BACKGROUND_REFLECTANCE = 0.85   # plade er lys (høj reflektans)
SEED_REFLECTANCE_MIN = 0.20     # frøne er mørkere
SEED_REFLECTANCE_MAX = 0.45
NOISE_LEVEL = 0.02              # mindre støj end før (Videometer er kalibreret)

# Skala (matcher typisk Videometer-opsætning)
MM_PER_PIXEL = 0.05         # 0.05 mm/pixel - typisk for VideometerLab makro

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


# =====================================================================
# FUNKTION 1: Tjek om to ellipser overlapper
# =====================================================================
def ellipses_overlap(e1, e2, gap):
    """
    Konservativ test: tjekker om to ellipser kommer for tæt på hinanden.

    e1, e2: (cx, cy, length, width, angle) - ellipse-parametre
    gap: minimum tilladt afstand mellem ellipse-kanter

    Vi bruger en simpel "konservativ" approksimation:
    afstand mellem centre < (storste radius af e1 + storste radius af e2 + gap)
    """
    cx1, cy1, L1, W1, _ = e1
    cx2, cy2, L2, W2, _ = e2

    # Den maksimale "radius" af hver ellipse er den halve længde
    r1 = L1 / 2.0
    r2 = L2 / 2.0

    # Euclidean afstand mellem centre
    dx = cx1 - cx2
    dy = cy1 - cy2
    distance = np.sqrt(dx * dx + dy * dy)

    # Hvis afstanden er mindre end summen af radier + gap, er de for tætte
    return distance < (r1 + r2 + gap)


# =====================================================================
# FUNKTION 2: Tegn én ellipse (frø) ind i et billede
# =====================================================================
def draw_ellipse(image, cx, cy, length, width, angle_deg, color):
    """
    Tegner en udfyldt ellipse i billedet ved at modificere pixels direkte.

    Hvordan det virker (matematisk):
    1. For hver pixel (x, y) i billedet:
    2. Translater så centrum er i origo: (x - cx, y - cy)
    3. Rotér med -angle så ellipse-akser er aligned med x og y
    4. Tjek standard ellipse-ligning: (x'/a)^2 + (y'/b)^2 < 1
       - a = length / 2 (halv-akse i x-retning efter rotation)
       - b = width / 2  (halv-akse i y-retning efter rotation)
    """
    a = length / 2.0
    b = width / 2.0
    angle_rad = np.deg2rad(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    # Lav grid af alle pixel-koordinater
    H, W = image.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

    # Translater til ellipse-centrum
    x_shifted = xx - cx
    y_shifted = yy - cy

    # Rotér -angle (så ellipsen "vendes om" til standard orientering)
    x_rotated = cos_a * x_shifted + sin_a * y_shifted
    y_rotated = -sin_a * x_shifted + cos_a * y_shifted

    # Standard ellipse-formel: (x/a)^2 + (y/b)^2 < 1 betyder "inde i ellipsen"
    inside = (x_rotated / a) ** 2 + (y_rotated / b) ** 2 < 1.0

    # Sæt farven, men kun inden for billedets grænser
    image[inside] = color


# =====================================================================
# FUNKTION 3: Generér ét realistisk billede med tætpakkede aflange frø
# =====================================================================
def generate_realistic_image(img_size):
    """
    Genererer ét billede der ligner Videometer's setup:
    aflange frø pakket tæt uden overlap.

    Returns:
      image: (H, W) float32 array i [0, 1]
      centers: liste af (x, y) tupler
    """
    # Start med en lys plade-baggrund (reflektans ~0.85)
    image = np.ones((img_size, img_size), dtype=np.float32) * BACKGROUND_REFLECTANCE

    # Vælg antal frø vi VIL placere (selvfølgelig ikke alle vil passe ind)
    target_seeds = np.random.randint(SEEDS_PER_IMAGE_MIN, SEEDS_PER_IMAGE_MAX + 1)

    placed_ellipses = []  # liste af succesfuldt placerede ellipse-parametre
    centers = []          # bare (cx, cy) til output

    # Margin så frø ikke går ud over kanten
    margin = SEED_LENGTH_MAX // 2 + 2

    # Forsøg at placere hvert frø med rejection sampling
    for seed_idx in range(target_seeds):
        # Vælg tilfældige fro-egenskaber
        length = np.random.randint(SEED_LENGTH_MIN, SEED_LENGTH_MAX + 1)
        width = np.random.randint(SEED_WIDTH_MIN, SEED_WIDTH_MAX + 1)

        # Forsøg at finde en ledig position
        placed = False
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            cx = np.random.randint(margin, img_size - margin)
            cy = np.random.randint(margin, img_size - margin)
            angle = np.random.uniform(0, 360)  # tilfældig orientering

            new_ellipse = (cx, cy, length, width, angle)

            # Tjek om den overlapper med nogen tidligere placerede frø
            collision = False
            for existing in placed_ellipses:
                if ellipses_overlap(new_ellipse, existing, PACKING_GAP_PIXELS):
                    collision = True
                    break

            if not collision:
                # Plads fundet! Tegn frøet og gem positionen
                color = np.random.uniform(SEED_REFLECTANCE_MIN, SEED_REFLECTANCE_MAX)
                draw_ellipse(image, cx, cy, length, width, angle, color)
                placed_ellipses.append(new_ellipse)
                centers.append((cx, cy))
                placed = True
                break

        # Hvis vi ikke kunne placere efter MAX forsøg: pladen er fuld, stop
        if not placed:
            break

    # Tilføj lille reflektans-støj (Videometer's data er kalibreret men ikke perfekt)
    noise = np.random.randn(img_size, img_size) * NOISE_LEVEL
    image = image + noise
    image = np.clip(image, 0.0, 1.0)

    return image, centers


# =====================================================================
# HOVEDLOGIK: Generér hele datasættet
# =====================================================================
print(f"Genererer {NUM_IMAGES} realistiske billeder på {IMG_SIZE}x{IMG_SIZE} pixels")
print(f"Mål: {SEEDS_PER_IMAGE_MIN}-{SEEDS_PER_IMAGE_MAX} aflange frø pr. billede")
print(f"Skala: {MM_PER_PIXEL} mm/pixel\n")

# Forhåndstildel array til billeder
X = np.zeros((NUM_IMAGES, IMG_SIZE, IMG_SIZE), dtype=np.float32)

# Centre kan ikke være et fast-shape array fordi antal frø varierer pr. billede
# Så vi gemmer dem som en liste-af-arrays (gemmes med np.savez senere)
all_centers = []

# Statistik over hvor mange frø vi rent faktisk får placeret
seed_counts = []

for i in range(NUM_IMAGES):
    img, centers = generate_realistic_image(IMG_SIZE)
    X[i] = img
    all_centers.append(np.array(centers, dtype=np.int32))
    seed_counts.append(len(centers))

    if (i + 1) % 50 == 0:
        avg = np.mean(seed_counts)
        print(f"  {i + 1}/{NUM_IMAGES} billeder genereret  (gennemsnit: {avg:.1f} frø pr. billede)")

print(f"\nFærdig! Statistik:")
print(f"  Total frø placeret: {sum(seed_counts):,}")
print(f"  Gennemsnit pr. billede: {np.mean(seed_counts):.1f}")
print(f"  Min: {min(seed_counts)}, Max: {max(seed_counts)}")


# =====================================================================
# TILFØJ KANAL-DIMENSION (så det matcher Keras-format direkte)
# =====================================================================
X = np.expand_dims(X, axis=-1)  # (N, H, W) -> (N, H, W, 1)
print(f"\nX shape: {X.shape}")


# =====================================================================
# GEM DATA - i format der matcher det vi forventer fra Videometer
# =====================================================================
# 1. Billed-arrays
np.save("X_realistic.npy", X)

# 2. Centre - en liste af arrays fordi antal frø varierer
# np.savez gemmer flere arrays i én fil. Vi bruger en dict-lignende syntaks.
centers_dict = {f"img_{i:04d}": all_centers[i] for i in range(NUM_IMAGES)}
np.savez("centers_realistic.npz", **centers_dict)

# 3. Metadata - skala og andre relevante info
metadata = {
    "mm_per_pixel": MM_PER_PIXEL,
    "image_size": IMG_SIZE,
    "num_images": NUM_IMAGES,
    "data_type": "synthetic_videometer_compatible",
    "reflectance_units": True,
    "wavelength_bands": 1,  # gråtone for nu - bliver opdateret ved rigtige data
    "seed_count_stats": {
        "total": int(sum(seed_counts)),
        "mean": float(np.mean(seed_counts)),
        "min": int(min(seed_counts)),
        "max": int(max(seed_counts)),
    }
}
with open("metadata_realistic.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\nGemte filer:")
print("  X_realistic.npy        (billeder)")
print("  centers_realistic.npz  (center-koordinater)")
print("  metadata_realistic.json (skala m.m.)")


# =====================================================================
# VISUALISERING - sanity check at det ser realistisk ud
# =====================================================================
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

sample_indices = np.random.choice(NUM_IMAGES, size=4, replace=False)

for col, idx in enumerate(sample_indices):
    img = X[idx, :, :, 0]
    centers = all_centers[idx]

    # Top: bare billedet (sådan som modellen vil se det)
    axes[0, col].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[0, col].set_title(f"Billede {idx}\n({len(centers)} frø)")
    axes[0, col].axis('off')

    # Bund: billede + markerede centre (sådan som ground truth ser ud)
    axes[1, col].imshow(img, cmap='gray', vmin=0, vmax=1)
    if len(centers) > 0:
        axes[1, col].scatter(centers[:, 0], centers[:, 1],
                              s=15, c='lime', marker='o', edgecolors='red', linewidths=0.5)
    axes[1, col].set_title(f"Med ground truth centre")
    axes[1, col].axis('off')

plt.suptitle(f"Realistisk syntetisk datasæt - aflange frø tætpakket  "
             f"({MM_PER_PIXEL} mm/pixel, reflektans-værdier)",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("realistic_data_examples.png", dpi=100, bbox_inches='tight')
plt.show()

print("\nVisualisering gemt som realistic_data_examples.png")
print("\n=== Step 1B færdig! ===")
print("Næste skridt: byg en Videometer-kompatibel data-loader,")
print("og træn modellen på det her sværere datasæt.")
