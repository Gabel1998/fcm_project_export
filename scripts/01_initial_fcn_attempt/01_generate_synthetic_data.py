"""
=====================================================================
 FCN Landmark Detection - Step 1: Generate synthetic training data
=====================================================================

 Formaal:
   Inden vi faar rigtige Videometer-data, bygger vi et syntetisk datasaet.
   Det giver os fuld kontrol og perfekte annotationer, saa vi kan teste
   at vores FCN-arkitektur virker, foer vi bruger tid paa rigtige data.

 Hvad scriptet laver:
   1. Genererer billeder af "froe" (cirkler) paa en plade-baggrund
   2. Genererer matchende heatmaps med Gaussian blobs paa froe-positioner
   3. Visualiserer et par eksempler saa vi kan se det virker

 Anvendelse:
   Koer i Google Colab. Output gemmes som numpy arrays, der kan
   indlaeses i naeste step (traening af FCN).
"""

# =====================================================================
# IMPORTS - alle biblioteker vi skal bruge
# =====================================================================
import numpy as np                    # numerisk regning og arrays
import matplotlib.pyplot as plt       # visualisering
from scipy.ndimage import gaussian_filter  # til at lave Gaussian blobs paa heatmaps

# =====================================================================
# KONFIGURATION - juster disse parametre efter behov
# =====================================================================
IMG_SIZE = 128            # billed-stoerrelse: 128x128 pixels (godt kompromis hastighed/detalje)
NUM_IMAGES = 300          # antal billeder i datasaettet
MIN_SEEDS = 5             # minimum antal froe pr. billede
MAX_SEEDS = 15            # maksimum antal froe pr. billede
SEED_RADIUS_MIN = 4       # minimum radius af et frø i pixels
SEED_RADIUS_MAX = 7       # maksimum radius af et frø i pixels
GAUSSIAN_SIGMA = 3.0      # spredning af Gaussian blob i heatmap (stoerre = bredere blob)
NOISE_LEVEL = 0.05        # billedstoej (0 = intet, 1 = meget)
RANDOM_SEED = 42          # fast seed saa resultater er reproducerbare

# Saet random seed saa vi faar samme datasaet hver gang vi koerer scriptet
np.random.seed(RANDOM_SEED)


# =====================================================================
# FUNKTION 1: Lav et enkelt syntetisk billede med froe
# =====================================================================
def generate_image_and_landmarks(img_size, min_seeds, max_seeds,
                                  radius_min, radius_max, noise_level):
    """
    Genererer ét billede med tilfaeldige "froe" og returnerer
    billedet samt liste af (x, y) koordinater hvor froene er.

    Hvorfor det virker som proxy for rigtige data:
    - Froene har en center-of-mass (det er landmarken vi vil finde)
    - Baggrunden har stoej (som rigtige fotos)
    - Forskelligt antal og stoerrelser (som rigtige plade-billeder)
    """

    # Start med en plade-baggrund: lys graa med lidt variation
    # np.ones giver alle pixels vaerdi 1.0 (hvid). Vi ganger med 0.85 for lys graa.
    image = np.ones((img_size, img_size), dtype=np.float32) * 0.85

    # Vaelg tilfaeldigt antal froe i dette billede (mellem min og max)
    num_seeds = np.random.randint(min_seeds, max_seeds + 1)

    # Liste til at gemme (x, y) koordinaterne for hvert frø
    landmarks = []

    # Lav et grid af koordinater - bruges til at tegne cirkler effektivt
    # np.meshgrid laver to 2D arrays: yy[i,j]=i og xx[i,j]=j
    yy, xx = np.meshgrid(np.arange(img_size), np.arange(img_size), indexing='ij')

    # Loop: tilfoej et frø ad gangen
    for _ in range(num_seeds):
        # Vaelg tilfaeldig position - hold lidt afstand til kanten saa froene ikke skaeres af
        margin = radius_max + 2
        cx = np.random.randint(margin, img_size - margin)  # center x
        cy = np.random.randint(margin, img_size - margin)  # center y

        # Vaelg tilfaeldig radius
        r = np.random.randint(radius_min, radius_max + 1)

        # Tegn en cirkel: alle pixels hvor afstanden til centrum er < radius
        # (xx - cx)**2 + (yy - cy)**2 er afstand-kvadreret. < r**2 betyder inde i cirklen.
        distance_squared = (xx - cx) ** 2 + (yy - cy) ** 2
        seed_mask = distance_squared < r ** 2

        # Giv froet en moerk farve med lidt variation (saa de ikke alle er ens)
        seed_color = np.random.uniform(0.15, 0.35)
        image[seed_mask] = seed_color

        # Gem positionen som (x, y) - dette er vores "ground truth landmark"
        landmarks.append((cx, cy))

    # Tilfoej Gaussian stoej saa billedet ligner et rigtigt foto
    # randn giver normalfordelte tal. Vi ganger med noise_level for at styre styrken.
    noise = np.random.randn(img_size, img_size) * noise_level
    image = image + noise

    # Klip vaerdier saa de holder sig i [0, 1] - vigtigt for neurale netvaerk
    image = np.clip(image, 0.0, 1.0)

    return image, landmarks


# =====================================================================
# FUNKTION 2: Lav et heatmap fra en liste af landmark-positioner
# =====================================================================
def landmarks_to_heatmap(landmarks, img_size, sigma):
    """
    Konverterer (x, y) koordinater til et heatmap med Gaussian blobs.

    Hvorfor heatmap og ikke bare punkter?
    - Et enkelt pixel-punkt er for "haardt" et signal at laere
    - Gaussian blobs giver modellen en blød graduering: "tæt på centrum = høj værdi"
    - Det stabiliserer traeningen markant
    """

    # Start med et tomt heatmap (alle nuller)
    heatmap = np.zeros((img_size, img_size), dtype=np.float32)

    # Saet en 1.0 i hver landmark-position
    for (cx, cy) in landmarks:
        heatmap[cy, cx] = 1.0  # NB: numpy bruger (row, col) = (y, x)

    # Anvend Gaussian filter - dette "smører" hver 1.0 ud til en bloed klat
    # sigma styrer hvor bred klatten bliver
    heatmap = gaussian_filter(heatmap, sigma=sigma)

    # Normaliser saa max-vaerdi er 1.0 (gør traeningen mere stabil)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    return heatmap


# =====================================================================
# HOVEDLOGIK: Generér hele datasaettet
# =====================================================================
print(f"Genererer {NUM_IMAGES} billeder paa {IMG_SIZE}x{IMG_SIZE} pixels...")

# Forhaandstildel arrays - hurtigere end at appende i et loop
# Shape: (NUM_IMAGES, IMG_SIZE, IMG_SIZE) - en stak af 2D billeder
X = np.zeros((NUM_IMAGES, IMG_SIZE, IMG_SIZE), dtype=np.float32)
Y = np.zeros((NUM_IMAGES, IMG_SIZE, IMG_SIZE), dtype=np.float32)

# Gem ogsaa de raa landmark-koordinater - kan bruges til evaluering senere
all_landmarks = []

# Loop og generér billede + heatmap par
for i in range(NUM_IMAGES):
    # Step 1: lav billede og faa landmarks
    img, landmarks = generate_image_and_landmarks(
        IMG_SIZE, MIN_SEEDS, MAX_SEEDS,
        SEED_RADIUS_MIN, SEED_RADIUS_MAX, NOISE_LEVEL
    )

    # Step 2: konvertér landmarks til heatmap
    heatmap = landmarks_to_heatmap(landmarks, IMG_SIZE, GAUSSIAN_SIGMA)

    # Gem i de praeparerede arrays
    X[i] = img
    Y[i] = heatmap
    all_landmarks.append(landmarks)

    # Print progress hver 50 billeder saa vi kan se det koerer
    if (i + 1) % 50 == 0:
        print(f"  Genereret {i + 1}/{NUM_IMAGES} billeder")

print("Faerdig med generering!")
print(f"X shape: {X.shape}  (billeder)")
print(f"Y shape: {Y.shape}  (heatmaps)")


# =====================================================================
# TILFOEJ KANAL-DIMENSION - Keras forventer (H, W, kanaler)
# =====================================================================
# Vi har lige nu (N, H, W). Keras Conv2D vil have (N, H, W, kanaler).
# np.expand_dims tilfoejer en ny dimension. axis=-1 betyder "i slutningen".
X = np.expand_dims(X, axis=-1)  # nu (N, H, W, 1) - 1 graatone-kanal
Y = np.expand_dims(Y, axis=-1)  # samme for heatmaps

print(f"\nEfter tilfoejelse af kanal-dimension:")
print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")


# =====================================================================
# GEM DATA TIL DISK - saa vi kan loade det i naeste script
# =====================================================================
np.save("X_synthetic.npy", X)
np.save("Y_synthetic.npy", Y)
print("\nData gemt som X_synthetic.npy og Y_synthetic.npy")


# =====================================================================
# VISUALISER ET PAR EKSEMPLER - sanity check at det virker
# =====================================================================
# Tegn 4 tilfaeldigt valgte billeder og deres heatmaps side om side
fig, axes = plt.subplots(2, 4, figsize=(14, 7))

# Vaelg 4 tilfaeldige indeks fra datasaettet
sample_indices = np.random.choice(NUM_IMAGES, size=4, replace=False)

for col, idx in enumerate(sample_indices):
    # Top raekke: input-billedet
    # X[idx, :, :, 0] henter billede idx, alle pixels, kun 1. kanal
    axes[0, col].imshow(X[idx, :, :, 0], cmap='gray')
    axes[0, col].set_title(f"Billede {idx}\n({len(all_landmarks[idx])} froe)")
    axes[0, col].axis('off')

    # Bund raekke: heatmap target som modellen skal laere at predicte
    axes[1, col].imshow(Y[idx, :, :, 0], cmap='hot')
    axes[1, col].set_title(f"Heatmap target")
    axes[1, col].axis('off')

plt.suptitle("Syntetisk datasaet - billeder (top) og ground truth heatmaps (bund)",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("synthetic_data_examples.png", dpi=100, bbox_inches='tight')
plt.show()

print("\nVisualisering gemt som synthetic_data_examples.png")
print("\n=== Step 1 faerdig! Klar til at bygge FCN-modellen ===")
