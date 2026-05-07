"""
=====================================================================
 FCN Landmark Detection - Step 3: Evaluering og visualisering
=====================================================================

 Forudsaetning:
   Du har koert step 1 og step 2.
   Filerne X_synthetic.npy, Y_synthetic.npy og fcn_landmark_model.keras
   skal ligge i Colab-filsystemet.

 Hvad scriptet laver:
   1. Loader den traeede model og test-data
   2. Laver predictions paa billeder modellen ALDRIG har set
   3. Ekstraherer (x,y) koordinater fra heatmaps via peak detection
   4. Maaler hvor praecist modellen finder froene (precision, recall, F1)
   5. Visualiserer: input -> predicted heatmap -> markerede landmarks

 Til eksamen:
   Disse visualiseringer og metrics er det du SKAL vise. Det beviser
   at modellen ikke bare "har lavt loss", men at den faktisk finder
   froene korrekt.
"""

# =====================================================================
# IMPORTS
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# peak_local_max finder lokale toppe i et heatmap - vores "landmark detector"
from skimage.feature import peak_local_max

# Til at loade modellen med custom loss
import tensorflow as tf


# Re-definer custom loss saa load_model kan finde den
# (samme funktion som i step 2 - skal vaere identisk)
def weighted_mse(y_true, y_pred):
    weights = tf.where(y_true > 0.1, 100.0, 1.0)
    return tf.reduce_mean(tf.square(y_true - y_pred) * weights)


# =====================================================================
# KONFIGURATION
# =====================================================================
TEST_SPLIT = 0.2          # samme split som step 2
RANDOM_SEED = 42          # samme seed = samme test-saet

# Parametre til peak detection (hvordan vi konverterer heatmap -> koordinater)
PEAK_MIN_DISTANCE = 8         # minimum afstand mellem peaks (i pixels). Forhindrer dobbelt-detektioner
PEAK_THRESHOLD_REL = 0.3      # threshold som RELATIV andel af heatmap-max (mere robust end absolut)
                              # 0.3 = peaks skal være mindst 30% af det stærkeste peak

# Parameter til evaluering: hvor taet skal predicted vaere paa ground truth
# for at vi kalder det en "korrekt" detektion?
MATCH_DISTANCE = 5        # i pixels. Cirka radius af et frø.


# =====================================================================
# STEP A: GENSKAB SAMME TEST-SAET SOM I STEP 2
# =====================================================================
print("Loader data og model...")
X = np.load("X_synthetic.npy")
Y = np.load("Y_synthetic.npy")

# VIGTIG: samme random_state og test_size = samme split.
# Saa vi tester paa billeder modellen ALDRIG har set under traeningen.
_, X_test, _, Y_test = train_test_split(
    X, Y,
    test_size=TEST_SPLIT,
    random_state=RANDOM_SEED
)

print(f"Test-saet: {X_test.shape[0]} billeder")

# Load den traeede model - vi skal fortaelle Keras om vores custom loss
model = load_model(
    "fcn_landmark_model.keras",
    custom_objects={"weighted_mse": weighted_mse}
)
print("Model loadet")


# =====================================================================
# STEP B: PREDICT PAA HELE TEST-SAETTET
# =====================================================================
print("\nLaver predictions paa test-saettet...")

# model.predict koerer alle billeder igennem netvaerket og returnerer heatmaps
# Output shape: (N, 128, 128, 1) - samme format som Y_test
Y_predicted = model.predict(X_test, verbose=1)

print(f"Predictions shape: {Y_predicted.shape}")


# =====================================================================
# STEP C: HJAELPEFUNKTION - konverter heatmap til (x,y) koordinater
# =====================================================================
def heatmap_to_landmarks(heatmap, min_distance, threshold_rel):
    """
    Finder peaks i et heatmap og returnerer dem som (x, y) koordinater.

    Hvorfor ikke bare argmax?
    - argmax giver kun ÉN position (det stoerste pixel)
    - Vi har MANGE froe i hvert billede, saa vi skal finde flere peaks
    - peak_local_max finder alle lokale toppe der er hoejere end threshold

    Hvorfor RELATIV threshold (i stedet for absolut)?
    - Modellens output kan have forskellige absolutte vaerdier afhaengig
      af traeningsforhold. En relativ threshold (fx 30% af max) er
      meget mere robust paa tvaers af modeller.

    Returns: numpy array af form (N, 2) med (y, x) koordinater
    """
    # Fjern eventuelle ekstra dimensioner (fra (128,128,1) til (128,128))
    if heatmap.ndim == 3:
        heatmap = heatmap[:, :, 0]

    # Beregn absolut threshold baseret paa heatmap-max
    # Hvis hele heatmappet er meget svagt (ingen klare peaks), spring over
    if heatmap.max() < 0.001:
        return np.empty((0, 2), dtype=int)
    threshold_abs = threshold_rel * heatmap.max()

    # Find alle lokale maksima
    peaks = peak_local_max(
        heatmap,
        min_distance=min_distance,
        threshold_abs=threshold_abs
    )
    # peaks er nu shape (N, 2) med (row, col) = (y, x)
    return peaks


# =====================================================================
# STEP D: KVANTITATIV EVALUERING - hvor godt finder modellen froene?
# =====================================================================
print("\nKvantitativ evaluering paa hele test-saettet...")

# Vi taeller op over alle billeder:
#   true_positive  = predicted landmark ligger taet paa et ground truth landmark
#   false_positive = predicted landmark uden noget ground truth taet paa
#   false_negative = ground truth landmark som ingen prediction matchede
total_tp = 0
total_fp = 0
total_fn = 0

# Loop over alle test-billeder
for i in range(len(X_test)):
    # Hent predicted og ground truth landmarks for billede i
    predicted_landmarks = heatmap_to_landmarks(
        Y_predicted[i], PEAK_MIN_DISTANCE, PEAK_THRESHOLD_REL
    )
    truth_landmarks = heatmap_to_landmarks(
        Y_test[i], PEAK_MIN_DISTANCE, PEAK_THRESHOLD_REL
    )

    # Track hvilke ground truth landmarks der allerede er matchet
    # (saa vi ikke tæller samme ground truth flere gange)
    matched_truth = np.zeros(len(truth_landmarks), dtype=bool)

    # For hver predicted landmark: find naermeste ground truth
    for pred in predicted_landmarks:
        if len(truth_landmarks) == 0:
            # Ingen ground truth at matche -> false positive
            total_fp += 1
            continue

        # Beregn afstand til alle ground truth landmarks
        # np.linalg.norm beregner Euclidean distance
        distances = np.linalg.norm(truth_landmarks - pred, axis=1)

        # Find den naermeste der ikke allerede er matched
        valid = ~matched_truth
        if not valid.any():
            total_fp += 1
            continue

        # Saet allerede-matched-distancer til uendeligt
        masked_distances = np.where(valid, distances, np.inf)
        nearest_idx = np.argmin(masked_distances)
        nearest_dist = masked_distances[nearest_idx]

        # Hvis naermeste ligger inden for vores tolerance: korrekt detektion
        if nearest_dist <= MATCH_DISTANCE:
            total_tp += 1
            matched_truth[nearest_idx] = True
        else:
            # For langt fra alle ground truth -> false positive
            total_fp += 1

    # Ground truth uden matching prediction -> false negatives
    total_fn += np.sum(~matched_truth)

# Beregn klassiske metrics
precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n=== EVALUERING ({len(X_test)} test-billeder) ===")
print(f"True positives:  {total_tp}  (korrekt fundne froe)")
print(f"False positives: {total_fp}  (predictions uden et rigtigt frø)")
print(f"False negatives: {total_fn}  (oversete froe)")
print(f"")
print(f"Precision: {precision:.3f}  (hvor mange af predictions er korrekte)")
print(f"Recall:    {recall:.3f}  (hvor mange af de rigtige froe finder vi)")
print(f"F1 score:  {f1:.3f}  (harmonisk middel - samlet score)")


# =====================================================================
# STEP E: KVALITATIV VISUALISERING - vis predictions paa konkrete billeder
# =====================================================================
print("\nLaver visualiseringer...")

# Vaelg 4 tilfaeldige test-billeder at vise
np.random.seed(123)  # forskelligt fra step 1 saa vi ikke ser samme billeder
sample_indices = np.random.choice(len(X_test), size=4, replace=False)

# Lav en figur med 4 raekker (et per eksempel) og 3 kolonner
fig, axes = plt.subplots(4, 3, figsize=(13, 16))

for row, idx in enumerate(sample_indices):
    # Hent input, ground truth og prediction
    img = X_test[idx, :, :, 0]
    truth_heatmap = Y_test[idx, :, :, 0]
    pred_heatmap = Y_predicted[idx, :, :, 0]

    # Find landmarks fra begge
    truth_landmarks = heatmap_to_landmarks(
        Y_test[idx], PEAK_MIN_DISTANCE, PEAK_THRESHOLD_REL
    )
    pred_landmarks = heatmap_to_landmarks(
        Y_predicted[idx], PEAK_MIN_DISTANCE, PEAK_THRESHOLD_REL
    )

    # --- Kolonne 1: Input billede + ground truth + prediction overlay ---
    axes[row, 0].imshow(img, cmap='gray')
    # Ground truth markeres med groenne cirkler
    if len(truth_landmarks) > 0:
        axes[row, 0].scatter(truth_landmarks[:, 1], truth_landmarks[:, 0],
                             s=100, facecolors='none', edgecolors='lime',
                             linewidth=2, label=f'Ground truth ({len(truth_landmarks)})')
    # Predictions markeres med roede krydser
    if len(pred_landmarks) > 0:
        axes[row, 0].scatter(pred_landmarks[:, 1], pred_landmarks[:, 0],
                             s=80, marker='x', color='red', linewidth=2,
                             label=f'Predicted ({len(pred_landmarks)})')
    axes[row, 0].set_title(f"Billede {idx}: Input + landmarks")
    axes[row, 0].legend(loc='upper right', fontsize=9)
    axes[row, 0].axis('off')

    # --- Kolonne 2: Ground truth heatmap (det modellen skulle laere) ---
    axes[row, 1].imshow(truth_heatmap, cmap='hot', vmin=0, vmax=1)
    axes[row, 1].set_title("Ground truth heatmap")
    axes[row, 1].axis('off')

    # --- Kolonne 3: Predicted heatmap (det modellen rent faktisk producerer) ---
    axes[row, 2].imshow(pred_heatmap, cmap='hot', vmin=0, vmax=1)
    axes[row, 2].set_title("Predicted heatmap")
    axes[row, 2].axis('off')

plt.suptitle(f"FCN Landmark Detection - Test resultater  "
             f"(Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f})",
             fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig("evaluation_results.png", dpi=100, bbox_inches='tight')
plt.show()

print("Visualiseringer gemt som evaluation_results.png")


# =====================================================================
# STEP F: BONUS - lav et "demo"-billede til praesentationen
# =====================================================================
# Et enkelt stort billede der viser én test-case meget tydeligt.
# Brug dette som hovedslide i din eksamenspraesentation.
demo_idx = sample_indices[0]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

img = X_test[demo_idx, :, :, 0]
pred_heatmap = Y_predicted[demo_idx, :, :, 0]
pred_landmarks = heatmap_to_landmarks(
    Y_predicted[demo_idx], PEAK_MIN_DISTANCE, PEAK_THRESHOLD_REL
)

axes[0].imshow(img, cmap='gray')
axes[0].set_title("1. Input billede", fontsize=14)
axes[0].axis('off')

axes[1].imshow(pred_heatmap, cmap='hot', vmin=0, vmax=1)
axes[1].set_title("2. Predicted heatmap (FCN output)", fontsize=14)
axes[1].axis('off')

axes[2].imshow(img, cmap='gray')
if len(pred_landmarks) > 0:
    axes[2].scatter(pred_landmarks[:, 1], pred_landmarks[:, 0],
                    s=120, marker='x', color='red', linewidth=3)
axes[2].set_title(f"3. Detekterede landmarks ({len(pred_landmarks)} froe)", fontsize=14)
axes[2].axis('off')

plt.suptitle("FCN Landmark Detection Pipeline", fontsize=16, y=1.05)
plt.tight_layout()
plt.savefig("demo_pipeline.png", dpi=100, bbox_inches='tight')
plt.show()

print("Demo-billede gemt som demo_pipeline.png")
print("\n=== Step 3 faerdig! ===")
print("Du har nu en komplet, fungerende FCN landmark detection pipeline.")
