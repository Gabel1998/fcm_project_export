"""
=====================================================================
 FCN Landmark Detection - Step 2: Byg og trae FCN-modellen
=====================================================================

 Forudsaetning:
   Du har koert step 1 og har filerne X_synthetic.npy og Y_synthetic.npy
   liggende i Colab-filsystemet.

 Hvad scriptet laver:
   1. Loader datasaettet fra step 1
   2. Splitter i traenings- og test-saet
   3. Bygger en FCN-arkitektur (encoder + decoder)
   4. Traener modellen
   5. Gemmer modellen som .keras-fil
   6. Visualiserer trae-kurven (loss over tid)

 Forskel fra Jons CNN-eksempler (bogstaver, cifre, groennsager):
   - INGEN Flatten + Dense layers (ville oedelaegge spatial information)
   - Conv2DTranspose laeger til at gendanne bilede-stoerrelsen
   - Loss = MSE (regression paa heatmaps), ikke categorical_crossentropy
"""

# =====================================================================
# IMPORTS - bygger paa Jons standard Keras-stack
# =====================================================================
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split  # samme som Jons DNN-guide
from tensorflow.keras.models import Sequential        # samme som Jons CNN-eksempler
from tensorflow.keras.layers import (
    Conv2D,           # convolutional layer - kender du fra bogstaver/cifre
    MaxPool2D,        # max pooling - kender du fra Jons CNN
    Conv2DTranspose,  # NYT: "omvendt" convolution der gør billedet stoerre
    Input             # input-lag - definerer billedets form
)
from tensorflow.keras.optimizers import Adam          # samme optimizer som Jone bruger
import tensorflow as tf                               # til custom loss


# =====================================================================
# KONFIGURATION
# =====================================================================
IMG_SIZE = 128         # skal matche step 1
BATCH_SIZE = 16        # antal billeder modellen ser pr. opdatering
EPOCHS = 60            # antal gange modellen ser hele datasaettet
LEARNING_RATE = 0.001  # standard Adam learning rate
TEST_SPLIT = 0.2       # 20% til test, 80% til traening
RANDOM_SEED = 42


# =====================================================================
# STEP A: LOAD DATA FRA STEP 1
# =====================================================================
print("Loader data fra step 1...")
X = np.load("X_synthetic.npy")
Y = np.load("Y_synthetic.npy")
print(f"X shape: {X.shape}  (billeder)")
print(f"Y shape: {Y.shape}  (heatmaps)")


# =====================================================================
# STEP B: SPLIT I TRAENING OG TEST
# =====================================================================
# train_test_split er samme funktion Jone bruger i sin DNN-guide.
# random_state sikrer at vi faar samme split hver gang (reproducerbart).
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=TEST_SPLIT,
    random_state=RANDOM_SEED
)

print(f"\nTraening: {X_train.shape[0]} billeder")
print(f"Test:     {X_test.shape[0]} billeder")


# =====================================================================
# STEP C: BYG FCN-ARKITEKTUREN
# =====================================================================
# Tankegangen:
#   ENCODER (forward path): Conv2D + MaxPool2D laver billedet mindre,
#   men ekstraherer features. Det er PRAECIS som Jons CNN.
#
#   DECODER (backward path): Conv2DTranspose gør featuresne tilbage
#   til samme stoerrelse som input. Output er ét heatmap.
#
# Visualisering:
#
#   Input 128x128x1
#        |
#        v  Conv2D + Pool   (encoder dybere/mindre)
#   Feature 64x64x32
#        |
#        v  Conv2D + Pool
#   Feature 32x32x64
#        |
#        v  Conv2D
#   Feature 32x32x64        (bottom of "U")
#        |
#        v  Conv2DTranspose (decoder dybere/storre)
#   Feature 64x64x32
#        |
#        v  Conv2DTranspose
#   Feature 128x128x16
#        |
#        v  Conv2D output 1 channel (heatmap)
#   Output 128x128x1

print("\nBygger FCN-modellen...")

model = Sequential()

# --- INPUT LAYER ---
# Definerer formen paa input. (H, W, kanaler) = (128, 128, 1) for graatone.
# Naar Videometer-data kommer, skal kun "1" aendres til antal kanaler.
model.add(Input(shape=(IMG_SIZE, IMG_SIZE, 1)))

# --- ENCODER: Block 1 ---
# 32 filtre, hver kerne 3x3, ReLU aktivering. Padding="same" bevarer stoerrelsen.
model.add(Conv2D(filters=32, kernel_size=3, activation="relu", padding="same"))
# Pool 2x2: halverer billedet (128 -> 64). Pr. spatial info bevares delvis.
model.add(MaxPool2D(pool_size=2))

# --- ENCODER: Block 2 ---
# Flere filtre nu fordi billedet er mindre. Detaljer = mere abstrakte mønstre.
model.add(Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"))
model.add(MaxPool2D(pool_size=2))   # 64 -> 32

# --- BOTTOM (mindste rum, mest abstrakte features) ---
model.add(Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"))

# --- DECODER: Block 1 (upsampling) ---
# Conv2DTranspose med strides=2 fordobler billed-stoerrelsen.
# Det er "omvendt convolution" - tager features og spreder dem ud igen.
model.add(Conv2DTranspose(filters=32, kernel_size=3, strides=2,
                          activation="relu", padding="same"))   # 32 -> 64

# --- DECODER: Block 2 (upsampling) ---
model.add(Conv2DTranspose(filters=16, kernel_size=3, strides=2,
                          activation="relu", padding="same"))   # 64 -> 128

# --- OUTPUT LAYER ---
# 1 filter (1 heatmap-kanal). Sigmoid presser output i [0, 1] - matcher
# vores normaliserede ground truth heatmaps.
# kernel_size=1: 1x1 convolution - "samler" alle features til ét tal pr. pixel.
model.add(Conv2D(filters=1, kernel_size=1, activation="sigmoid", padding="same"))

# Print arkitekturen saa vi kan se den (skal vises som screenshot til eksamen!)
model.summary()


# =====================================================================
# CUSTOM LOSS: Vaegtet MSE
# =====================================================================
# Problemet med standard MSE: heatmaps er ~97% nuller. Modellen kan
# opnaa lav loss ved bare at predicte 0 overalt - men saa finder den
# ingen peaks!
#
# Loesning: vaegt foreground-pixels (hvor heatmap > 0) tungere.
# Saa "betaler" modellen mere for at misse en peak end at predicte
# en lille falsk peak i baggrunden.
def weighted_mse(y_true, y_pred):
    # Hvor groud truth har vaerdi > 0.1 = foreground (taet paa landmark)
    # Hvor ground truth er 0 = baggrund
    foreground_weight = 100.0   # foreground-fejl vaegtes 100x tungere
    background_weight = 1.0

    # Beregn vaegt-mask baseret paa target
    weights = tf.where(y_true > 0.1, foreground_weight, background_weight)

    # Vaegtet kvadratisk fejl
    squared_error = tf.square(y_true - y_pred)
    weighted_error = squared_error * weights

    return tf.reduce_mean(weighted_error)


# =====================================================================
# STEP D: COMPILE MODELLEN
# =====================================================================
# Forklaring af valg:
#   - Optimizer Adam: standard valg, samme som Jone bruger
#   - Loss = vaegtet MSE: vi predicter et kontinuerligt heatmap (regression).
#     Standard MSE ville faa modellen til at "spille det sikre kort" og
#     predicte 0 overalt. Vores vaegtede version straffer misset af peaks
#     hardere end falske peaks i baggrunden.
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=weighted_mse,
    metrics=["mae"]   # mean absolute error - en ekstra maaling vi kan se under traeningen
)


# =====================================================================
# STEP E: TRAEN MODELLEN
# =====================================================================
print("\nStarter traening...")
print(f"Epochs: {EPOCHS}, batch size: {BATCH_SIZE}")

history = model.fit(
    X_train, Y_train,                  # input og target til traening
    validation_data=(X_test, Y_test),  # test paa data modellen ikke har set
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1                          # 1 = vis progress bar
)


# =====================================================================
# STEP F: GEM MODELLEN (samme format som Jons Keras-til-webserver guide)
# =====================================================================
model.save("fcn_landmark_model.keras")
print("\nModel gemt som fcn_landmark_model.keras")


# =====================================================================
# STEP G: VISUALISER TRAE-KURVEN
# =====================================================================
# Loss-kurven viser om modellen laerer. Vi vil se:
#   - train_loss falder = modellen laerer
#   - val_loss falder ogsaa = modellen generaliserer
#   - val_loss begynder at stige igen = overfitting
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Plot loss
axes[0].plot(history.history["loss"], label="Træning", linewidth=2)
axes[0].plot(history.history["val_loss"], label="Validering", linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss (MSE)")
axes[0].set_title("Trae-kurve - Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot mean absolute error
axes[1].plot(history.history["mae"], label="Træning", linewidth=2)
axes[1].plot(history.history["val_mae"], label="Validering", linewidth=2)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Mean Absolute Error")
axes[1].set_title("Trae-kurve - MAE")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=100, bbox_inches='tight')
plt.show()

print("\nTraeningskurver gemt som training_curves.png")
print("\n=== Step 2 faerdig! ===")
print("Naeste skridt: koer step 3 for at evaluere modellen og se predictions")
