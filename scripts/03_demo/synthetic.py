"""Syntetisk datagenerator — refaktoreret fra 01b_realistic_synthetic_data.py.

En ren funktion: parametre ind, (billede, centre, mm/pixel) ud.
Ingen Streamlit-afhængigheder.
"""
from typing import Literal

import numpy as np

# Mapping fra preset-navn til generator-værdier
NOISE_PRESETS = {
    "kalibreret": 0.02,   # som model blev trænet på
    "realistisk": 0.05,
    "svært":      0.10,
}

SEED_SIZE_PRESETS = {
    # (length_min, length_max, width_min, width_max) i pixels
    "mindre": (8, 12, 3, 5),
    "normal": (12, 18, 5, 7),  # som model blev trænet på
    "større": (18, 24, 7, 9),
}

# Konstanter (samme som det model'en så under træning)
BACKGROUND_REFLECTANCE = 0.85
SEED_REFLECTANCE_MIN = 0.20
SEED_REFLECTANCE_MAX = 0.45
PACKING_GAP_PIXELS = 2
MAX_PLACEMENT_ATTEMPTS = 200
MM_PER_PIXEL = 0.05


def _ellipses_overlap(e1, e2, gap):
    """Konservativ overlap-tjek: afstand mellem centre vs sum af radier."""
    cx1, cy1, L1, _, _ = e1
    cx2, cy2, L2, _, _ = e2
    r1, r2 = L1 / 2.0, L2 / 2.0
    dx, dy = cx1 - cx2, cy1 - cy2
    return np.sqrt(dx * dx + dy * dy) < (r1 + r2 + gap)


def _draw_ellipse(image, cx, cy, length, width, angle_deg, color):
    """Tegn udfyldt ellipse direkte i numpy-arrayet."""
    a, b = length / 2.0, width / 2.0
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    H, W = image.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    x_shifted, y_shifted = xx - cx, yy - cy
    x_rot = cos_a * x_shifted + sin_a * y_shifted
    y_rot = -sin_a * x_shifted + cos_a * y_shifted

    inside = (x_rot / a) ** 2 + (y_rot / b) ** 2 < 1.0
    image[inside] = color


def generate_image(
    num_seeds: int,
    img_size: int,
    noise_preset: Literal["kalibreret", "realistisk", "svært"],
    seed_size_preset: Literal["mindre", "normal", "større"],
    random_seed: int | None = None,
):
    """Generér ét syntetisk billede der ligner Videometer-data.

    Args:
        num_seeds: antal frø generatoren forsøger at placere (rejection sampling
            kan dropper nogle hvis pladen er fuld).
        img_size: kvadratisk billed-størrelse i pixels (fx 128 eller 256).
        noise_preset: navn fra NOISE_PRESETS — styrer reflektans-støj.
        seed_size_preset: navn fra SEED_SIZE_PRESETS — styrer frø-dimensioner.
        random_seed: None = ny tilfældig, ellers determistisk output.

    Returns:
        Tuple (img_uint8, centers, mm_per_pixel):
            img_uint8: (H, W) uint8 array i [0, 255]
            centers: liste af (x, y) tupler — ground truth centre
            mm_per_pixel: skala-faktor (typisk 0.05)

    Raises:
        ValueError: hvis preset-navn ikke findes.
    """
    # Validering først så fejlen er tydelig
    if noise_preset not in NOISE_PRESETS:
        raise ValueError(
            f"noise_preset '{noise_preset}' ukendt. "
            f"Brug en af: {list(NOISE_PRESETS.keys())}"
        )
    if seed_size_preset not in SEED_SIZE_PRESETS:
        raise ValueError(
            f"seed_size_preset '{seed_size_preset}' ukendt. "
            f"Brug en af: {list(SEED_SIZE_PRESETS.keys())}"
        )

    rng = np.random.default_rng(random_seed)
    noise_level = NOISE_PRESETS[noise_preset]
    L_min, L_max, W_min, W_max = SEED_SIZE_PRESETS[seed_size_preset]

    # Start med lys plade-baggrund
    image = np.ones((img_size, img_size), dtype=np.float32) * BACKGROUND_REFLECTANCE

    placed_ellipses = []
    centers = []
    margin = L_max // 2 + 2

    for _ in range(num_seeds):
        length = int(rng.integers(L_min, L_max + 1))
        width = int(rng.integers(W_min, W_max + 1))

        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            cx = int(rng.integers(margin, img_size - margin))
            cy = int(rng.integers(margin, img_size - margin))
            angle = float(rng.uniform(0, 360))

            new_e = (cx, cy, length, width, angle)
            collision = any(
                _ellipses_overlap(new_e, ex, PACKING_GAP_PIXELS)
                for ex in placed_ellipses
            )
            if not collision:
                color = float(rng.uniform(SEED_REFLECTANCE_MIN, SEED_REFLECTANCE_MAX))
                _draw_ellipse(image, cx, cy, length, width, angle, color)
                placed_ellipses.append(new_e)
                centers.append((cx, cy))
                break
        # Hvis vi ikke kunne placere efter MAX forsøg: pladen er fuld, skip
        # (vi forsøger ikke at placere flere — lader bare være)

    # Tilføj reflektans-støj
    noise = rng.standard_normal((img_size, img_size)) * noise_level
    image = np.clip(image + noise, 0.0, 1.0)

    # Konvertér til uint8 for visning og YOLO
    img_uint8 = (image * 255).astype(np.uint8)

    return img_uint8, centers, MM_PER_PIXEL
