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

# Konstanter (samme som det modellen så under træning)
BACKGROUND_REFLECTANCE = 0.85
SEED_REFLECTANCE_MIN = 0.20
SEED_REFLECTANCE_MAX = 0.45
PACKING_GAP_PIXELS = 2
MAX_PLACEMENT_ATTEMPTS = 200
MM_PER_PIXEL = 0.05


def _ellipses_overlap(ellipse_a, ellipse_b, gap):
    """Konservativ overlap-tjek: afstand mellem centre vs. sum af radier."""
    center_x_a, center_y_a, length_a, _, _ = ellipse_a
    center_x_b, center_y_b, length_b, _, _ = ellipse_b
    radius_a = length_a / 2.0
    radius_b = length_b / 2.0
    delta_x = center_x_a - center_x_b
    delta_y = center_y_a - center_y_b
    return np.sqrt(delta_x * delta_x + delta_y * delta_y) < (radius_a + radius_b + gap)


def _draw_ellipse(image, center_x, center_y, length, width, angle_deg, color):
    """Tegn udfyldt ellipse direkte i numpy-arrayet."""
    # a og b er ellipsens semi-akser (matematisk notation)
    semi_axis_along_length = length / 2.0
    semi_axis_along_width = width / 2.0
    angle_rad = np.deg2rad(angle_deg)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    image_height, image_width = image.shape
    y_grid, x_grid = np.meshgrid(np.arange(image_height), np.arange(image_width), indexing='ij')
    x_shifted = x_grid - center_x
    y_shifted = y_grid - center_y
    x_rotated = cos_angle * x_shifted + sin_angle * y_shifted
    y_rotated = -sin_angle * x_shifted + cos_angle * y_shifted

    inside_ellipse = (x_rotated / semi_axis_along_length) ** 2 + (y_rotated / semi_axis_along_width) ** 2 < 1.0
    image[inside_ellipse] = color


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
    length_min, length_max, width_min, width_max = SEED_SIZE_PRESETS[seed_size_preset]

    # Start med lys plade-baggrund
    image = np.ones((img_size, img_size), dtype=np.float32) * BACKGROUND_REFLECTANCE

    placed_ellipses = []
    centers = []
    margin = length_max // 2 + 2

    for _ in range(num_seeds):
        length = int(rng.integers(length_min, length_max + 1))
        width = int(rng.integers(width_min, width_max + 1))

        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            center_x = int(rng.integers(margin, img_size - margin))
            center_y = int(rng.integers(margin, img_size - margin))
            angle = float(rng.uniform(0, 360))

            new_ellipse = (center_x, center_y, length, width, angle)
            has_collision = any(
                _ellipses_overlap(new_ellipse, existing_ellipse, PACKING_GAP_PIXELS)
                for existing_ellipse in placed_ellipses
            )
            if not has_collision:
                color = float(rng.uniform(SEED_REFLECTANCE_MIN, SEED_REFLECTANCE_MAX))
                _draw_ellipse(image, center_x, center_y, length, width, angle, color)
                placed_ellipses.append(new_ellipse)
                centers.append((center_x, center_y))
                break
        # Hvis vi ikke kunne placere efter MAX forsøg: pladen er fuld, skip
        # (vi forsøger ikke at placere flere — lader bare være)

    # Tilføj reflektans-støj
    noise = rng.standard_normal((img_size, img_size)) * noise_level
    image = np.clip(image + noise, 0.0, 1.0)

    # Konvertér til uint8 for visning og YOLO
    img_uint8 = (image * 255).astype(np.uint8)

    return img_uint8, centers, MM_PER_PIXEL
