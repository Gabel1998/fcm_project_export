"""Tests for synthetic data generator."""
import sys
from pathlib import Path

import numpy as np
import pytest

# Tilføj demo-mappen til sys.path så vi kan importere modulet direkte
DEMO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO_DIR))

import synthetic  # noqa: E402


def test_default_parametre_giver_korrekt_shape():
    """Default-kald skal returnere et 256x256 billede og en liste af centre."""
    img, centers, mm_per_pixel = synthetic.generate_image(
        num_seeds=58, img_size=256, noise_preset="kalibreret",
        seed_size_preset="normal", random_seed=42,
    )
    assert img.shape == (256, 256), f"forventet (256,256), fik {img.shape}"
    assert img.dtype == np.uint8, f"forventet uint8, fik {img.dtype}"
    assert isinstance(centers, list)
    assert all(len(center) == 2 for center in centers), "hvert center skal være (x, y)"
    assert mm_per_pixel == 0.05


def test_lille_billede_virker():
    """128x128 skal også fungere."""
    img, centers, _ = synthetic.generate_image(
        num_seeds=20, img_size=128, noise_preset="kalibreret",
        seed_size_preset="normal", random_seed=1,
    )
    assert img.shape == (128, 128)
    # mindst nogle frø placeret (rejection sampling kan dropper nogle)
    assert len(centers) >= 5


def test_random_seed_er_deterministisk():
    """Samme random_seed → samme output."""
    image_first, centers_first, _ = synthetic.generate_image(
        num_seeds=30, img_size=256, noise_preset="kalibreret",
        seed_size_preset="normal", random_seed=123,
    )
    image_second, centers_second, _ = synthetic.generate_image(
        num_seeds=30, img_size=256, noise_preset="kalibreret",
        seed_size_preset="normal", random_seed=123,
    )
    assert np.array_equal(image_first, image_second)
    assert centers_first == centers_second


def test_centre_er_inden_for_billedet():
    """Alle centre skal ligge inden for billed-grænserne."""
    img, centers, _ = synthetic.generate_image(
        num_seeds=50, img_size=256, noise_preset="kalibreret",
        seed_size_preset="normal", random_seed=7,
    )
    for x, y in centers:
        assert 0 <= x < 256, f"x={x} ude af billedet"
        assert 0 <= y < 256, f"y={y} ude af billedet"


def test_invalid_preset_giver_fejl():
    """Ukendt preset-navn skal kaste ValueError."""
    with pytest.raises(ValueError):
        synthetic.generate_image(
            num_seeds=30, img_size=256, noise_preset="DETTE_FINDES_IKKE",
            seed_size_preset="normal", random_seed=1,
        )
