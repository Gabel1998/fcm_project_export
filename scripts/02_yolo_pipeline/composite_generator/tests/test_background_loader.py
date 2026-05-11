"""Tests for background_loader.py."""
import numpy as np
import pytest

import background_loader


def test_load_returns_background_pool(tiny_blue_background_png):
    """load() returnerer BackgroundPool med korrekt antal billeder."""
    pool = background_loader.load([tiny_blue_background_png])
    assert len(pool.images) == 1
    assert len(pool.blue_zone_masks) == 1


def test_blue_zone_mask_identifies_blue_region(tiny_blue_background_png):
    """Blå zone-masken markerer pixels i midten som blå, ikke kanter."""
    pool = background_loader.load([tiny_blue_background_png])
    mask = pool.blue_zone_masks[0]
    # Midten skal være True (blå), kanterne False (brun)
    assert mask[75, 100] == True   # midten
    assert mask[0, 0] == False     # hjørne (brun)


def test_sample_returns_correct_size(tiny_blue_background_png):
    """sample(size) returnerer et numpy-array af præcis target-størrelse."""
    pool = background_loader.load([tiny_blue_background_png])
    crop = pool.sample(size=(80, 60))   # (width, height)
    assert crop.shape == (60, 80, 3)
    assert crop.dtype == np.uint8


def test_sample_only_returns_blue_region(tiny_blue_background_png):
    """sample() skal kun returnere crops der ligger inden for blå zone."""
    pool = background_loader.load([tiny_blue_background_png])
    # Lav 20 samples og verificer at alle er overvejende blå
    for _ in range(20):
        crop = pool.sample(size=(50, 50))
        # Mean blue-channel skal være højere end mean red — krude blå-test
        mean_b = crop[:, :, 2].mean()
        mean_r = crop[:, :, 0].mean()
        # Vores fixture-baggrund har RGB(30,100,180) i blå zone
        assert mean_b > mean_r, f"Crop er ikke blå: R={mean_r:.0f}, B={mean_b:.0f}"


def test_sample_fails_when_target_too_large(tiny_blue_background_png):
    """Hvis requested size er større end blå zone, hård fail."""
    pool = background_loader.load([tiny_blue_background_png])
    with pytest.raises(ValueError, match="for stor"):
        pool.sample(size=(500, 500))
