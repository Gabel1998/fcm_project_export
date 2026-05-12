"""Tests for background_loader.py."""
import numpy as np
import pytest

import background_loader


def test_load_returns_background_pool(tiny_blue_background_png):
    """load() returnerer BackgroundPool med korrekt antal billeder."""
    pool = background_loader.load([tiny_blue_background_png])
    assert len(pool.images) == 1
    assert len(pool.blue_zone_masks) == 1


def test_load_pre_crops_to_blue_zone(tiny_blue_background_png):
    """Efter load() skal billedet være croppet til blå zone (mindre end original)."""
    pool = background_loader.load([tiny_blue_background_png])
    h, w = pool.images[0].shape[:2]
    # Original fixture er 200x150, blå zone er rows 20-130 cols 30-170 = ~140x110
    assert w < 200, f"Bredde {w} ikke croppet (original 200)"
    assert h < 150, f"Højde {h} ikke croppet (original 150)"


def test_load_resizes_multiple_backgrounds_to_canonical(tiny_blue_background_png, tmp_path):
    """Hvis flere baggrunde har forskellige størrelser, resizes alle til den første."""
    # Lav en anden baggrund med anderledes blå-zone størrelse
    from PIL import Image
    bgr = np.zeros((180, 240, 3), dtype=np.uint8)
    bgr[:, :] = (50, 30, 80)
    bgr[25:155, 35:205] = (180, 100, 30)
    bg2_path = tmp_path / "tiny_belt2.png"
    Image.fromarray(bgr[..., ::-1], mode="RGB").save(bg2_path)

    pool = background_loader.load([tiny_blue_background_png, bg2_path])
    # Begge skal have samme størrelse (første billedes blå-zone-størrelse)
    assert pool.images[0].shape == pool.images[1].shape


def test_sample_returns_correct_size(tiny_blue_background_png):
    """sample(size) returnerer et numpy-array af præcis target-størrelse."""
    pool = background_loader.load([tiny_blue_background_png])
    crop = pool.sample(size=(80, 60))   # (width, height)
    assert crop.shape == (60, 80, 3)
    assert crop.dtype == np.uint8


def test_sample_only_returns_blue_region(tiny_blue_background_png):
    """sample() skal returnere overvejende blå crop (hele pre-cropped zone er blå)."""
    pool = background_loader.load([tiny_blue_background_png])
    for _ in range(10):
        crop = pool.sample(size=(50, 50))
        mean_b = crop[:, :, 2].mean()
        mean_r = crop[:, :, 0].mean()
        assert mean_b > mean_r, f"Crop er ikke blå: R={mean_r:.0f}, B={mean_b:.0f}"
