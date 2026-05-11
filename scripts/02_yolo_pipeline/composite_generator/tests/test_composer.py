"""Tests for composer.py."""
import numpy as np
import pytest

import composer
from seed_loader import SeedStamp
from packing import Placement


def _make_stamp(w: int, h: int, idx: int = 0) -> SeedStamp:
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = 200    # lys gul-agtig
    rgba[:, :, 3] = 255     # solid alpha
    return SeedStamp(rgba=rgba, source_id=f"seed_{idx}",
                     bbox_after_crop=(0, 0, w, h))


def _make_bg(w: int, h: int) -> np.ndarray:
    return np.full((h, w, 3), 50, dtype=np.uint8)   # mørk baggrund


def test_render_returns_correct_image_size():
    """Output-billedets størrelse matcher baggrunds-størrelsen."""
    bg = _make_bg(200, 150)
    stamps = [_make_stamp(20, 30)]
    placements = [Placement(center_x=100, center_y=75, rotation_deg=0, seed_index=0)]
    result = composer.render(bg, stamps, placements,
                              shadow_config=None, feather_px=0.0)
    assert result.image.shape == (150, 200, 3)
    assert result.image.dtype == np.uint8


def test_render_returns_centers_matching_placements():
    """Antal centers = antal placements (når alle er inden for canvas)."""
    bg = _make_bg(300, 300)
    stamps = [_make_stamp(20, 20) for _ in range(3)]
    placements = [
        Placement(center_x=50, center_y=50, rotation_deg=0, seed_index=0),
        Placement(center_x=150, center_y=150, rotation_deg=45, seed_index=1),
        Placement(center_x=250, center_y=100, rotation_deg=90, seed_index=2),
    ]
    result = composer.render(bg, stamps, placements,
                              shadow_config=None, feather_px=0.0)
    assert len(result.centers) == 3
    assert len(result.bbox_sizes) == 3


def test_render_actually_modifies_background():
    """Efter render skal pixels under frø være anderledes end baggrund."""
    bg = _make_bg(200, 200)
    stamps = [_make_stamp(40, 40)]
    placements = [Placement(center_x=100, center_y=100, rotation_deg=0, seed_index=0)]
    result = composer.render(bg, stamps, placements,
                              shadow_config=None, feather_px=0.0)
    # Pixel ved center skal nu have stamp-farve (200), ikke baggrunds-farve (50)
    assert result.image[100, 100, 0] > 100


def test_render_centers_within_canvas():
    """Alle registrerede centers ligger inden for canvas."""
    bg = _make_bg(200, 200)
    stamps = [_make_stamp(20, 20) for _ in range(5)]
    placements = [
        Placement(center_x=50, center_y=50, rotation_deg=0, seed_index=0),
        Placement(center_x=150, center_y=150, rotation_deg=0, seed_index=1),
        Placement(center_x=100, center_y=100, rotation_deg=0, seed_index=2),
        Placement(center_x=30, center_y=180, rotation_deg=0, seed_index=3),
        Placement(center_x=180, center_y=30, rotation_deg=0, seed_index=4),
    ]
    result = composer.render(bg, stamps, placements,
                              shadow_config=None, feather_px=0.0)
    for cx, cy in result.centers:
        assert 0 <= cx < 200
        assert 0 <= cy < 200
