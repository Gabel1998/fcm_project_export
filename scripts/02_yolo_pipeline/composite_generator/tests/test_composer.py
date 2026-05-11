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


def test_render_applies_shadow():
    """Når shadow.enabled=True, skal pixels under skygge være mørkere end uden skygge."""
    bg = _make_bg(200, 200)
    stamps = [_make_stamp(40, 40)]
    placements = [Placement(center_x=100, center_y=100, rotation_deg=0, seed_index=0)]

    no_shadow = composer.render(bg, stamps, placements,
                                  shadow_config=None, feather_px=0.0)
    with_shadow = composer.render(bg, stamps, placements,
                                    shadow_config={"enabled": True,
                                                    "offset_px": [10, 10],
                                                    "blur_sigma": 3.0,
                                                    "opacity": 0.5},
                                    feather_px=0.0)
    # Pixel nedenfor og til højre for stampen skal være mørkere med skygge
    no_sh_val = no_shadow.image[125, 125, :].mean()
    with_sh_val = with_shadow.image[125, 125, :].mean()
    assert with_sh_val < no_sh_val - 5, \
        f"Skygge ikke synlig: uden={no_sh_val:.1f}, med={with_sh_val:.1f}"


def test_render_applies_feather():
    """Med feather > 0, skal kanten af paste-området være blødere."""
    bg = _make_bg(200, 200)
    stamps = [_make_stamp(40, 40)]
    placements = [Placement(center_x=100, center_y=100, rotation_deg=0, seed_index=0)]

    no_feather = composer.render(bg, stamps, placements,
                                   shadow_config=None, feather_px=0.0)
    with_feather = composer.render(bg, stamps, placements,
                                     shadow_config=None, feather_px=3.0)

    # Lige uden for stamp-grænsen: med feather skal pixelen være en blanding,
    # uden feather er pixelen ren baggrund. Tjek gradient.
    edge_no = no_feather.image[80, 100, 0]      # ren baggrund
    edge_with = with_feather.image[80, 100, 0]  # bør være blandet
    assert edge_with != edge_no, "Feather har ingen effekt"


def test_render_applies_scale_jitter():
    """Med scale > 1.0 i placement, skal bbox være større."""
    bg = _make_bg(400, 400)
    stamps = [_make_stamp(30, 30)]

    placement_small = [Placement(center_x=200, center_y=200,
                                   rotation_deg=0, seed_index=0, scale=1.0)]
    placement_large = [Placement(center_x=200, center_y=200,
                                   rotation_deg=0, seed_index=0, scale=1.5)]

    result_small = composer.render(bg, stamps, placement_small,
                                     shadow_config=None, feather_px=0.0)
    result_large = composer.render(bg, stamps, placement_large,
                                     shadow_config=None, feather_px=0.0)

    w_small, h_small = result_small.bbox_sizes[0]
    w_large, h_large = result_large.bbox_sizes[0]
    assert w_large > w_small * 1.3
    assert h_large > h_small * 1.3
