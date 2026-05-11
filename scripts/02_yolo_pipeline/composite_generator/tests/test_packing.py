"""Tests for packing.py."""
import numpy as np
import pytest

import packing
from seed_loader import SeedStamp


def _make_stamp(w: int, h: int, idx: int = 0) -> SeedStamp:
    """Hjælper: lav en SeedStamp af given størrelse."""
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255  # solid alpha
    return SeedStamp(rgba=rgba, source_id=f"seed_{idx}",
                     bbox_after_crop=(0, 0, w, h))


def test_scatter_returns_correct_count():
    """scatter med plads til alle returnerer alle placements."""
    stamps = [_make_stamp(30, 30, i) for i in range(5)]
    placements = packing.scatter(
        canvas_size=(500, 500), seed_stamps=stamps,
        gap_pixels=4, max_attempts=200, rng=np.random.default_rng(0))
    assert len(placements) == 5


def test_scatter_positions_within_canvas():
    """Alle placements har center inden for canvas."""
    stamps = [_make_stamp(20, 20, i) for i in range(10)]
    placements = packing.scatter(
        canvas_size=(300, 300), seed_stamps=stamps,
        gap_pixels=4, max_attempts=200, rng=np.random.default_rng(0))
    for p in placements:
        assert 0 <= p.center_x < 300
        assert 0 <= p.center_y < 300


def test_scatter_no_overlap():
    """Ingen to placements overlapper (afstand > sum af radier + gap)."""
    stamps = [_make_stamp(30, 30, i) for i in range(8)]
    placements = packing.scatter(
        canvas_size=(400, 400), seed_stamps=stamps,
        gap_pixels=4, max_attempts=200, rng=np.random.default_rng(0))
    for i, p1 in enumerate(placements):
        for p2 in placements[i+1:]:
            dx = p1.center_x - p2.center_x
            dy = p1.center_y - p2.center_y
            dist = (dx*dx + dy*dy) ** 0.5
            # Radier er ~15 hver, gap=4 → forventet min afstand ~34
            assert dist >= 30, f"For tæt: {dist:.1f}"


def test_scatter_accepts_fewer_when_canvas_full():
    """Hvis canvas er for fyldt, accepter færre placements."""
    stamps = [_make_stamp(80, 80, i) for i in range(20)]
    placements = packing.scatter(
        canvas_size=(200, 200), seed_stamps=stamps,
        gap_pixels=4, max_attempts=50, rng=np.random.default_rng(0))
    assert len(placements) < 20  # ikke alle passer


def test_scatter_rotation_in_range():
    """Alle rotationer er i [0, 360)."""
    stamps = [_make_stamp(25, 25, i) for i in range(5)]
    placements = packing.scatter(
        canvas_size=(400, 400), seed_stamps=stamps,
        gap_pixels=4, max_attempts=200, rng=np.random.default_rng(0))
    for p in placements:
        assert 0 <= p.rotation_deg < 360


def test_scatter_deterministic_with_seed():
    """Samme RNG-seed giver samme placements."""
    stamps = [_make_stamp(25, 25, i) for i in range(5)]
    p1 = packing.scatter(canvas_size=(400, 400), seed_stamps=stamps,
                          gap_pixels=4, max_attempts=200,
                          rng=np.random.default_rng(42))
    p2 = packing.scatter(canvas_size=(400, 400), seed_stamps=stamps,
                          gap_pixels=4, max_attempts=200,
                          rng=np.random.default_rng(42))
    assert [(p.center_x, p.center_y) for p in p1] == \
           [(p.center_x, p.center_y) for p in p2]


def test_dense_packs_tighter_than_scatter():
    """dense-strategi placerer frø tættere end scatter med samme antal."""
    stamps = [_make_stamp(25, 25, i) for i in range(15)]

    p_scatter = packing.scatter(
        canvas_size=(400, 400), seed_stamps=stamps,
        gap_pixels=10, max_attempts=200,
        rng=np.random.default_rng(0))
    p_dense = packing.dense(
        canvas_size=(400, 400), seed_stamps=stamps,
        max_attempts=200, rng=np.random.default_rng(0))

    # Gennemsnitlig minimum-afstand mellem nabofrø
    def avg_min_dist(placements):
        if len(placements) < 2:
            return float('inf')
        dists = []
        for i, p1 in enumerate(placements):
            min_d = min(
                ((p1.center_x - p2.center_x) ** 2 +
                 (p1.center_y - p2.center_y) ** 2) ** 0.5
                for j, p2 in enumerate(placements) if i != j)
            dists.append(min_d)
        return sum(dists) / len(dists)

    assert avg_min_dist(p_dense) < avg_min_dist(p_scatter)


def test_t_config_creates_touching_seeds():
    """t_config returnerer en cluster af 2-3 frø der rør hinanden."""
    stamps = [_make_stamp(30, 30, i) for i in range(3)]

    placements = packing.t_config(
        canvas_size=(400, 400), seed_stamps=stamps,
        rng=np.random.default_rng(0))

    assert 2 <= len(placements) <= 3

    # Mindst ét par skal være tæt på hinanden (distance < sum af radier + lille gap)
    if len(placements) >= 2:
        radius = 0.5 * (30 * 30 + 30 * 30) ** 0.5  # ~21
        min_dist = min(
            ((p1.center_x - p2.center_x) ** 2 +
             (p1.center_y - p2.center_y) ** 2) ** 0.5
            for i, p1 in enumerate(placements)
            for p2 in placements[i+1:])
        # Frø rør hinanden = distance < 2*radius
        assert min_dist < 2 * radius + 5


def test_t_config_perpendicular_orientation():
    """For ekstra realisme: andet frø står ~vinkelret på første."""
    stamps = [_make_stamp(40, 15, i) for i in range(3)]  # aflange majs

    placements = packing.t_config(
        canvas_size=(400, 400), seed_stamps=stamps,
        rng=np.random.default_rng(0))

    if len(placements) >= 2:
        rot_diff = abs(placements[0].rotation_deg - placements[1].rotation_deg) % 180
        # Vinkelret = 90° ± 30° tolerance
        assert 60 < rot_diff < 120, f"Ikke vinkelret: {rot_diff}°"
