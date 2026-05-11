"""Tests for seed_loader.py."""
import numpy as np
from pathlib import Path
import pytest

import seed_loader


def test_load_returns_seed_stamp(tiny_seed_png):
    """Gyldig RGBA-PNG returnerer en SeedStamp."""
    stamp = seed_loader.load(tiny_seed_png)
    assert stamp is not None
    assert stamp.rgba.shape[2] == 4  # RGBA
    assert stamp.rgba.dtype == np.uint8
    assert stamp.source_id == "tiny_seed"


def test_load_crops_transparent_margin(tiny_seed_png):
    """Transparent margin skal være fjernet — output har solid alpha tæt på kanten."""
    stamp = seed_loader.load(tiny_seed_png)
    # Top-row alpha skal være > 0 (vi har croppet til indhold)
    assert stamp.rgba[0, :, 3].max() > 0
    # Output skal være mindre end input (20x30 → cropped)
    assert stamp.rgba.shape[0] < 30 or stamp.rgba.shape[1] < 20


def test_load_rejects_non_rgba(rgb_only_seed_png):
    """PNG uden alpha-kanal returnerer None."""
    assert seed_loader.load(rgb_only_seed_png) is None


def test_load_rejects_fully_transparent(fully_transparent_seed_png):
    """PNG med kun transparente pixels returnerer None."""
    assert seed_loader.load(fully_transparent_seed_png) is None


def test_load_handles_corrupt(corrupt_seed_png):
    """Korrupt PNG returnerer None uden at crashe."""
    assert seed_loader.load(corrupt_seed_png) is None


def test_scan_finds_all_pngs(tmp_path, tiny_seed_png):
    """scan() finder alle PNG'er i en mappe."""
    # tiny_seed_png ligger allerede i tmp_path
    paths = seed_loader.scan(tmp_path)
    assert len(paths) >= 1
    assert all(p.suffix == ".png" for p in paths)
