"""Pytest-fixtures til composite-pipeline tests.

Vi laver små test-fixtures (10x10 frø, 200x200 baggrunde) i en tempdir
så testene er hurtige og uafhængige af det rigtige dataset.
"""
import sys
from pathlib import Path

# Tilføj parent-dir til sys.path så testene kan importere moduler som "seed_loader"
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def tiny_seed_png(tmp_path):
    """Lav en 20x30 RGBA-PNG med solid alpha i midten, transparent i kanten."""
    rgba = np.zeros((30, 20, 4), dtype=np.uint8)
    rgba[5:25, 3:17, :3] = 200          # gul-agtig RGB
    rgba[5:25, 3:17, 3] = 255           # solid alpha midt
    path = tmp_path / "tiny_seed.png"
    Image.fromarray(rgba, mode="RGBA").save(path)
    return path


@pytest.fixture
def corrupt_seed_png(tmp_path):
    """Lav en fil der ligner en PNG men er korrupt."""
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nnotapng")
    return path


@pytest.fixture
def fully_transparent_seed_png(tmp_path):
    """RGBA-PNG hvor alle pixels har alpha=0."""
    rgba = np.zeros((20, 20, 4), dtype=np.uint8)
    path = tmp_path / "fully_transparent.png"
    Image.fromarray(rgba, mode="RGBA").save(path)
    return path


@pytest.fixture
def rgb_only_seed_png(tmp_path):
    """PNG uden alpha-kanal (mode RGB) — skal afvises af loader."""
    rgb = np.full((20, 20, 3), 100, dtype=np.uint8)
    path = tmp_path / "rgb_only.png"
    Image.fromarray(rgb, mode="RGB").save(path)
    return path


@pytest.fixture
def tiny_blue_background_png(tmp_path):
    """200x150 PNG hvor midten er blå (HSV ~200°), kanterne er brune."""
    bgr = np.zeros((150, 200, 3), dtype=np.uint8)
    bgr[:, :] = (50, 30, 80)           # brun-agtig BGR (mørk)
    bgr[20:130, 30:170] = (180, 100, 30)  # blå BGR (lyseblå)
    path = tmp_path / "tiny_belt.png"
    Image.fromarray(bgr[..., ::-1], mode="RGB").save(path)  # konverter til RGB
    return path
