"""Integration-tests for generate_dataset.py."""
import numpy as np
import pytest
import yaml
from pathlib import Path
from PIL import Image

import generate_dataset


@pytest.fixture
def mini_corpus(tmp_path):
    """Lav et mini-corpus: 10 frø-PNG'er + 1 baggrund."""
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    for i in range(10):
        rgba = np.zeros((40, 30, 4), dtype=np.uint8)
        rgba[5:35, 3:27, :3] = 200
        rgba[5:35, 3:27, 3] = 255
        Image.fromarray(rgba, mode="RGBA").save(seed_dir / f"seed_{i:03d}.png")

    bg_path = tmp_path / "background.png"
    bg = np.zeros((400, 500, 3), dtype=np.uint8)
    bg[:, :] = (50, 30, 80)         # brun
    bg[40:360, 50:450] = (30, 100, 180)  # blå (RGB)
    Image.fromarray(bg, mode="RGB").save(bg_path)

    return seed_dir, [bg_path]


def test_generates_correct_file_structure(mini_corpus, tmp_path):
    """Efter run findes alle forventede mapper og filer."""
    seed_dir, bg_paths = mini_corpus
    output_dir = tmp_path / "output"

    config = generate_dataset.Config(
        seed_image_dir=seed_dir, background_paths=bg_paths,
        output_dir=output_dir, num_composites=4, train_val_split=0.75,
        random_seed=42, canvas_size=(300, 200),
        seeds_min=3, seeds_max=5,
        gap_pixels=2, max_attempts=100,
    )
    generate_dataset.run(config)

    assert (output_dir / "images" / "train").is_dir()
    assert (output_dir / "images" / "val").is_dir()
    assert (output_dir / "labels" / "train").is_dir()
    assert (output_dir / "labels" / "val").is_dir()
    assert (output_dir / "data.yaml").is_file()
    assert (output_dir / "generation_metadata.json").is_file()

    # 4 composites med 75% split = 3 train + 1 val
    assert len(list((output_dir / "images" / "train").glob("*.png"))) == 3
    assert len(list((output_dir / "images" / "val").glob("*.png"))) == 1


def test_labels_have_valid_yolo_format(mini_corpus, tmp_path):
    """Hver label-linje har 5 floats i [0, 1]."""
    seed_dir, bg_paths = mini_corpus
    output_dir = tmp_path / "output"
    config = generate_dataset.Config(
        seed_image_dir=seed_dir, background_paths=bg_paths,
        output_dir=output_dir, num_composites=2, train_val_split=0.5,
        random_seed=42, canvas_size=(300, 200),
        seeds_min=3, seeds_max=5,
        gap_pixels=2, max_attempts=100,
    )
    generate_dataset.run(config)

    for label_file in (output_dir / "labels" / "train").glob("*.txt"):
        for line in label_file.read_text().strip().splitlines():
            parts = line.split()
            assert len(parts) == 5
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            assert cls == 0
            assert 0 <= cx <= 1, f"cx ud af range i {label_file}: {cx}"
            assert 0 <= cy <= 1
            assert 0 < w <= 1
            assert 0 < h <= 1


def test_data_yaml_has_correct_structure(mini_corpus, tmp_path):
    """data.yaml har de felter YOLOv11 forventer."""
    seed_dir, bg_paths = mini_corpus
    output_dir = tmp_path / "output"
    config = generate_dataset.Config(
        seed_image_dir=seed_dir, background_paths=bg_paths,
        output_dir=output_dir, num_composites=2, train_val_split=0.5,
        random_seed=42, canvas_size=(300, 200),
        seeds_min=3, seeds_max=5,
        gap_pixels=2, max_attempts=100,
    )
    generate_dataset.run(config)

    data_yaml = yaml.safe_load((output_dir / "data.yaml").read_text())
    assert "path" in data_yaml
    assert data_yaml["train"] == "images/train"
    assert data_yaml["val"] == "images/val"
    assert data_yaml["names"] == ["seed"]
    assert data_yaml["nc"] == 1
