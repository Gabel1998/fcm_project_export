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


def test_no_data_leakage_between_train_and_val(tmp_path):
    """Generér en større mængde frø og verificér at train/val ikke deler stamps."""
    # Lav 20 unikke seed-PNG'er — nok til at split kan give meningsfuld disjunktion
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    for i in range(20):
        rgba = np.zeros((40, 30, 4), dtype=np.uint8)
        # Indkod seed-index i RGB-værdier så vi kan identificere hvilke der bruges
        rgba[5:35, 3:27, 0] = i * 10           # unik R-værdi pr. seed
        rgba[5:35, 3:27, 1] = 100
        rgba[5:35, 3:27, 2] = 200
        rgba[5:35, 3:27, 3] = 255
        Image.fromarray(rgba, mode="RGBA").save(seed_dir / f"seed_{i:03d}.png")

    bg_path = tmp_path / "background.png"
    bg = np.zeros((400, 500, 3), dtype=np.uint8)
    bg[:, :] = (50, 30, 80)
    bg[40:360, 50:450] = (30, 100, 180)
    Image.fromarray(bg, mode="RGB").save(bg_path)

    output_dir = tmp_path / "output"
    config = generate_dataset.Config(
        seed_image_dir=seed_dir, background_paths=[bg_path],
        output_dir=output_dir, num_composites=10, train_val_split=0.80,
        random_seed=42, canvas_size=(300, 200),
        seeds_min=2, seeds_max=3,
        gap_pixels=2, max_attempts=100,
    )
    generate_dataset.run(config)

    # Verificér via metadata at train/val pools er disjunkte
    import json
    meta = json.loads((output_dir / "generation_metadata.json").read_text())
    assert meta["num_seeds_train_pool"] == 16, "16 seeds (80% af 20) skal være i train-pool"
    assert meta["num_seeds_val_pool"] == 4, "4 seeds (20% af 20) skal være i val-pool"
    assert meta["num_seeds_train_pool"] + meta["num_seeds_val_pool"] == 20
