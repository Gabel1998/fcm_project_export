"""Driver: binder seed_loader + background_loader + packing + composer sammen.

MVP-version (Task 6):
- Bruger kun scatter-pakning
- Ingen skygge eller edge-feather
- Læser config fra CLI-arg eller dataclass

Realisme-features tilføjes i Task 9-12.
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import numpy as np
import yaml
from PIL import Image

import seed_loader
import background_loader
import packing
import composer


@dataclass
class Config:
    """Subset af config.yaml — kun det MVP-driveren bruger."""
    seed_image_dir: Path
    background_paths: list[Path]
    output_dir: Path
    num_composites: int
    train_val_split: float
    random_seed: int
    canvas_size: tuple[int, int]   # (width, height)
    seeds_min: int
    seeds_max: int
    gap_pixels: int
    max_attempts: int

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            seed_image_dir=Path(data["seed_image_dir"]).expanduser(),
            background_paths=[Path(p).expanduser() for p in data["background_paths"]],
            output_dir=Path(data["output_dir"]).expanduser(),
            num_composites=data["num_composites"],
            train_val_split=data["train_val_split"],
            random_seed=data["random_seed"],
            canvas_size=tuple(data["canvas_size"]),
            seeds_min=data["seeds_per_composite"]["min"],
            seeds_max=data["seeds_per_composite"]["max"],
            gap_pixels=data["packing_gap_pixels"],
            max_attempts=data["max_placement_attempts"],
        )


def _setup_dirs(output_dir: Path) -> None:
    # Opret alle nødvendige mapper til YOLO-struktur
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def _write_label(path: Path, centers, bbox_sizes, canvas_size) -> None:
    """Skriv YOLO-format label: class cx cy w h (alle normaliseret 0-1)."""
    canvas_w, canvas_h = canvas_size
    lines = []
    for (cx, cy), (w, h) in zip(centers, bbox_sizes):
        # Normaliser koordinater til [0, 1] ved at dividere med canvas-størrelsen
        cx_n = cx / canvas_w
        cy_n = cy / canvas_h
        w_n = w / canvas_w
        h_n = h / canvas_h
        # Clip til [0, 1] for at undgå out-of-bounds værdier
        cx_n = max(0.0, min(1.0, cx_n))
        cy_n = max(0.0, min(1.0, cy_n))
        w_n = max(0.001, min(1.0, w_n))
        h_n = max(0.001, min(1.0, h_n))
        lines.append(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _atomic_write_image(path: Path, image: np.ndarray) -> None:
    """Skriv PNG atomic: først .tmp, så rename — undgår halvt-skrevne filer.

    Vi angiver format="PNG" eksplicit fordi PIL bestemmer format ud fra
    filendelsen, og .png.tmp giver endelsen .tmp (ukendt for PIL).
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(image, mode="RGB").save(tmp, format="PNG")
    tmp.rename(path)


def run(config: Config) -> None:
    """Hovedfunktion: generér num_composites composites."""
    _setup_dirs(config.output_dir)

    # Fast seed for reproducerbarhed
    rng = np.random.default_rng(config.random_seed)

    print(f"[1/3] Loader baggrunde fra {len(config.background_paths)} stier")
    bg_pool = background_loader.load(config.background_paths)

    print(f"[2/3] Scanner seed-mappen: {config.seed_image_dir}")
    seed_paths = seed_loader.scan(config.seed_image_dir)
    print(f"      Fandt {len(seed_paths)} frø-PNG'er")
    if len(seed_paths) == 0:
        raise RuntimeError(f"Ingen PNG'er i {config.seed_image_dir}")

    print(f"[3/3] Genererer {config.num_composites} composites")
    start = time.time()
    # Beregn antal trænings-billeder ud fra split-ratio
    n_train = int(config.num_composites * config.train_val_split)

    for i in range(config.num_composites):
        # Tildel split baseret på index: de første n_train er train, resten val
        split = "train" if i < n_train else "val"
        image_path = config.output_dir / "images" / split / f"composite_{i:05d}.png"
        label_path = config.output_dir / "labels" / split / f"composite_{i:05d}.txt"

        # Resume: spring over hvis output allerede er genereret
        if image_path.exists() and label_path.exists():
            continue

        # Sample en tilfældig baggrund og resize den til canvas-størrelsen
        bg = bg_pool.sample(size=config.canvas_size)

        # Vælg et tilfældigt antal frø (mellem seeds_min og seeds_max)
        seed_count = int(rng.integers(config.seeds_min, config.seeds_max + 1))
        # Vælg seed_count unikke frø-indeks fra seed-poolen
        chosen = rng.choice(len(seed_paths), size=seed_count, replace=False)
        stamps = []
        for idx in chosen:
            stamp = seed_loader.load(seed_paths[idx])
            if stamp is not None:
                stamps.append(stamp)

        if not stamps:
            print(f"  Composite {i}: ingen gyldige frø, springer over")
            continue

        # Pak frø på canvas med scatter-strategi (ingen overlap)
        placements = packing.scatter(
            canvas_size=config.canvas_size, seed_stamps=stamps,
            gap_pixels=config.gap_pixels, max_attempts=config.max_attempts,
            rng=rng)

        # Render compositen: baggrund + frø paste'et med alpha-blending
        result = composer.render(bg, stamps, placements,
                                  shadow_config=None, feather_px=0.0)

        # Skriv billede og label atomisk (ingen halvt-skrevne filer)
        _atomic_write_image(image_path, result.image)
        _write_label(label_path, result.centers, result.bbox_sizes,
                      config.canvas_size)

        # Statusopdatering hvert 5. billede
        if (i + 1) % 5 == 0 or (i + 1) == config.num_composites:
            elapsed = time.time() - start
            avg_seeds = len(result.centers)
            print(f"  {i+1}/{config.num_composites}  ({elapsed:.0f}s,  "
                  f"sidste scene: {avg_seeds} frø)")

    # Skriv data.yaml til YOLOv11-træning
    data_yaml = {
        "path": str(config.output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": ["seed"],
        "nc": 1,
    }
    (config.output_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml))

    # Gem metadata om genererings-kørslen (nyttigt til reproducerbarhed)
    metadata = {
        "config": {k: (str(v) if isinstance(v, Path) else
                        [str(p) for p in v] if isinstance(v, list) else v)
                    for k, v in asdict(config).items()},
        "num_seeds_in_pool": len(seed_paths),
        "elapsed_seconds": time.time() - start,
    }
    (config.output_dir / "generation_metadata.json").write_text(
        json.dumps(metadata, indent=2))

    print(f"\nFærdig. Output: {config.output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                         default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--num-composites", type=int, default=None,
                         help="Override num_composites fra config")
    parser.add_argument("--output-dir", default=None,
                         help="Override output_dir fra config")
    args = parser.parse_args()

    config = Config.from_yaml(Path(args.config))
    if args.num_composites is not None:
        config.num_composites = args.num_composites
    if args.output_dir is not None:
        config.output_dir = Path(args.output_dir).expanduser()

    run(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
