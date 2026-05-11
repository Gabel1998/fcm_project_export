"""Sanity-check af genereret composite-dataset.

Auto-mode: kaldes af generate_dataset.py som sidste step.
Standalone:  python visualize.py --dataset data/composite_v1 --n 8

Output: preview.png (grid med composites + grønne GT-markører).
"""
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _load_labels(label_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    """Læs YOLO-labels, returnér array af (cx, cy) i pixel-koordinater."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    w, h = image_size
    centers = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        cx = float(parts[1]) * w
        cy = float(parts[2]) * h
        centers.append((cx, cy))
    return np.array(centers, dtype=np.float32)


def make_preview(dataset_dir: Path, n_samples: int = 8,
                  output: Path = None, split: str = "train") -> Path:
    """Lav preview.png med n_samples random composites + GT-overlay."""
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split

    all_images = sorted(images_dir.glob("*.png"))
    if not all_images:
        raise FileNotFoundError(f"Ingen billeder i {images_dir}")

    n_actual = min(n_samples, len(all_images))
    rng = np.random.default_rng(0)
    chosen_idx = rng.choice(len(all_images), size=n_actual, replace=False)
    chosen = [all_images[i] for i in chosen_idx]

    cols = 4
    rows = (n_actual + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_2d(axes).flatten()

    for ax_idx, img_path in enumerate(chosen):
        img = np.array(Image.open(img_path))
        label_path = labels_dir / (img_path.stem + ".txt")
        centers = _load_labels(label_path, image_size=img.shape[:2][::-1])

        ax = axes[ax_idx]
        ax.imshow(img)
        if len(centers) > 0:
            ax.scatter(centers[:, 0], centers[:, 1],
                        s=15, c="lime", marker="o",
                        edgecolors="red", linewidths=0.5)
        ax.set_title(f"{img_path.stem}  ({len(centers)} frø)", fontsize=8)
        ax.axis("off")

    # Skjul ubrugte axes
    for j in range(ax_idx + 1, len(axes)):
        axes[j].axis("off")

    if output is None:
        output = dataset_dir / "preview.png"
    plt.tight_layout()
    plt.savefig(output, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = make_preview(args.dataset, args.n, args.output, args.split)
    print(f"Preview gemt: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
