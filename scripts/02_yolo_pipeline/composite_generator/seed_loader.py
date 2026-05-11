"""Læs og preprocess enkelt-frø PNG'er.

Hver PNG er en alpha-stanset majs på transparent baggrund.
Vi cropper transparent margin væk så vi har en kompakt "stamp".
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError


@dataclass
class SeedStamp:
    """Et preprocessed frø klar til at blive paste'et på en baggrund."""
    rgba: np.ndarray              # (H, W, 4) uint8 — RGBA efter crop
    source_id: str                # filename uden .png — til debugging
    bbox_after_crop: tuple        # (x, y, w, h) i originalen — til metadata


def scan(seed_dir: Path) -> list[Path]:
    """Find alle .png-filer i seed_dir (ikke-rekursivt)."""
    return sorted(Path(seed_dir).glob("*.png"))


def load(path: Path) -> SeedStamp | None:
    """Load én PNG, valider RGBA, crop til ikke-transparent indhold.

    Returnerer None hvis filen:
    - er korrupt eller ikke en PNG
    - ikke har alpha-kanal (mode != RGBA)
    - kun har transparente pixels (intet at vise)
    """
    try:
        img = Image.open(path)
    except (UnidentifiedImageError, OSError):
        return None

    if img.mode != "RGBA":
        return None

    rgba = np.array(img, dtype=np.uint8)
    alpha = rgba[:, :, 3]

    # Find bounding box af ikke-transparente pixels
    nonzero_rows = np.any(alpha > 0, axis=1)
    nonzero_cols = np.any(alpha > 0, axis=0)

    if not nonzero_rows.any():
        return None  # fuldt transparent

    y_min, y_max = np.where(nonzero_rows)[0][[0, -1]]
    x_min, x_max = np.where(nonzero_cols)[0][[0, -1]]

    cropped = rgba[y_min:y_max + 1, x_min:x_max + 1, :]

    return SeedStamp(
        rgba=cropped,
        source_id=path.stem,
        bbox_after_crop=(int(x_min), int(y_min),
                          int(x_max - x_min + 1), int(y_max - y_min + 1)),
    )
