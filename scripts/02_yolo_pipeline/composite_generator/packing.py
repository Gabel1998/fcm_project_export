"""Pakkings-strategier til composite-pipeline.

Tre strategier blandes når vi bygger en scene:
- scatter:  random placement med rejection sampling (ingen overlap)
- dense:    som scatter men med nul-gap (frø rør hinanden)  [Task 10]
- t_config: eksplicit T/L-konstellationer                    [Task 11]

I MVP (Task 4) implementeres kun scatter.
"""
from dataclasses import dataclass
import numpy as np

from seed_loader import SeedStamp


@dataclass
class Placement:
    """Hvor og hvordan et frø skal placeres på canvas."""
    center_x: int
    center_y: int
    rotation_deg: float
    seed_index: int      # index ind i seed_stamps-listen
    scale: float = 1.0   # bruges fra Task 9 til scale-jitter


def _bounding_radius(stamp: SeedStamp) -> float:
    """Konservativ radius som halv-diagonalen af stampen efter alle rotationer."""
    h, w = stamp.rgba.shape[:2]
    return 0.5 * (h * h + w * w) ** 0.5


def scatter(
    canvas_size: tuple[int, int],
    seed_stamps: list[SeedStamp],
    gap_pixels: int,
    max_attempts: int,
    rng: np.random.Generator,
) -> list[Placement]:
    """Random scatter med rejection sampling — ingen overlap.

    Returnerer en liste af Placements. Kan returnere færre end len(seed_stamps)
    hvis canvas er fyldt og max_attempts er opbrugt.
    """
    canvas_w, canvas_h = canvas_size
    placements: list[Placement] = []
    radii: list[float] = []

    for seed_idx, stamp in enumerate(seed_stamps):
        radius = _bounding_radius(stamp)
        margin = int(radius) + 2

        if margin * 2 >= canvas_w or margin * 2 >= canvas_h:
            continue  # stamp er for stor til canvas

        for _ in range(max_attempts):
            cx = int(rng.integers(margin, canvas_w - margin))
            cy = int(rng.integers(margin, canvas_h - margin))
            rot = float(rng.uniform(0, 360))

            # Tjek overlap med tidligere placements
            collision = False
            for j, prev in enumerate(placements):
                dx = cx - prev.center_x
                dy = cy - prev.center_y
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < (radius + radii[j] + gap_pixels):
                    collision = True
                    break

            if not collision:
                placements.append(Placement(
                    center_x=cx, center_y=cy,
                    rotation_deg=rot, seed_index=seed_idx))
                radii.append(radius)
                break
        # hvis vi ikke fandt plads efter max_attempts: fortsæt med næste frø

    return placements


def dense(
    canvas_size: tuple[int, int],
    seed_stamps: list[SeedStamp],
    max_attempts: int,
    rng: np.random.Generator,
) -> list[Placement]:
    """Tæt pakning: som scatter men med gap_pixels=0 (frø rør hinanden).

    Genbruges scatter med gap_pixels=0 — det er den enkleste DRY-måde.
    """
    return scatter(canvas_size=canvas_size, seed_stamps=seed_stamps,
                   gap_pixels=0, max_attempts=max_attempts, rng=rng)
