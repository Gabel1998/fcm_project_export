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


def t_config(
    canvas_size: tuple[int, int],
    seed_stamps: list[SeedStamp],
    rng: np.random.Generator,
) -> list[Placement]:
    """Sample 2-3 frø placeret i T- eller L-konfiguration.

    Algoritme:
    1. Placer første frø på random rotation
    2. Placer andet frø vinkelret på første, så de rør i en kant
    3. Med 50% chance, placer tredje frø oven på krydset

    Bruges som mindre cluster — composer-driver kalder denne flere gange
    pr. scene for at få mange T-konfigurationer.
    """
    canvas_w, canvas_h = canvas_size
    if len(seed_stamps) < 2:
        return []

    # Vælg op til 3 stamps
    n = min(3, len(seed_stamps))
    placements: list[Placement] = []

    # Frø 1: midt på canvas
    radius_1 = _bounding_radius(seed_stamps[0])
    margin = int(radius_1 * 1.5) + 10
    if margin * 2 >= canvas_w or margin * 2 >= canvas_h:
        return []

    cx1 = int(rng.integers(margin, canvas_w - margin))
    cy1 = int(rng.integers(margin, canvas_h - margin))
    rot1 = float(rng.uniform(0, 360))
    placements.append(Placement(center_x=cx1, center_y=cy1,
                                  rotation_deg=rot1, seed_index=0))

    # Frø 2: vinkelret, rører frø 1
    if n >= 2:
        radius_2 = _bounding_radius(seed_stamps[1])
        # Offset i retning vinkelret på frø 1's hovedakse
        rot2 = rot1 + 90.0 + float(rng.uniform(-15, 15))
        offset_dist = radius_1 + radius_2 - 5  # lille overlap så de rør
        rad = np.deg2rad(rot1)
        cx2 = cx1 + int(offset_dist * np.cos(rad))
        cy2 = cy1 + int(offset_dist * np.sin(rad))
        # Clip til canvas
        cx2 = max(margin, min(canvas_w - margin, cx2))
        cy2 = max(margin, min(canvas_h - margin, cy2))
        placements.append(Placement(center_x=cx2, center_y=cy2,
                                      rotation_deg=rot2, seed_index=1))

    # Frø 3 (50% chance): på krydset
    if n >= 3 and rng.random() < 0.5:
        rot3 = float(rng.uniform(0, 360))
        cx3 = (cx1 + cx2) // 2
        cy3 = (cy1 + cy2) // 2
        placements.append(Placement(center_x=cx3, center_y=cy3,
                                      rotation_deg=rot3, seed_index=2))

    return placements


def compute(
    canvas_size: tuple[int, int],
    seed_stamps: list[SeedStamp],
    strategy_mix: dict[str, float],
    gap_pixels: int,
    max_attempts: int,
    rng: np.random.Generator,
) -> list[Placement]:
    """Top-level dispatcher: bland strategy_mix-vægt for hvert frø.

    Algoritme:
    - Først laver vi T-config-clusters: bruger ~strategy_mix['t_config']
      procent af frøene i clusters á 2-3.
    - Resten af frøene placeres med en blanding af scatter og dense
      proportional til deres relative vægt.
    - Alle placements har globalt-unikke seed_index'er.
    """
    n_total = len(seed_stamps)
    if n_total == 0:
        return []

    # Fallback: hvis ingen mix er angivet, bruges ren scatter
    if strategy_mix is None:
        strategy_mix = {"scatter": 1.0, "dense": 0.0, "t_config": 0.0}

    # Step 1: hvor mange frø skal i t-config clusters?
    t_weight = strategy_mix.get("t_config", 0.0)
    n_t_config = int(n_total * t_weight)

    all_placements: list[Placement] = []
    used_stamps_indices = set()

    # Step 1a: T-config clusters (2-3 frø ad gangen)
    seed_idx = 0
    while n_t_config >= 2 and seed_idx + 1 < n_total:
        cluster_stamps = seed_stamps[seed_idx:seed_idx + 3]
        sub_placements = t_config(canvas_size, cluster_stamps, rng)
        # Remap seed_index fra lokalt (0/1/2) til globalt offset
        for sp in sub_placements:
            sp.seed_index = seed_idx + sp.seed_index
            all_placements.append(sp)
            used_stamps_indices.add(sp.seed_index)
        consumed = len(sub_placements)
        # Hvis t_config returnerede 0 placements: skip en stamp for at undgå uendelig løkke
        seed_idx += consumed if consumed > 0 else 1
        n_t_config -= consumed

    # Step 2: scatter vs dense for de resterende frø
    remaining_stamps_with_idx = [
        (i, s) for i, s in enumerate(seed_stamps)
        if i not in used_stamps_indices
    ]

    scatter_weight = strategy_mix.get("scatter", 0.0)
    dense_weight = strategy_mix.get("dense", 0.0)
    total_w = scatter_weight + dense_weight
    if total_w == 0:
        return all_placements

    # Beregn opdeling: de første n_scatter går til scatter, resten til dense
    n_scatter = int(len(remaining_stamps_with_idx) * scatter_weight / total_w)

    scatter_idxs = remaining_stamps_with_idx[:n_scatter]
    dense_idxs = remaining_stamps_with_idx[n_scatter:]

    # Scatter-del: kald scatter med de udvalgte stamps
    if scatter_idxs:
        scatter_stamps = [s for _, s in scatter_idxs]
        scatter_placements = scatter(canvas_size, scatter_stamps,
                                     gap_pixels, max_attempts, rng)
        # Remap lokalt seed_index (0..len-1) til globalt index
        for sp in scatter_placements:
            sp.seed_index = scatter_idxs[sp.seed_index][0]
            all_placements.append(sp)

    # Dense-del: kald dense med de udvalgte stamps
    if dense_idxs:
        dense_stamps = [s for _, s in dense_idxs]
        dense_placements = dense(canvas_size, dense_stamps,
                                 max_attempts, rng)
        # Remap lokalt seed_index (0..len-1) til globalt index
        for dp in dense_placements:
            dp.seed_index = dense_idxs[dp.seed_index][0]
            all_placements.append(dp)

    return all_placements
