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
    existing_occupied: list[tuple[int, int, float]] | None = None,
) -> list[Placement]:
    """Random scatter med rejection sampling — ingen overlap.

    existing_occupied: liste af (cx, cy, radius) for regioner der allerede
    er taget af andre strategier. scatter undgår at placere oven på dem.

    Returnerer en liste af Placements. Kan returnere færre end len(seed_stamps)
    hvis canvas er fyldt og max_attempts er opbrugt.
    """
    canvas_w, canvas_h = canvas_size
    placements: list[Placement] = []
    # occupied er alle radier vi skal undgå — både fra tidligere strategier og fra denne
    occupied: list[tuple[int, int, float]] = list(existing_occupied or [])

    for seed_idx, stamp in enumerate(seed_stamps):
        radius = _bounding_radius(stamp)
        margin = int(radius) + 2

        if margin * 2 >= canvas_w or margin * 2 >= canvas_h:
            continue  # stamp er for stor til canvas

        for _ in range(max_attempts):
            cx = int(rng.integers(margin, canvas_w - margin))
            cy = int(rng.integers(margin, canvas_h - margin))
            rot = float(rng.uniform(0, 360))

            # Tjek overlap mod alle optagne regioner (inkl. andre strategier)
            collision = False
            for ocx, ocy, oradius in occupied:
                dx = cx - ocx
                dy = cy - ocy
                dist = (dx * dx + dy * dy) ** 0.5
                if dist < (radius + oradius + gap_pixels):
                    collision = True
                    break

            if not collision:
                placements.append(Placement(
                    center_x=cx, center_y=cy,
                    rotation_deg=rot, seed_index=seed_idx))
                # Tilføj til occupied så efterfølgende frø undgår dette
                occupied.append((cx, cy, radius))
                break
        # hvis vi ikke fandt plads efter max_attempts: fortsæt med næste frø

    return placements


def dense(
    canvas_size: tuple[int, int],
    seed_stamps: list[SeedStamp],
    max_attempts: int,
    rng: np.random.Generator,
    existing_occupied: list[tuple[int, int, float]] | None = None,
) -> list[Placement]:
    """Tæt pakning: som scatter men med gap_pixels=0 (frø rør hinanden)."""
    return scatter(canvas_size=canvas_size, seed_stamps=seed_stamps,
                   gap_pixels=0, max_attempts=max_attempts, rng=rng,
                   existing_occupied=existing_occupied)


def t_config(
    canvas_size: tuple[int, int],
    seed_stamps: list[SeedStamp],
    rng: np.random.Generator,
    existing_occupied: list[tuple[int, int, float]] | None = None,
    max_attempts: int = 200,
) -> list[Placement]:
    """Sample 2-3 frø placeret i T- eller L-konfiguration.

    Algoritme:
    1. Placer første frø på random rotation (undgår existing_occupied)
    2. Placer andet frø vinkelret på første, så de rør i en kant
    3. Med 50% chance, placer tredje frø der forlænger kæden fra seed 2

    existing_occupied: liste af (cx, cy, radius) for regioner der allerede
    er taget — t_config undgår at placere frø 1 oven på dem.

    Bruges som mindre cluster — composer-driver kalder denne flere gange
    pr. scene for at få mange T-konfigurationer.
    """
    canvas_w, canvas_h = canvas_size
    if len(seed_stamps) < 2:
        return []

    # Vælg op til 3 stamps
    n = min(3, len(seed_stamps))
    placements: list[Placement] = []

    # Frø 1: find en ledig position med rejection sampling
    radius_1 = _bounding_radius(seed_stamps[0])
    margin = int(radius_1 * 1.5) + 10
    if margin * 2 >= canvas_w or margin * 2 >= canvas_h:
        return []

    # Basisregioner vi skal undgå (kopier — vi muterer ikke kalderens liste)
    ext_occupied: list[tuple[int, int, float]] = list(existing_occupied or [])

    # Rejection sampling: forsøg at placere hele clusteren (seed 1 + seed 2)
    # uden at kollidere med ext_occupied. Seed 3 springes over hvis den kolliderer.
    placed_cx1 = placed_cy1 = placed_rot1 = None
    placed_cx2 = placed_cy2 = placed_rot2 = None
    radius_2: float = 0.0

    for _ in range(max_attempts):
        # Kandidat-position for frø 1
        cx1_try = int(rng.integers(margin, canvas_w - margin))
        cy1_try = int(rng.integers(margin, canvas_h - margin))
        rot1_try = float(rng.uniform(0, 360))

        # Tjek frø 1 mod alle ydre optagne regioner
        collision = any(
            (cx1_try - ocx) ** 2 + (cy1_try - ocy) ** 2 < (radius_1 + oradius) ** 2
            for ocx, ocy, oradius in ext_occupied
        )
        if collision:
            continue

        # Beregn frø 2's position så de visuelt rør (bbox-projektion langs rot1)
        # Seed 1 (rot=rot1): lokal x-akse peger langs rot1, så half-extent = w1/2
        # Seed 2 (rot=rot1+90): lokal y-akse peger langs rot1, så half-extent = h2/2
        h1, w1 = seed_stamps[0].rgba.shape[:2]
        h2, w2 = seed_stamps[1].rgba.shape[:2]
        offset_dist = w1 / 2 + h2 / 2  # ingen buffer — bbox-edges rør netop
        # Beholdes til radius_2 reference og collision-tjek
        r2 = _bounding_radius(seed_stamps[1])
        rad = np.deg2rad(rot1_try)
        cx2_try = cx1_try + int(offset_dist * np.cos(rad))
        cy2_try = cy1_try + int(offset_dist * np.sin(rad))

        # Afvis hvis frø 2 er uden for canvas — clip ville bryde touching-invarianten
        if not (margin <= cx2_try < canvas_w - margin
                and margin <= cy2_try < canvas_h - margin):
            continue

        # Tjek frø 2 mod ydre optagne regioner
        collision2 = any(
            (cx2_try - ocx) ** 2 + (cy2_try - ocy) ** 2 < (r2 + oradius) ** 2
            for ocx, ocy, oradius in ext_occupied
        )
        if collision2:
            continue

        # Begge frø kan placeres uden kollision — gem og bryd ud
        placed_cx1, placed_cy1, placed_rot1 = cx1_try, cy1_try, rot1_try
        placed_cx2, placed_cy2 = cx2_try, cy2_try
        placed_rot2 = rot1_try + 90.0 + float(rng.uniform(-15, 15))
        radius_2 = r2
        break

    # Ingen ledig plads fundet — returner tom liste
    if placed_cx1 is None:
        return []

    cx1, cy1, rot1 = placed_cx1, placed_cy1, placed_rot1
    cx2, cy2 = placed_cx2, placed_cy2
    rot2 = placed_rot2

    placements.append(Placement(center_x=cx1, center_y=cy1,
                                  rotation_deg=rot1, seed_index=0))

    # Frø 2: vinkelret, rører frø 1
    if n >= 2:
        placements.append(Placement(center_x=cx2, center_y=cy2,
                                      rotation_deg=rot2, seed_index=1))

    # Frø 3 (50% chance): rør seed 2 i samme retning som seed 1→seed 2 (forlænger kæden)
    if n >= 3 and rng.random() < 0.5:
        radius_3 = _bounding_radius(seed_stamps[2])
        rot3 = rot1 + float(rng.uniform(-15, 15))  # samme orientering som seed 1
        # Forlæng kæden: seed 2's half-extent langs rot1 = h2/2; seed 3 ved rot1 = w3/2
        h3, w3 = seed_stamps[2].rgba.shape[:2]
        offset_dist_3 = h2 / 2 + w3 / 2  # ingen buffer — bbox-edges rør netop
        rad = np.deg2rad(rot1)  # samme retning som seed 1→seed 2
        cx3 = cx2 + int(offset_dist_3 * np.cos(rad))
        cy3 = cy2 + int(offset_dist_3 * np.sin(rad))
        # Tjek seed 3 mod ydre optagne regioner — skip hvis kolliderer
        collision3 = any(
            (cx3 - ocx) ** 2 + (cy3 - ocy) ** 2 < (radius_3 + oradius) ** 2
            for ocx, ocy, oradius in ext_occupied
        )
        # Clip til canvas — hvis tredje frø ikke kan placeres, skip det
        if (margin <= cx3 < canvas_w - margin and margin <= cy3 < canvas_h - margin
                and not collision3):
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
    # Akkumuleret liste af (cx, cy, radius) — opdateres løbende så hver
    # strategi ved hvilke regioner de øvrige strategier allerede har besat
    occupied_so_far: list[tuple[int, int, float]] = []

    # Step 1a: T-config clusters (2-3 frø ad gangen)
    seed_idx = 0
    while n_t_config >= 2 and seed_idx + 1 < n_total:
        cluster_stamps = seed_stamps[seed_idx:seed_idx + 3]
        # Send occupied_so_far så denne cluster undgår tidligere clusters
        sub_placements = t_config(canvas_size, cluster_stamps, rng,
                                  existing_occupied=occupied_so_far,
                                  max_attempts=max_attempts)
        # Remap seed_index fra lokalt (0/1/2) til globalt offset
        for sp in sub_placements:
            sp.seed_index = seed_idx + sp.seed_index
            all_placements.append(sp)
            used_stamps_indices.add(sp.seed_index)
            # Tilføj til occupied så næste cluster og scatter/dense undgår denne
            r = _bounding_radius(seed_stamps[sp.seed_index])
            occupied_so_far.append((sp.center_x, sp.center_y, r))
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

    # Scatter-del: kald scatter med de udvalgte stamps, send occupied_so_far med
    if scatter_idxs:
        scatter_stamps = [s for _, s in scatter_idxs]
        scatter_placements = scatter(canvas_size, scatter_stamps,
                                     gap_pixels, max_attempts, rng,
                                     existing_occupied=occupied_so_far)
        # Remap lokalt seed_index (0..len-1) til globalt index
        for sp in scatter_placements:
            sp.seed_index = scatter_idxs[sp.seed_index][0]
            all_placements.append(sp)
            # Tilføj til occupied så dense undgår scatter-placements også
            r = _bounding_radius(seed_stamps[sp.seed_index])
            occupied_so_far.append((sp.center_x, sp.center_y, r))

    # Dense-del: kald dense med de udvalgte stamps, send opdateret occupied_so_far med
    if dense_idxs:
        dense_stamps = [s for _, s in dense_idxs]
        dense_placements = dense(canvas_size, dense_stamps,
                                 max_attempts, rng,
                                 existing_occupied=occupied_so_far)
        # Remap lokalt seed_index (0..len-1) til globalt index
        for dp in dense_placements:
            dp.seed_index = dense_idxs[dp.seed_index][0]
            all_placements.append(dp)

    return all_placements
