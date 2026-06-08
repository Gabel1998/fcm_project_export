"""Delt kerne til frø-morfologi — bruges af både demoen og script 07.

Tre rene funktioner der oversætter YOLO-detektioner til en blob-analyse
(segmentering → instans-split → center-of-mass + orientering), så ÉT
detektions-skridt kan levere det samme som Videometers flersteps-pipeline.

Alle funktioner forventer billeder i RGB (som PIL/numpy leverer dem).
"""
import cv2
import numpy as np


def segment_foreground(image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Adskil frø-pixels fra den blå plade og den sorte kant via HSV.

    Returnerer (seed_mask, plate_bbox):
      - seed_mask: bool-array [H, W], True hvor der er frø (ikke plade, ikke kant)
      - plate_bbox: (x0, y0, x1, y1) bounding box om den blå plade-zone
    """
    # HSV adskiller farve (H) fra lysstyrke (V) — robust til at skelne blå plade
    # fra gule/hvide frø uanset små lys-variationer i diamant-teksturen.
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Blå plade: OpenCV-hue ~95-125, rimelig mættet og lys
    blue = (hue >= 90) & (hue <= 125) & (sat >= 60) & (val >= 50)
    # Sort kant øverst/nederst: meget lav lysstyrke
    black = val < 45
    # Frø = alt der hverken er blå plade eller sort kant
    seed = ~blue & ~black

    # Plade-region = bounding box om den blå zone (det Videometer ser i produktion,
    # når den brune/sorte kant er skåret væk). Vi lukker huller med en morfologisk
    # close så diamant-teksturen tæller som én sammenhængende blå zone.
    blue_u8 = (blue.astype(np.uint8)) * 255
    blue_u8 = cv2.morphologyEx(blue_u8, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    ys, xs = np.where(blue_u8 > 0)
    if len(xs) > 0:
        x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
    else:
        x0, y0, x1, y1 = 0, 0, image_rgb.shape[1] - 1, image_rgb.shape[0] - 1

    # Begræns frø-masken til plade-regionen (ignorér alt udenfor pladen)
    region = np.zeros_like(seed)
    region[y0:y1 + 1, x0:x1 + 1] = True
    seed = seed & region

    # Fjern små støj-pletter (morfologisk åbning fjerner enkelt-pixel-prikker)
    seed_u8 = (seed.astype(np.uint8)) * 255
    seed_u8 = cv2.morphologyEx(seed_u8, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return seed_u8 > 0, (x0, y0, x1, y1)


def assign_pixels_to_seeds(seed_mask: np.ndarray, centers: np.ndarray,
                           max_dist: float) -> np.ndarray:
    """Tildel hver frø-pixel til det nærmeste detektions-center (marker-styret).

    Det er HER de rørende frø splittes: detektoren har givet ét center pr. frø,
    og vi deler frø-masken op efter "hvilket center er tættest" — svarende til en
    afstands-watershed med detektions-centrene som markører.

    Returnerer et label-billede [H, W] (int): 0 = ingen frø, 1..N = frø-id.
    """
    if len(centers) == 0:
        return np.zeros(seed_mask.shape, dtype=np.int32)

    ys, xs = np.where(seed_mask)
    coords_x = xs.astype(np.float32)
    coords_y = ys.astype(np.float32)

    # Løb over centrene og hold løbende min-afstand + argmin. Det er memory-let
    # (vi laver ikke en kæmpe M×N-matrix) og hurtigt nok til ~40 centre.
    best_id = np.full(len(xs), -1, dtype=np.int32)
    best_d2 = np.full(len(xs), np.inf, dtype=np.float32)
    for i, (cx, cy) in enumerate(centers):
        d2 = (coords_x - cx) ** 2 + (coords_y - cy) ** 2  # kvadreret afstand
        closer = d2 < best_d2
        best_d2[closer] = d2[closer]
        best_id[closer] = i

    # Drop pixels der ligger længere end max_dist fra ethvert center (plade-støj)
    best_id[best_d2 > max_dist ** 2] = -1

    labels = np.zeros(seed_mask.shape, dtype=np.int32)
    labels[ys, xs] = best_id + 1  # +1 så 0 forbliver "ingen frø"
    return labels


def measure_seeds(labels: np.ndarray, n_seeds: int, min_area: int = 80) -> list[dict]:
    """Beregn center-of-mass, areal og orientering pr. frø ud fra dets maske.

    - center-of-mass = gennemsnit af maske-pixlernes koordinater (præcis det
      Videometers blob-analyse rapporterer som "X/Y Center mass").
    - orientering = hovedaksen via 2.-ordens centrale momenter (meningsfuld for
      de aflange majskerner; udefineret for runde frø).
    """
    measurements = []
    for sid in range(1, n_seeds + 1):
        mask = labels == sid
        area = int(mask.sum())
        if area < min_area:
            continue  # for lille klat = støj eller fragment, springes over

        ys, xs = np.where(mask)
        com_x = float(xs.mean())
        com_y = float(ys.mean())

        # Centrale momenter → orientering. mu20/mu02/mu11 beskriver hvordan
        # massen er fordelt; vinklen på hovedaksen er 0.5*atan2(2*mu11, mu20-mu02).
        dx = xs - com_x
        dy = ys - com_y
        mu20 = float((dx * dx).mean())
        mu02 = float((dy * dy).mean())
        mu11 = float((dx * dy).mean())
        angle_rad = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)

        # Aksens halve længde (til tegning): ~2 std langs hovedaksen
        axis_half_len = np.sqrt(max(mu20, mu02)) * 2.0

        measurements.append({
            "id": sid,
            "com_x": com_x,
            "com_y": com_y,
            "area_px": area,
            "angle_deg": float(np.degrees(angle_rad)),
            "axis_half_len": float(axis_half_len),
        })
    return measurements


def analyze(image_rgb: np.ndarray, bbox_xyxy: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """Bekvem wrapper: segmentér → tildel pixels → mål, ud fra detektions-bokse.

    Returnerer (labels, measurements). bbox_xyxy er (N, 4) i pixel-koordinater.
    """
    if len(bbox_xyxy) == 0:
        return np.zeros(image_rgb.shape[:2], dtype=np.int32), []

    centers = np.stack([(bbox_xyxy[:, 0] + bbox_xyxy[:, 2]) / 2,
                        (bbox_xyxy[:, 1] + bbox_xyxy[:, 3]) / 2], axis=1)
    # Typisk frø-størrelse → fornuftig max-afstand for pixel-tildelingen
    box_diag = np.sqrt((bbox_xyxy[:, 2] - bbox_xyxy[:, 0]) ** 2
                       + (bbox_xyxy[:, 3] - bbox_xyxy[:, 1]) ** 2)
    max_dist = float(np.median(box_diag)) if len(box_diag) else 200.0

    seed_mask, _ = segment_foreground(image_rgb)
    labels = assign_pixels_to_seeds(seed_mask, centers, max_dist=max_dist)
    measurements = measure_seeds(labels, n_seeds=len(bbox_xyxy))
    return labels, measurements
