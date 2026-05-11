"""Render én composite-scene: baggrund + frø + ground truth.

MVP-version (Task 5): paste med alpha-blending, ingen skygge, ingen feather.
Task 9 tilføjer dropshadow og edge-feather.
"""
from dataclasses import dataclass
import numpy as np
import cv2

from seed_loader import SeedStamp
from packing import Placement


@dataclass
class CompositeResult:
    """Output fra composer.render()."""
    image: np.ndarray                          # (H, W, 3) uint8 RGB
    centers: list[tuple[int, int]]             # (cx, cy) pr. frø
    bbox_sizes: list[tuple[int, int]]          # (w, h) pr. frø, akse-aligneret efter rotation


def _rotate_stamp(rgba: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotér RGBA-stamp omkring sit centrum, uden at klippe indhold væk.

    OpenCV's getRotationMatrix2D roterer omkring et givet punkt.
    Vi beregner den nye boundingbox så hele det roterede stamp passer ind.
    """
    h, w = rgba.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, scale=1.0)

    # Beregn ny bounding box så hele det roterede stamp passer ind
    # Diagonalen efter rotation = h*|sin(a)| + w*|cos(a)|  for begge aksler
    cos_a = abs(matrix[0, 0])
    sin_a = abs(matrix[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # Justér translation så midten af det nye canvas rammer samme punkt
    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2

    # warpAffine med BORDER_CONSTANT (0,0,0,0) så transparency udenfor passer
    return cv2.warpAffine(rgba, matrix, (new_w, new_h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(0, 0, 0, 0))


def render(
    background: np.ndarray,
    seed_stamps: list[SeedStamp],
    placements: list[Placement],
    shadow_config=None,    # bruges fra Task 9
    feather_px: float = 0.0,  # bruges fra Task 9
) -> CompositeResult:
    """Render én composite med alle placements paste'et på baggrund.

    Algorithm:
    1. For hver placement:
       - Rotér stampen (RGBA)
       - Beregn position på canvas (center_x/y som anchor)
       - Clip mod canvas-grænser
       - Alpha-blend: canvas = stamp*alpha + canvas*(1-alpha)
    2. Registrér centers og bbox_sizes for alle synlige frø

    Args:
        background: (H, W, 3) uint8 RGB
        seed_stamps: liste af SeedStamp (RGBA)
        placements: liste af Placement
        shadow_config: ignoreres i MVP (bruges Task 9)
        feather_px: ignoreres i MVP (bruges Task 9)

    Returns:
        CompositeResult med render-billede og ground truth metadata
    """
    canvas = background.copy()
    canvas_h, canvas_w = canvas.shape[:2]
    centers: list[tuple[int, int]] = []
    bbox_sizes: list[tuple[int, int]] = []

    for p in placements:
        stamp = seed_stamps[p.seed_index]
        rotated = _rotate_stamp(stamp.rgba, p.rotation_deg)
        sh, sw = rotated.shape[:2]

        # Top-left coords i canvas (centeret på p.center_x/y)
        x0 = p.center_x - sw // 2
        y0 = p.center_y - sh // 2
        x1 = x0 + sw
        y1 = y0 + sh

        # Clipping mod canvas-grænser
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(canvas_w, x1), min(canvas_h, y1)
        if cx0 >= cx1 or cy0 >= cy1:
            continue  # helt uden for canvas

        # Tilsvarende slice i stamp (efter clipping)
        sx0 = cx0 - x0
        sy0 = cy0 - y0
        sx1 = sx0 + (cx1 - cx0)
        sy1 = sy0 + (cy1 - cy0)

        stamp_rgb = rotated[sy0:sy1, sx0:sx1, :3]
        stamp_alpha = rotated[sy0:sy1, sx0:sx1, 3:4].astype(np.float32) / 255.0

        # Alpha-blend: canvas = stamp*alpha + canvas*(1-alpha)
        bg_slice = canvas[cy0:cy1, cx0:cx1, :].astype(np.float32)
        blended = stamp_rgb.astype(np.float32) * stamp_alpha + bg_slice * (1 - stamp_alpha)
        canvas[cy0:cy1, cx0:cx1, :] = np.clip(blended, 0, 255).astype(np.uint8)

        # Registrér center og bbox for denne frø
        centers.append((p.center_x, p.center_y))
        bbox_sizes.append((sw, sh))

    return CompositeResult(image=canvas, centers=centers, bbox_sizes=bbox_sizes)
