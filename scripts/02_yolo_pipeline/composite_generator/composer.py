"""Render én composite-scene: baggrund + frø + ground truth.

Task 5: paste med alpha-blending, MVP.
Task 9: dropshadow, edge-feather og scale-jitter tilføjet.
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


def _scale_and_rotate_stamp(rgba: np.ndarray, scale: float,
                              angle_deg: float) -> np.ndarray:
    """Skalér og rotér RGBA-stamp omkring sit centrum."""
    h, w = rgba.shape[:2]
    # Først skalér via resize
    if scale != 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        rgba = cv2.resize(rgba, (new_w, new_h),
                           interpolation=cv2.INTER_LINEAR)
        h, w = new_h, new_w

    # Så rotér
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, scale=1.0)
    cos_a = abs(matrix[0, 0])
    sin_a = abs(matrix[0, 1])
    new_w_rot = int(h * sin_a + w * cos_a)
    new_h_rot = int(h * cos_a + w * sin_a)
    # Justér translation så midten af det nye canvas rammer samme punkt
    matrix[0, 2] += (new_w_rot - w) / 2
    matrix[1, 2] += (new_h_rot - h) / 2

    # warpAffine med BORDER_CONSTANT (0,0,0,0) så transparency udenfor passer
    return cv2.warpAffine(rgba, matrix, (new_w_rot, new_h_rot),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT,
                           borderValue=(0, 0, 0, 0))


def _apply_shadow(canvas: np.ndarray, alpha_mask: np.ndarray,
                   top_left: tuple[int, int],
                   shadow_config: dict) -> None:
    """Mørkn baggrund hvor skyggen falder. Modificerer canvas in-place."""
    offset_x, offset_y = shadow_config["offset_px"]
    blur_sigma = shadow_config["blur_sigma"]
    opacity = shadow_config["opacity"]

    # Blur alpha-masken som basis for skyggen — skaber en blød skygge-form
    blurred = cv2.GaussianBlur(alpha_mask.astype(np.float32),
                                 ksize=(0, 0), sigmaX=blur_sigma)
    blurred = blurred / 255.0 * opacity   # normalisér til 0-opacity

    canvas_h, canvas_w = canvas.shape[:2]
    sh, sw = blurred.shape
    # Skyggen er forskudt med offset fra frøets top-left hjørne
    x0, y0 = top_left[0] + offset_x, top_left[1] + offset_y
    x1, y1 = x0 + sw, y0 + sh

    # Clip skyggen til canvas-grænser
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(canvas_w, x1), min(canvas_h, y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return

    # Tilsvarende slice i shadow-masken
    sx0 = cx0 - x0
    sy0 = cy0 - y0
    sx1 = sx0 + (cx1 - cx0)
    sy1 = sy0 + (cy1 - cy0)

    # Multiplicér canvas-pixels med (1 - shadow_slice) for at gøre dem mørkere
    shadow_slice = blurred[sy0:sy1, sx0:sx1, np.newaxis]
    canvas[cy0:cy1, cx0:cx1, :] = (
        canvas[cy0:cy1, cx0:cx1, :].astype(np.float32) * (1 - shadow_slice)
    ).clip(0, 255).astype(np.uint8)


def render(
    background: np.ndarray,
    seed_stamps: list[SeedStamp],
    placements: list[Placement],
    shadow_config=None,
    feather_px: float = 0.0,
) -> CompositeResult:
    """Render én composite med alle placements paste'et på baggrund.

    shadow_config: None for ingen skygge, ellers dict med
        'enabled', 'offset_px', 'blur_sigma', 'opacity'
    feather_px: gaussian-blur sigma for alpha-edge softening (0 = hård kant)
    """
    canvas = background.copy()
    canvas_h, canvas_w = canvas.shape[:2]
    centers: list[tuple[int, int]] = []
    bbox_sizes: list[tuple[int, int]] = []

    # Afgør én gang om skygge er aktiv for hele render-kaldet
    apply_shadow = shadow_config is not None and shadow_config.get("enabled", False)

    for p in placements:
        stamp = seed_stamps[p.seed_index]
        # Hent scale fra placement (default 1.0 hvis ikke sat)
        scale = getattr(p, "scale", 1.0)
        rotated = _scale_and_rotate_stamp(stamp.rgba, scale, p.rotation_deg)
        sh, sw = rotated.shape[:2]

        # Top-left coords i canvas (centeret på p.center_x/y)
        x0 = p.center_x - sw // 2
        y0 = p.center_y - sh // 2
        x1 = x0 + sw
        y1 = y0 + sh

        # Skygge først (under frøet) — tegnes direkte på canvas
        if apply_shadow:
            alpha_mask = rotated[:, :, 3]
            _apply_shadow(canvas, alpha_mask, (x0, y0), shadow_config)

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

        # Edge feather: gaussian blur af alpha-kanten for blød overgang.
        # Vi zero-padder alpha-masken inden blur så kanten "ser" nuller udenfor —
        # uden padding ville GaussianBlur give 255 helt ud til kanten på en solid maske.
        if feather_px > 0:
            pad = int(feather_px * 2 + 1)   # bred nok til at blurren når kanten
            full_alpha = rotated[:, :, 3].astype(np.float32)
            padded = cv2.copyMakeBorder(full_alpha, pad, pad, pad, pad,
                                         cv2.BORDER_CONSTANT, value=0)
            blurred_padded = cv2.GaussianBlur(padded, ksize=(0, 0), sigmaX=feather_px)
            # Clip padding væk igen og tag det samme slice som vi ville clippe
            full_alpha_blurred = blurred_padded[pad:pad + rotated.shape[0],
                                                 pad:pad + rotated.shape[1]]
            alpha = full_alpha_blurred[sy0:sy1, sx0:sx1]
        else:
            alpha = rotated[sy0:sy1, sx0:sx1, 3].astype(np.float32)

        # Alpha-blend: canvas = stamp*alpha + canvas*(1-alpha)
        alpha_norm = (alpha / 255.0)[..., np.newaxis]
        bg_slice = canvas[cy0:cy1, cx0:cx1, :].astype(np.float32)
        blended = stamp_rgb.astype(np.float32) * alpha_norm + bg_slice * (1 - alpha_norm)
        canvas[cy0:cy1, cx0:cx1, :] = np.clip(blended, 0, 255).astype(np.uint8)

        # Registrér center og bbox for denne frø
        centers.append((p.center_x, p.center_y))
        bbox_sizes.append((sw, sh))

    return CompositeResult(image=canvas, centers=centers, bbox_sizes=bbox_sizes)
