"""Lag-baseret overlay-render til demoen.

Én ren funktion der stabler de valgte filter-lag oven på basis-billedet, så
brugeren kan tænde/slukke hvert lag uafhængigt. Rækkefølgen er fast (masker
nederst, stregtegning øverst) så markører altid er synlige.

Holdt som ren funktion (numpy ind, numpy ud) så den kan testes uden Streamlit.
"""
import cv2
import numpy as np


def _scaled_sizes(image_shape: tuple[int, int]) -> dict:
    """Skalér linje-tykkelser og markør-størrelser til billed-opløsningen.

    En 4096-px plade og et 640-px upload skal se ens ud relativt.
    """
    short_side = min(image_shape[0], image_shape[1])
    return {
        "line": max(2, int(short_side / 500)),
        "marker": max(4, int(short_side / 220)),
        "cross": max(8, int(short_side / 110)),
        "font": max(0.4, short_side / 1600),
    }


def render_overlay(
    image_rgb: np.ndarray,
    bbox_xyxy: np.ndarray,
    confs: np.ndarray,
    *,
    show_masks: bool = False,
    show_boxes: bool = True,
    show_bbox_center: bool = False,
    show_com: bool = False,
    show_orientation: bool = False,
    show_confidence: bool = False,
    labels: np.ndarray | None = None,
    measurements: list[dict] | None = None,
) -> np.ndarray:
    """Stabl de valgte lag oven på billedet og returnér resultatet (RGB).

    Lag-rækkefølge nedefra: masker → boxes → bbox-center → center-of-mass(+)
    → orienterings-pil → confidence-tal.

    labels/measurements kræves kun hvis masks/com/orientation er slået til
    (de kommer fra seed_morphology.analyze).
    """
    canvas = image_rgb.copy()
    sizes = _scaled_sizes(image_rgb.shape[:2])

    # --- LAG 1: instans-masker (nederst, gennemsigtig farve) ---
    if show_masks and labels is not None and labels.max() > 0:
        rng = np.random.default_rng(0)  # deterministiske farver pr. id
        n = labels.max()
        colors = rng.integers(60, 256, size=(n + 1, 3), dtype=np.uint8)
        colors[0] = (0, 0, 0)
        color_layer = colors[labels]
        mask_any = labels > 0
        canvas[mask_any] = (0.5 * canvas[mask_any]
                            + 0.5 * color_layer[mask_any]).astype(np.uint8)

    # --- LAG 2: bounding boxes ---
    if show_boxes:
        for (x1, y1, x2, y2) in bbox_xyxy:
            cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                          (255, 60, 60), sizes["line"])

    # --- LAG 3: center-markør (bbox-midte) — lime prik ---
    if show_bbox_center:
        for (x1, y1, x2, y2) in bbox_xyxy:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.circle(canvas, (cx, cy), sizes["marker"], (0, 255, 0), -1)

    # --- LAG 4: center-of-mass som "+" (segmenteret tyngdepunkt) ---
    if show_com and measurements:
        h = sizes["cross"]
        for m in measurements:
            cx, cy = int(round(m["com_x"])), int(round(m["com_y"]))
            cv2.line(canvas, (cx - h, cy), (cx + h, cy), (0, 0, 0), sizes["line"] + 2)
            cv2.line(canvas, (cx, cy - h), (cx, cy + h), (0, 0, 0), sizes["line"] + 2)
            cv2.line(canvas, (cx - h, cy), (cx + h, cy), (0, 255, 255), sizes["line"])
            cv2.line(canvas, (cx, cy - h), (cx, cy + h), (0, 255, 255), sizes["line"])

    # --- LAG 5: orienterings-pil (langs frøets hovedakse) ---
    if show_orientation and measurements:
        for m in measurements:
            cx, cy = int(round(m["com_x"])), int(round(m["com_y"]))
            ang = np.radians(m["angle_deg"])
            half = m["axis_half_len"]
            dx = int(round(np.cos(ang) * half))
            dy = int(round(np.sin(ang) * half))
            cv2.arrowedLine(canvas, (cx, cy), (cx + dx, cy + dy),
                            (255, 0, 0), sizes["line"], tipLength=0.3)

    # --- LAG 6: confidence-tal (øverst) ---
    if show_confidence:
        for (x1, y1, x2, y2), conf in zip(bbox_xyxy, confs):
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.putText(canvas, f"{conf:.2f}", (cx + sizes["marker"] + 2, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, sizes["font"], (255, 255, 0),
                        max(1, sizes["line"] - 1), cv2.LINE_AA)

    return canvas
