"""Læs plade-baggrunde og pre-crop til blå zone.

Belt1.png og Belt2.png har en blå pyramide-tekstureret zone i midten
omgivet af en mørk brun kant. Videometers script segmenterer brun-kanten
væk i produktion, så modellen ser kun den blå zone — vi gør det samme her.

Alle baggrunde resizes til samme størrelse (første baggrunds blå-bbox)
så composites har en fast canonical canvas.
"""
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import cv2


# HSV-tærskler for blå zone (OpenCV: H er 0-180, ikke 0-360)
BLUE_HUE_MIN = 95     # ~190° i normalt rum
BLUE_HUE_MAX = 115    # ~230°
BLUE_SAT_MIN = 80     # mindst middel-mæthed
BLUE_VAL_MIN = 60     # mindst middel-lysstyrke


@dataclass
class BackgroundPool:
    """En pulje af pre-croppede plade-baggrunde, alle samme størrelse."""
    images: list[np.ndarray] = field(default_factory=list)
    # blue_zone_masks beholdes for bagudkompatibilitet med tests
    # men er nu hele billedet True (vi har allerede croppet)
    blue_zone_masks: list[np.ndarray] = field(default_factory=list)

    def sample(self, size: tuple[int, int]) -> np.ndarray:
        """Returnér én tilfældig pre-cropped baggrund, resized til `size`.

        size = (width, height). Da alle pre-croppede baggrunde har samme
        størrelse, er resize kun nødvendigt hvis caller ønsker en anden
        størrelse end pre-crop'et.
        """
        if not self.images:
            raise ValueError("BackgroundPool er tom")
        target_w, target_h = size
        idx = np.random.randint(len(self.images))
        img = self.images[idx]
        h, w = img.shape[:2]
        if (w, h) != (target_w, target_h):
            img = cv2.resize(img, (target_w, target_h),
                              interpolation=cv2.INTER_AREA)
        return img.copy()


def _compute_blue_zone_mask(rgb: np.ndarray) -> np.ndarray:
    """Identificér blå zone via HSV-tærskler. Returnerer boolsk maske.

    Anvender morphological closing for at fylde små huller fra tekstur.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    mask = (
        (h >= BLUE_HUE_MIN) & (h <= BLUE_HUE_MAX)
        & (s >= BLUE_SAT_MIN)
        & (v >= BLUE_VAL_MIN)
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask_closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    return mask_closed.astype(bool)


def _crop_to_blue_bbox(rgb: np.ndarray,
                        mask: np.ndarray) -> np.ndarray:
    """Crop billede til blå zones bounding box. Skærer brun-kant væk."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        raise ValueError("Ingen blå zone fundet i billede")
    y0, y1 = np.where(rows)[0][[0, -1]]
    x0, x1 = np.where(cols)[0][[0, -1]]
    return rgb[y0:y1 + 1, x0:x1 + 1, :]


def load(paths: list[Path]) -> BackgroundPool:
    """Load alle baggrunde, pre-crop til blå-zone, resize til fælles størrelse.

    Første baggrunds blå-zone-størrelse bliver canonical. Resten resizes
    til samme dimensioner så composites altid har én fast canvas.
    """
    pool = BackgroundPool()
    canonical_size = None  # (width, height) fra første baggrund

    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Kunne ikke loade {path}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = _compute_blue_zone_mask(rgb)
        cropped = _crop_to_blue_bbox(rgb, mask)
        h, w = cropped.shape[:2]

        if canonical_size is None:
            canonical_size = (w, h)
        elif (w, h) != canonical_size:
            # Resize til canonical så alle pre-croppede er samme størrelse
            cropped = cv2.resize(cropped, canonical_size,
                                  interpolation=cv2.INTER_AREA)

        pool.images.append(cropped)
        # Hele det pre-croppede billede er nu blå zone
        pool.blue_zone_masks.append(
            np.ones(cropped.shape[:2], dtype=bool))

    return pool
