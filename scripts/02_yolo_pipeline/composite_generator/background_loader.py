"""Læs plade-baggrunde og sample crops fra blå zone.

Belt1.png og Belt2.png har en blå pyramide-tekstureret zone i midten
omgivet af en mørk brun kant. Modellen skal ikke se kanten under træning.
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
    """En pulje af baggrund-billeder med deres blå-zone-masker."""
    images: list[np.ndarray] = field(default_factory=list)
    blue_zone_masks: list[np.ndarray] = field(default_factory=list)

    def sample(self, size: tuple[int, int]) -> np.ndarray:
        """Sample en random crop af `size = (width, height)` fra blå zone.

        Forsøger op til 50 random crop-ankre. Hvis ingen passer i blå zone,
        rejser ValueError.
        """
        target_w, target_h = size
        for _ in range(50):
            idx = np.random.randint(len(self.images))
            img = self.images[idx]
            mask = self.blue_zone_masks[idx]
            h, w = img.shape[:2]

            if target_w > w or target_h > h:
                raise ValueError(
                    f"Target {size} for stor til baggrund {(w, h)}")

            x = np.random.randint(0, w - target_w + 1)
            y = np.random.randint(0, h - target_h + 1)

            # Tjek at hele crop'et ligger inden for blå zone
            if mask[y:y + target_h, x:x + target_w].all():
                return img[y:y + target_h, x:x + target_w, :].copy()

        raise ValueError(
            f"Kunne ikke finde gyldigt blue-zone-crop af {size} efter 50 forsøg")


def _compute_blue_zone_mask(rgb: np.ndarray) -> np.ndarray:
    """Identificér blå zone via HSV-tærskler. Returnerer boolsk maske.

    Anvender morphological closing for at fylde små huller fra tekstur.
    """
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Råmaske fra HSV-tærskler
    mask = (
        (h >= BLUE_HUE_MIN) & (h <= BLUE_HUE_MAX)
        & (s >= BLUE_SAT_MIN)
        & (v >= BLUE_VAL_MIN)
    )

    # Morphological closing (dilation + erosion) for at fylde små huller
    # 21x21 kernel er lille nok til at bevare kanter men stor nok til støj
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask_closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

    return mask_closed.astype(bool)


def load(paths: list[Path]) -> BackgroundPool:
    """Load alle baggrunde og beregn deres blå-zone-masker én gang."""
    pool = BackgroundPool()
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Kunne ikke loade {path}")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pool.images.append(rgb)
        pool.blue_zone_masks.append(_compute_blue_zone_mask(rgb))
    return pool
