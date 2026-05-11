# Composite-pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bygge en data-genererings-pipeline der tager 12.384 enkelt-frø PNG'er + 2 plade-baggrunde og producerer 2.000 composite-scener med YOLO-labels — klar til træning af v4-model.

**Architecture:** Ny Python-pakke `scripts/02_yolo_pipeline/composite_generator/` med 6 fokuserede moduler (seed_loader, background_loader, packing, composer, generate_dataset, visualize), TDD for det deterministiske (packing, label-format, alpha-handling) og visualize-driven for rendering-kvalitet. To manuelle review-gates inden fuld kørsel. Bevarer eksisterende `03_train_yolo.py` og `04_predict_and_evaluate_landmarks.py` uændrede.

**Tech Stack:** Python 3.11, NumPy, Pillow, OpenCV (HSV + rotation), SciPy (gaussian filter), PyYAML, pytest, Ultralytics YOLOv11n.

**Reference til spec:** `docs/specs/2026-05-11-composite-pipeline-design.md`.

---

## File Structure

```
scripts/02_yolo_pipeline/composite_generator/
├── __init__.py
├── seed_loader.py          # SeedStamp dataclass + scan() + load()
├── background_loader.py    # BackgroundPool dataclass + load() + sample()
├── packing.py              # Placement dataclass + scatter/dense/t_config + compute()
├── composer.py             # CompositeResult dataclass + render()
├── generate_dataset.py     # main() — driver der binder alt sammen
├── visualize.py            # CLI + funktion til preview.png
├── config.yaml             # alle hyper-parametre
└── tests/
    ├── __init__.py
    ├── conftest.py         # pytest-fixtures (mini-PNG'er, mini-baggrunde)
    ├── test_seed_loader.py
    ├── test_background_loader.py
    ├── test_packing.py
    ├── test_composer.py
    └── test_generate_dataset.py

scripts/02a_archived_ellipse_synthetic/
├── 01b_realistic_synthetic_data.py    # flyttet fra 02_yolo_pipeline/
├── 02_convert_to_yolo_format.py       # flyttet fra 02_yolo_pipeline/
└── README.md                          # forklarer hvorfor det er arkiveret

data/composite_v1/                     # output fra fuld kørsel
├── images/{train,val}/composite_*.png
├── labels/{train,val}/composite_*.txt
├── data.yaml
├── preview.png
└── generation_metadata.json

models/v4_composite/                   # ny model trænet på composites
├── best.pt
└── notes.md
```

---

## Task 1: Pakke-skelet og config

Sætte mappestruktur, config-fil og test-infrastructure op. Alt andet bygger ovenpå.

**Files:**
- Create: `scripts/02_yolo_pipeline/composite_generator/__init__.py`
- Create: `scripts/02_yolo_pipeline/composite_generator/config.yaml`
- Create: `scripts/02_yolo_pipeline/composite_generator/tests/__init__.py`
- Create: `scripts/02_yolo_pipeline/composite_generator/tests/conftest.py`

- [ ] **Step 1.1: Opret pakke-mapper og tomme `__init__.py`-filer**

```bash
mkdir -p scripts/02_yolo_pipeline/composite_generator/tests
touch scripts/02_yolo_pipeline/composite_generator/__init__.py
touch scripts/02_yolo_pipeline/composite_generator/tests/__init__.py
```

- [ ] **Step 1.2: Skriv `config.yaml` med alle defaults fra spec**

Indhold til `scripts/02_yolo_pipeline/composite_generator/config.yaml`:

```yaml
# === Paths (kan overrides ved kørsel) ===
seed_image_dir: ~/Development/dataset/Corn/images
background_paths:
  - ~/Development/dataset/Corn/Belt1.png
  - ~/Development/dataset/Corn/Belt2.png
output_dir: ~/Development/school/fcm_project_export/data/composite_v1

# === Dataset-skala ===
num_composites: 2000
train_val_split: 0.80
random_seed: 42

# === Composite-størrelse ===
canvas_size: [2000, 1500]   # (width, height)

# === Frø pr. scene ===
seeds_per_composite:
  min: 40
  max: 80

# === Pakkings-strategi ===
packing_strategy_mix:
  scatter: 0.50
  dense: 0.30
  t_config: 0.20
packing_gap_pixels: 4
max_placement_attempts: 200

# === Frø-transformation ===
seed_rotation_range: [0, 360]
seed_scale_jitter: [0.85, 1.15]

# === Rendering ===
shadow:
  enabled: true
  offset_px: [3, 3]
  blur_sigma: 4.0
  opacity: 0.35
edge_feather_px: 1.5

# === Baggrund-augmentation ===
background_random_rotation: true
background_color_jitter: 0.05
background_brightness_jitter: 0.10

# === Output ===
image_format: png
bbox_strategy: tight

# === Validering ===
visualize_n_samples: 8
log_seed_metadata: true
```

- [ ] **Step 1.3: Skriv `conftest.py` med pytest-fixtures**

Indhold til `scripts/02_yolo_pipeline/composite_generator/tests/conftest.py`:

```python
"""Pytest-fixtures til composite-pipeline tests.

Vi laver små test-fixtures (10x10 frø, 200x200 baggrunde) i en tempdir
så testene er hurtige og uafhængige af det rigtige dataset.
"""
import numpy as np
import pytest
from pathlib import Path
from PIL import Image


@pytest.fixture
def tiny_seed_png(tmp_path):
    """Lav en 20x30 RGBA-PNG med solid alpha i midten, transparent i kanten."""
    rgba = np.zeros((30, 20, 4), dtype=np.uint8)
    rgba[5:25, 3:17, :3] = 200          # gul-agtig RGB
    rgba[5:25, 3:17, 3] = 255           # solid alpha midt
    path = tmp_path / "tiny_seed.png"
    Image.fromarray(rgba, mode="RGBA").save(path)
    return path


@pytest.fixture
def corrupt_seed_png(tmp_path):
    """Lav en fil der ligner en PNG men er korrupt."""
    path = tmp_path / "corrupt.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\nnotapng")
    return path


@pytest.fixture
def fully_transparent_seed_png(tmp_path):
    """RGBA-PNG hvor alle pixels har alpha=0."""
    rgba = np.zeros((20, 20, 4), dtype=np.uint8)
    path = tmp_path / "fully_transparent.png"
    Image.fromarray(rgba, mode="RGBA").save(path)
    return path


@pytest.fixture
def rgb_only_seed_png(tmp_path):
    """PNG uden alpha-kanal (mode RGB) — skal afvises af loader."""
    rgb = np.full((20, 20, 3), 100, dtype=np.uint8)
    path = tmp_path / "rgb_only.png"
    Image.fromarray(rgb, mode="RGB").save(path)
    return path


@pytest.fixture
def tiny_blue_background_png(tmp_path):
    """200x150 PNG hvor midten er blå (HSV ~200°), kanterne er brune."""
    bgr = np.zeros((150, 200, 3), dtype=np.uint8)
    bgr[:, :] = (50, 30, 80)           # brun-agtig BGR (mørk)
    bgr[20:130, 30:170] = (180, 100, 30)  # blå BGR (lyseblå)
    path = tmp_path / "tiny_belt.png"
    Image.fromarray(bgr[..., ::-1], mode="RGB").save(path)  # konverter til RGB
    return path
```

- [ ] **Step 1.4: Verificér at pytest kan finde testene**

Kør:
```bash
cd /Users/andreassogardgabel/Development/school/fcm_project_export
.venv/bin/pytest scripts/02_yolo_pipeline/composite_generator/tests/ -v
```

Forventet output: `no tests ran in X.XX s` (ingen tests endnu, men fixtures-modulet loader uden fejl).

- [ ] **Step 1.5: Commit**

```bash
git add scripts/02_yolo_pipeline/composite_generator/
git commit -m "composite_generator: pakke-skelet + config.yaml + pytest-fixtures"
```

---

## Task 2: `seed_loader.py` — TDD

Læs og preprocess frø-PNG'er. Skal håndtere RGBA, crop til indhold, skip korrupte filer.

**Files:**
- Create: `scripts/02_yolo_pipeline/composite_generator/tests/test_seed_loader.py`
- Create: `scripts/02_yolo_pipeline/composite_generator/seed_loader.py`

- [ ] **Step 2.1: Skriv failing tests**

Indhold til `scripts/02_yolo_pipeline/composite_generator/tests/test_seed_loader.py`:

```python
"""Tests for seed_loader.py."""
import numpy as np
from pathlib import Path
import pytest

from scripts.02_yolo_pipeline.composite_generator import seed_loader


def test_load_returns_seed_stamp(tiny_seed_png):
    """Gyldig RGBA-PNG returnerer en SeedStamp."""
    stamp = seed_loader.load(tiny_seed_png)
    assert stamp is not None
    assert stamp.rgba.shape[2] == 4  # RGBA
    assert stamp.rgba.dtype == np.uint8
    assert stamp.source_id == "tiny_seed"


def test_load_crops_transparent_margin(tiny_seed_png):
    """Transparent margin skal være fjernet — output har solid alpha tæt på kanten."""
    stamp = seed_loader.load(tiny_seed_png)
    # Top-row alpha skal være > 0 (vi har croppet til indhold)
    assert stamp.rgba[0, :, 3].max() > 0
    # Output skal være mindre end input (20x30 → cropped)
    assert stamp.rgba.shape[0] < 30 or stamp.rgba.shape[1] < 20


def test_load_rejects_non_rgba(rgb_only_seed_png):
    """PNG uden alpha-kanal returnerer None."""
    assert seed_loader.load(rgb_only_seed_png) is None


def test_load_rejects_fully_transparent(fully_transparent_seed_png):
    """PNG med kun transparente pixels returnerer None."""
    assert seed_loader.load(fully_transparent_seed_png) is None


def test_load_handles_corrupt(corrupt_seed_png):
    """Korrupt PNG returnerer None uden at crashe."""
    assert seed_loader.load(corrupt_seed_png) is None


def test_scan_finds_all_pngs(tmp_path, tiny_seed_png):
    """scan() finder alle PNG'er i en mappe."""
    # tiny_seed_png ligger allerede i tmp_path
    paths = seed_loader.scan(tmp_path)
    assert len(paths) >= 1
    assert all(p.suffix == ".png" for p in paths)
```

- [ ] **Step 2.2: Kør tests, verificer FAIL**

```bash
.venv/bin/pytest scripts/02_yolo_pipeline/composite_generator/tests/test_seed_loader.py -v
```

Forventet: alle 6 tests fejler med "ModuleNotFoundError: No module named 'scripts.02_yolo_pipeline...'" eller "no attribute 'load'".

Bemærk: pakke-stien `scripts.02_yolo_pipeline...` virker ikke fordi mappenavnet starter med tal. Vi importerer i stedet via `sys.path`-tricket i `conftest.py`. Tilføj øverst i conftest.py:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

Og rette imports i test-filen til:
```python
import seed_loader
```

Kør testen igen. Forventet: FAIL med "ModuleNotFoundError: No module named 'seed_loader'".

- [ ] **Step 2.3: Skriv minimal `seed_loader.py`**

Indhold til `scripts/02_yolo_pipeline/composite_generator/seed_loader.py`:

```python
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
```

- [ ] **Step 2.4: Kør tests, verificer PASS**

```bash
.venv/bin/pytest scripts/02_yolo_pipeline/composite_generator/tests/test_seed_loader.py -v
```

Forventet: alle 6 tests PASS.

- [ ] **Step 2.5: Commit**

```bash
git add scripts/02_yolo_pipeline/composite_generator/seed_loader.py \
        scripts/02_yolo_pipeline/composite_generator/tests/test_seed_loader.py \
        scripts/02_yolo_pipeline/composite_generator/tests/conftest.py
git commit -m "composite_generator: seed_loader + tests (TDD)"
```

---

## Task 3: `background_loader.py` — TDD

Læs plade-baggrunde, detekter blå zone via HSV, sample crops.

**Files:**
- Create: `scripts/02_yolo_pipeline/composite_generator/tests/test_background_loader.py`
- Create: `scripts/02_yolo_pipeline/composite_generator/background_loader.py`

- [ ] **Step 3.1: Skriv failing tests**

Indhold til `scripts/02_yolo_pipeline/composite_generator/tests/test_background_loader.py`:

```python
"""Tests for background_loader.py."""
import numpy as np
import pytest

import background_loader


def test_load_returns_background_pool(tiny_blue_background_png):
    """load() returnerer BackgroundPool med korrekt antal billeder."""
    pool = background_loader.load([tiny_blue_background_png])
    assert len(pool.images) == 1
    assert len(pool.blue_zone_masks) == 1


def test_blue_zone_mask_identifies_blue_region(tiny_blue_background_png):
    """Blå zone-masken markerer pixels i midten som blå, ikke kanter."""
    pool = background_loader.load([tiny_blue_background_png])
    mask = pool.blue_zone_masks[0]
    # Midten skal være True (blå), kanterne False (brun)
    assert mask[75, 100] == True   # midten
    assert mask[0, 0] == False     # hjørne (brun)


def test_sample_returns_correct_size(tiny_blue_background_png):
    """sample(size) returnerer et numpy-array af præcis target-størrelse."""
    pool = background_loader.load([tiny_blue_background_png])
    crop = pool.sample(size=(80, 60))   # (width, height)
    assert crop.shape == (60, 80, 3)
    assert crop.dtype == np.uint8


def test_sample_only_returns_blue_region(tiny_blue_background_png):
    """sample() skal kun returnere crops der ligger inden for blå zone."""
    pool = background_loader.load([tiny_blue_background_png])
    # Lav 20 samples og verificer at alle er overvejende blå
    for _ in range(20):
        crop = pool.sample(size=(50, 50))
        # Mean blue-channel skal være højere end mean red — krude blå-test
        mean_b = crop[:, :, 2].mean()
        mean_r = crop[:, :, 0].mean()
        # Vores fixture-baggrund har RGB(30,100,180) i blå zone
        assert mean_b > mean_r, f"Crop er ikke blå: R={mean_r:.0f}, B={mean_b:.0f}"


def test_sample_fails_when_target_too_large(tiny_blue_background_png):
    """Hvis requested size er større end blå zone, hård fail."""
    pool = background_loader.load([tiny_blue_background_png])
    with pytest.raises(ValueError, match="for stor"):
        pool.sample(size=(500, 500))
```

- [ ] **Step 3.2: Kør tests, verificer FAIL**

```bash
.venv/bin/pytest scripts/02_yolo_pipeline/composite_generator/tests/test_background_loader.py -v
```

Forventet: alle 5 tests fejler med `ModuleNotFoundError`.

- [ ] **Step 3.3: Skriv minimal `background_loader.py`**

Indhold til `scripts/02_yolo_pipeline/composite_generator/background_loader.py`:

```python
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
    """Identificér blå zone via HSV-tærskler. Returnerer boolsk maske."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    return (
        (h >= BLUE_HUE_MIN) & (h <= BLUE_HUE_MAX)
        & (s >= BLUE_SAT_MIN)
        & (v >= BLUE_VAL_MIN)
    )


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
```

- [ ] **Step 3.4: Kør tests, verificer PASS**

```bash
.venv/bin/pytest scripts/02_yolo_pipeline/composite_generator/tests/test_background_loader.py -v
```

Forventet: alle 5 tests PASS.

- [ ] **Step 3.5: Sanity-check på rigtige Belt-billeder**

Kør i terminal:
```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, 'scripts/02_yolo_pipeline/composite_generator')
import background_loader
from pathlib import Path
pool = background_loader.load([
    Path.home() / 'Development/dataset/Corn/Belt1.png',
    Path.home() / 'Development/dataset/Corn/Belt2.png',
])
print('Loaded:', len(pool.images), 'backgrounds')
for i, m in enumerate(pool.blue_zone_masks):
    print(f'  Belt{i+1}: blå zone er {m.sum()/m.size:.1%} af billede ({m.shape})')
crop = pool.sample(size=(800, 600))
print('Sample crop shape:', crop.shape, 'mean BGR:', crop.mean(axis=(0,1)))
"
```

Forventet: begge baggrunde loader, blå zone er >50% af hvert billede, sample-crop er overvejende blå (mean B > mean R).

Hvis blå zone er <30%: gå tilbage og justér `BLUE_HUE_MIN/MAX/SAT/VAL` i `background_loader.py`.

- [ ] **Step 3.6: Commit**

```bash
git add scripts/02_yolo_pipeline/composite_generator/background_loader.py \
        scripts/02_yolo_pipeline/composite_generator/tests/test_background_loader.py
git commit -m "composite_generator: background_loader med HSV blue-zone detection"
```

---

## Task 4: `packing.py` — scatter-strategi (TDD)

Bare scatter-strategi nu (MVP). Dense og t_config kommer i Task 10-12.

**Files:**
- Create: `scripts/02_yolo_pipeline/composite_generator/tests/test_packing.py`
- Create: `scripts/02_yolo_pipeline/composite_generator/packing.py`

- [ ] **Step 4.1: Skriv failing tests**

Indhold til `scripts/02_yolo_pipeline/composite_generator/tests/test_packing.py`:

```python
"""Tests for packing.py."""
import numpy as np
import pytest

import packing
from seed_loader import SeedStamp


def _make_stamp(w: int, h: int, idx: int = 0) -> SeedStamp:
    """Hjælper: lav en SeedStamp af given størrelse."""
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 3] = 255  # solid alpha
    return SeedStamp(rgba=rgba, source_id=f"seed_{idx}",
                     bbox_after_crop=(0, 0, w, h))


def test_scatter_returns_correct_count():
    """scatter med plads til alle returnerer alle placements."""
    stamps = [_make_stamp(30, 30, i) for i in range(5)]
    placements = packing.scatter(
        canvas_size=(500, 500), seed_stamps=stamps,
        gap_pixels=4, max_attempts=200, rng=np.random.default_rng(0))
    assert len(placements) == 5


def test_scatter_positions_within_canvas():
    """Alle placements har center inden for canvas."""
    stamps = [_make_stamp(20, 20, i) for i in range(10)]
    placements = packing.scatter(
        canvas_size=(300, 300), seed_stamps=stamps,
        gap_pixels=4, max_attempts=200, rng=np.random.default_rng(0))
    for p in placements:
        assert 0 <= p.center_x < 300
        assert 0 <= p.center_y < 300


def test_scatter_no_overlap():
    """Ingen to placements overlapper (afstand > sum af radier + gap)."""
    stamps = [_make_stamp(30, 30, i) for i in range(8)]
    placements = packing.scatter(
        canvas_size=(400, 400), seed_stamps=stamps,
        gap_pixels=4, max_attempts=200, rng=np.random.default_rng(0))
    for i, p1 in enumerate(placements):
        for p2 in placements[i+1:]:
            dx = p1.center_x - p2.center_x
            dy = p1.center_y - p2.center_y
            dist = (dx*dx + dy*dy) ** 0.5
            # Radier er ~15 hver, gap=4 → forventet min afstand ~34
            assert dist >= 30, f"For tæt: {dist:.1f}"


def test_scatter_accepts_fewer_when_canvas_full():
    """Hvis canvas er for fyldt, accepter færre placements."""
    stamps = [_make_stamp(80, 80, i) for i in range(20)]
    placements = packing.scatter(
        canvas_size=(200, 200), seed_stamps=stamps,
        gap_pixels=4, max_attempts=50, rng=np.random.default_rng(0))
    assert len(placements) < 20  # ikke alle passer


def test_scatter_rotation_in_range():
    """Alle rotationer er i [0, 360)."""
    stamps = [_make_stamp(25, 25, i) for i in range(5)]
    placements = packing.scatter(
        canvas_size=(400, 400), seed_stamps=stamps,
        gap_pixels=4, max_attempts=200, rng=np.random.default_rng(0))
    for p in placements:
        assert 0 <= p.rotation_deg < 360


def test_scatter_deterministic_with_seed():
    """Samme RNG-seed giver samme placements."""
    stamps = [_make_stamp(25, 25, i) for i in range(5)]
    p1 = packing.scatter(canvas_size=(400, 400), seed_stamps=stamps,
                          gap_pixels=4, max_attempts=200,
                          rng=np.random.default_rng(42))
    p2 = packing.scatter(canvas_size=(400, 400), seed_stamps=stamps,
                          gap_pixels=4, max_attempts=200,
                          rng=np.random.default_rng(42))
    assert [(p.center_x, p.center_y) for p in p1] == \
           [(p.center_x, p.center_y) for p in p2]
```

- [ ] **Step 4.2: Kør tests, verificer FAIL**

```bash
.venv/bin/pytest scripts/02_yolo_pipeline/composite_generator/tests/test_packing.py -v
```

Forventet: alle 6 tests fejler med `ModuleNotFoundError`.

- [ ] **Step 4.3: Skriv minimal `packing.py` med scatter**

Indhold til `scripts/02_yolo_pipeline/composite_generator/packing.py`:

```python
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
```

- [ ] **Step 4.4: Kør tests, verificer PASS**

```bash
.venv/bin/pytest scripts/02_yolo_pipeline/composite_generator/tests/test_packing.py -v
```

Forventet: alle 6 tests PASS.

- [ ] **Step 4.5: Commit**

```bash
git add scripts/02_yolo_pipeline/composite_generator/packing.py \
        scripts/02_yolo_pipeline/composite_generator/tests/test_packing.py
git commit -m "composite_generator: packing.scatter() med rejection sampling (TDD)"
```

---

## Task 5: `composer.py` — minimal (uden skygge)

MVP-version: paste frø på baggrund via alpha-blending. Ingen skygge, ingen feather. Tilføjes i Task 9.

**Files:**
- Create: `scripts/02_yolo_pipeline/composite_generator/tests/test_composer.py`
- Create: `scripts/02_yolo_pipeline/composite_generator/composer.py`

- [ ] **Step 5.1: Skriv failing tests**

Indhold til `scripts/02_yolo_pipeline/composite_generator/tests/test_composer.py`:

```python
"""Tests for composer.py."""
import numpy as np
import pytest

import composer
from seed_loader import SeedStamp
from packing import Placement


def _make_stamp(w: int, h: int, idx: int = 0) -> SeedStamp:
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = 200    # lys gul-agtig
    rgba[:, :, 3] = 255     # solid alpha
    return SeedStamp(rgba=rgba, source_id=f"seed_{idx}",
                     bbox_after_crop=(0, 0, w, h))


def _make_bg(w: int, h: int) -> np.ndarray:
    return np.full((h, w, 3), 50, dtype=np.uint8)   # mørk baggrund


def test_render_returns_correct_image_size():
    """Output-billedets størrelse matcher baggrunds-størrelsen."""
    bg = _make_bg(200, 150)
    stamps = [_make_stamp(20, 30)]
    placements = [Placement(center_x=100, center_y=75, rotation_deg=0, seed_index=0)]
    result = composer.render(bg, stamps, placements,
                              shadow_config=None, feather_px=0.0)
    assert result.image.shape == (150, 200, 3)
    assert result.image.dtype == np.uint8


def test_render_returns_centers_matching_placements():
    """Antal centers = antal placements (når alle er inden for canvas)."""
    bg = _make_bg(300, 300)
    stamps = [_make_stamp(20, 20) for _ in range(3)]
    placements = [
        Placement(center_x=50, center_y=50, rotation_deg=0, seed_index=0),
        Placement(center_x=150, center_y=150, rotation_deg=45, seed_index=1),
        Placement(center_x=250, center_y=100, rotation_deg=90, seed_index=2),
    ]
    result = composer.render(bg, stamps, placements,
                              shadow_config=None, feather_px=0.0)
    assert len(result.centers) == 3
    assert len(result.bbox_sizes) == 3


def test_render_actually_modifies_background():
    """Efter render skal pixels under frø være anderledes end baggrund."""
    bg = _make_bg(200, 200)
    stamps = [_make_stamp(40, 40)]
    placements = [Placement(center_x=100, center_y=100, rotation_deg=0, seed_index=0)]
    result = composer.render(bg, stamps, placements,
                              shadow_config=None, feather_px=0.0)
    # Pixel ved center skal nu have stamp-farve (200), ikke baggrunds-farve (50)
    assert result.image[100, 100, 0] > 100


def test_render_centers_within_canvas():
    """Alle registrerede centers ligger inden for canvas."""
    bg = _make_bg(200, 200)
    stamps = [_make_stamp(20, 20) for _ in range(5)]
    placements = [
        Placement(center_x=50, center_y=50, rotation_deg=0, seed_index=0),
        Placement(center_x=150, center_y=150, rotation_deg=0, seed_index=1),
        Placement(center_x=100, center_y=100, rotation_deg=0, seed_index=2),
        Placement(center_x=30, center_y=180, rotation_deg=0, seed_index=3),
        Placement(center_x=180, center_y=30, rotation_deg=0, seed_index=4),
    ]
    result = composer.render(bg, stamps, placements,
                              shadow_config=None, feather_px=0.0)
    for cx, cy in result.centers:
        assert 0 <= cx < 200
        assert 0 <= cy < 200
```

- [ ] **Step 5.2: Kør tests, verificer FAIL**

```bash
.venv/bin/pytest scripts/02_yolo_pipeline/composite_generator/tests/test_composer.py -v
```

Forventet: alle 4 tests fejler med `ModuleNotFoundError`.

- [ ] **Step 5.3: Skriv minimal `composer.py`**

Indhold til `scripts/02_yolo_pipeline/composite_generator/composer.py`:

```python
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
    """Rotér RGBA-stamp omkring sit centrum, uden at klippe indhold væk."""
    h, w = rgba.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, scale=1.0)

    # Beregn ny bounding box så hele det roterede stamp passer ind
    cos_a = abs(matrix[0, 0])
    sin_a = abs(matrix[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    # Justér translation så midten rammer ny center
    matrix[0, 2] += (new_w - w) / 2
    matrix[1, 2] += (new_h - h) / 2

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
    """Render én composite med alle placements paste'et på baggrund."""
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

        # Clipping mod canvas
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(canvas_w, x1), min(canvas_h, y1)
        if cx0 >= cx1 or cy0 >= cy1:
            continue  # helt uden for canvas

        # Tilsvarende slice i stamp
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

        centers.append((p.center_x, p.center_y))
        bbox_sizes.append((sw, sh))

    return CompositeResult(image=canvas, centers=centers, bbox_sizes=bbox_sizes)
```

- [ ] **Step 5.4: Kør tests, verificer PASS**

```bash
.venv/bin/pytest scripts/02_yolo_pipeline/composite_generator/tests/test_composer.py -v
```

Forventet: alle 4 tests PASS.

- [ ] **Step 5.5: Commit**

```bash
git add scripts/02_yolo_pipeline/composite_generator/composer.py \
        scripts/02_yolo_pipeline/composite_generator/tests/test_composer.py
git commit -m "composite_generator: composer.render() MVP uden skygge (TDD)"
```

---

## Task 6: `generate_dataset.py` — MVP driver

Binder seed_loader + background_loader + scatter + composer sammen. Skriver YOLO-format dataset.

**Files:**
- Create: `scripts/02_yolo_pipeline/composite_generator/tests/test_generate_dataset.py`
- Create: `scripts/02_yolo_pipeline/composite_generator/generate_dataset.py`

- [ ] **Step 6.1: Skriv failing tests**

Indhold til `scripts/02_yolo_pipeline/composite_generator/tests/test_generate_dataset.py`:

```python
"""Integration-tests for generate_dataset.py."""
import numpy as np
import pytest
import yaml
from pathlib import Path
from PIL import Image

import generate_dataset


@pytest.fixture
def mini_corpus(tmp_path):
    """Lav et mini-corpus: 10 frø-PNG'er + 1 baggrund."""
    seed_dir = tmp_path / "seeds"
    seed_dir.mkdir()
    for i in range(10):
        rgba = np.zeros((40, 30, 4), dtype=np.uint8)
        rgba[5:35, 3:27, :3] = 200
        rgba[5:35, 3:27, 3] = 255
        Image.fromarray(rgba, mode="RGBA").save(seed_dir / f"seed_{i:03d}.png")

    bg_path = tmp_path / "background.png"
    bg = np.zeros((400, 500, 3), dtype=np.uint8)
    bg[:, :] = (50, 30, 80)         # brun
    bg[40:360, 50:450] = (30, 100, 180)  # blå (RGB)
    Image.fromarray(bg, mode="RGB").save(bg_path)

    return seed_dir, [bg_path]


def test_generates_correct_file_structure(mini_corpus, tmp_path):
    """Efter run findes alle forventede mapper og filer."""
    seed_dir, bg_paths = mini_corpus
    output_dir = tmp_path / "output"

    config = generate_dataset.Config(
        seed_image_dir=seed_dir, background_paths=bg_paths,
        output_dir=output_dir, num_composites=4, train_val_split=0.75,
        random_seed=42, canvas_size=(300, 200),
        seeds_min=3, seeds_max=5,
        gap_pixels=2, max_attempts=100,
    )
    generate_dataset.run(config)

    assert (output_dir / "images" / "train").is_dir()
    assert (output_dir / "images" / "val").is_dir()
    assert (output_dir / "labels" / "train").is_dir()
    assert (output_dir / "labels" / "val").is_dir()
    assert (output_dir / "data.yaml").is_file()
    assert (output_dir / "generation_metadata.json").is_file()

    # 4 composites med 75% split = 3 train + 1 val
    assert len(list((output_dir / "images" / "train").glob("*.png"))) == 3
    assert len(list((output_dir / "images" / "val").glob("*.png"))) == 1


def test_labels_have_valid_yolo_format(mini_corpus, tmp_path):
    """Hver label-linje har 5 floats i [0, 1]."""
    seed_dir, bg_paths = mini_corpus
    output_dir = tmp_path / "output"
    config = generate_dataset.Config(
        seed_image_dir=seed_dir, background_paths=bg_paths,
        output_dir=output_dir, num_composites=2, train_val_split=0.5,
        random_seed=42, canvas_size=(300, 200),
        seeds_min=3, seeds_max=5,
        gap_pixels=2, max_attempts=100,
    )
    generate_dataset.run(config)

    for label_file in (output_dir / "labels" / "train").glob("*.txt"):
        for line in label_file.read_text().strip().splitlines():
            parts = line.split()
            assert len(parts) == 5
            cls = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
            assert cls == 0
            assert 0 <= cx <= 1, f"cx ud af range i {label_file}: {cx}"
            assert 0 <= cy <= 1
            assert 0 < w <= 1
            assert 0 < h <= 1


def test_data_yaml_has_correct_structure(mini_corpus, tmp_path):
    """data.yaml har de felter YOLOv11 forventer."""
    seed_dir, bg_paths = mini_corpus
    output_dir = tmp_path / "output"
    config = generate_dataset.Config(
        seed_image_dir=seed_dir, background_paths=bg_paths,
        output_dir=output_dir, num_composites=2, train_val_split=0.5,
        random_seed=42, canvas_size=(300, 200),
        seeds_min=3, seeds_max=5,
        gap_pixels=2, max_attempts=100,
    )
    generate_dataset.run(config)

    data_yaml = yaml.safe_load((output_dir / "data.yaml").read_text())
    assert "path" in data_yaml
    assert data_yaml["train"] == "images/train"
    assert data_yaml["val"] == "images/val"
    assert data_yaml["names"] == ["seed"]
    assert data_yaml["nc"] == 1
```

- [ ] **Step 6.2: Kør tests, verificer FAIL**

```bash
.venv/bin/pytest scripts/02_yolo_pipeline/composite_generator/tests/test_generate_dataset.py -v
```

Forventet: alle 3 tests fejler med `ModuleNotFoundError`.

- [ ] **Step 6.3: Skriv `generate_dataset.py` MVP-version**

Indhold til `scripts/02_yolo_pipeline/composite_generator/generate_dataset.py`:

```python
"""Driver: binder seed_loader + background_loader + packing + composer sammen.

MVP-version (Task 6):
- Bruger kun scatter-pakning
- Ingen skygge eller edge-feather
- Læser config fra CLI-arg eller dataclass

Realisme-features tilføjes i Task 9-12.
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import numpy as np
import yaml
from PIL import Image

import seed_loader
import background_loader
import packing
import composer


@dataclass
class Config:
    """Subset af config.yaml — kun det MVP-driveren bruger."""
    seed_image_dir: Path
    background_paths: list[Path]
    output_dir: Path
    num_composites: int
    train_val_split: float
    random_seed: int
    canvas_size: tuple[int, int]   # (width, height)
    seeds_min: int
    seeds_max: int
    gap_pixels: int
    max_attempts: int

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            seed_image_dir=Path(data["seed_image_dir"]).expanduser(),
            background_paths=[Path(p).expanduser() for p in data["background_paths"]],
            output_dir=Path(data["output_dir"]).expanduser(),
            num_composites=data["num_composites"],
            train_val_split=data["train_val_split"],
            random_seed=data["random_seed"],
            canvas_size=tuple(data["canvas_size"]),
            seeds_min=data["seeds_per_composite"]["min"],
            seeds_max=data["seeds_per_composite"]["max"],
            gap_pixels=data["packing_gap_pixels"],
            max_attempts=data["max_placement_attempts"],
        )


def _setup_dirs(output_dir: Path) -> None:
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def _write_label(path: Path, centers, bbox_sizes, canvas_size) -> None:
    """Skriv YOLO-format label: class cx cy w h (alle normaliseret 0-1)."""
    canvas_w, canvas_h = canvas_size
    lines = []
    for (cx, cy), (w, h) in zip(centers, bbox_sizes):
        cx_n = cx / canvas_w
        cy_n = cy / canvas_h
        w_n = w / canvas_w
        h_n = h / canvas_h
        # Clip til [0, 1] for at undgå out-of-bounds
        cx_n = max(0.0, min(1.0, cx_n))
        cy_n = max(0.0, min(1.0, cy_n))
        w_n = max(0.001, min(1.0, w_n))
        h_n = max(0.001, min(1.0, h_n))
        lines.append(f"0 {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _atomic_write_image(path: Path, image: np.ndarray) -> None:
    """Skriv PNG atomic: først .tmp, så rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(image, mode="RGB").save(tmp)
    tmp.rename(path)


def run(config: Config) -> None:
    """Hovedfunktion: generér num_composites composites."""
    _setup_dirs(config.output_dir)

    rng = np.random.default_rng(config.random_seed)

    print(f"[1/3] Loader baggrunde fra {len(config.background_paths)} stier")
    bg_pool = background_loader.load(config.background_paths)

    print(f"[2/3] Scanner seed-mappen: {config.seed_image_dir}")
    seed_paths = seed_loader.scan(config.seed_image_dir)
    print(f"      Fandt {len(seed_paths)} frø-PNG'er")
    if len(seed_paths) == 0:
        raise RuntimeError(f"Ingen PNG'er i {config.seed_image_dir}")

    print(f"[3/3] Genererer {config.num_composites} composites")
    start = time.time()
    n_train = int(config.num_composites * config.train_val_split)

    for i in range(config.num_composites):
        split = "train" if i < n_train else "val"
        image_path = config.output_dir / "images" / split / f"composite_{i:05d}.png"
        label_path = config.output_dir / "labels" / split / f"composite_{i:05d}.txt"

        # Resume: skip hvis output allerede findes
        if image_path.exists() and label_path.exists():
            continue

        # Sample baggrund
        bg = bg_pool.sample(size=config.canvas_size)

        # Sample frø
        seed_count = int(rng.integers(config.seeds_min, config.seeds_max + 1))
        chosen = rng.choice(len(seed_paths), size=seed_count, replace=False)
        stamps = []
        for idx in chosen:
            stamp = seed_loader.load(seed_paths[idx])
            if stamp is not None:
                stamps.append(stamp)

        if not stamps:
            print(f"  Composite {i}: ingen gyldige frø, springer over")
            continue

        # Pak frø
        placements = packing.scatter(
            canvas_size=config.canvas_size, seed_stamps=stamps,
            gap_pixels=config.gap_pixels, max_attempts=config.max_attempts,
            rng=rng)

        # Render
        result = composer.render(bg, stamps, placements,
                                  shadow_config=None, feather_px=0.0)

        # Skriv atomic
        _atomic_write_image(image_path, result.image)
        _write_label(label_path, result.centers, result.bbox_sizes,
                      config.canvas_size)

        if (i + 1) % 5 == 0 or (i + 1) == config.num_composites:
            elapsed = time.time() - start
            avg_seeds = len(result.centers)
            print(f"  {i+1}/{config.num_composites}  ({elapsed:.0f}s,  "
                  f"sidste scene: {avg_seeds} frø)")

    # data.yaml
    data_yaml = {
        "path": str(config.output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": ["seed"],
        "nc": 1,
    }
    (config.output_dir / "data.yaml").write_text(yaml.safe_dump(data_yaml))

    # generation_metadata.json
    metadata = {
        "config": {k: (str(v) if isinstance(v, Path) else
                        [str(p) for p in v] if isinstance(v, list) else v)
                    for k, v in asdict(config).items()},
        "num_seeds_in_pool": len(seed_paths),
        "elapsed_seconds": time.time() - start,
    }
    (config.output_dir / "generation_metadata.json").write_text(
        json.dumps(metadata, indent=2))

    print(f"\nFærdig. Output: {config.output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",
                         default=str(Path(__file__).parent / "config.yaml"))
    parser.add_argument("--num-composites", type=int, default=None,
                         help="Override num_composites fra config")
    parser.add_argument("--output-dir", default=None,
                         help="Override output_dir fra config")
    args = parser.parse_args()

    config = Config.from_yaml(Path(args.config))
    if args.num_composites is not None:
        config.num_composites = args.num_composites
    if args.output_dir is not None:
        config.output_dir = Path(args.output_dir).expanduser()

    run(config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6.4: Kør tests, verificer PASS**

```bash
.venv/bin/pytest scripts/02_yolo_pipeline/composite_generator/tests/test_generate_dataset.py -v
```

Forventet: alle 3 tests PASS.

- [ ] **Step 6.5: Commit**

```bash
git add scripts/02_yolo_pipeline/composite_generator/generate_dataset.py \
        scripts/02_yolo_pipeline/composite_generator/tests/test_generate_dataset.py
git commit -m "composite_generator: generate_dataset MVP driver med scatter-only (TDD)"
```

---

## Task 7: `visualize.py` — sanity-check

Producér `preview.png` med 8 random samples + GT-markører.

**Files:**
- Create: `scripts/02_yolo_pipeline/composite_generator/visualize.py`

- [ ] **Step 7.1: Skriv `visualize.py`**

Indhold til `scripts/02_yolo_pipeline/composite_generator/visualize.py`:

```python
"""Sanity-check af genereret composite-dataset.

Auto-mode: kaldes af generate_dataset.py som sidste step.
Standalone:  python visualize.py --dataset data/composite_v1 --n 8

Output: preview.png (grid med composites + grønne GT-markører).
"""
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def _load_labels(label_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    """Læs YOLO-labels, returnér array af (cx, cy) i pixel-koordinater."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    w, h = image_size
    centers = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        cx = float(parts[1]) * w
        cy = float(parts[2]) * h
        centers.append((cx, cy))
    return np.array(centers, dtype=np.float32)


def make_preview(dataset_dir: Path, n_samples: int = 8,
                  output: Path = None, split: str = "train") -> Path:
    """Lav preview.png med n_samples random composites + GT-overlay."""
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split

    all_images = sorted(images_dir.glob("*.png"))
    if not all_images:
        raise FileNotFoundError(f"Ingen billeder i {images_dir}")

    n_actual = min(n_samples, len(all_images))
    rng = np.random.default_rng(0)
    chosen_idx = rng.choice(len(all_images), size=n_actual, replace=False)
    chosen = [all_images[i] for i in chosen_idx]

    cols = 4
    rows = (n_actual + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.atleast_2d(axes).flatten()

    for ax_idx, img_path in enumerate(chosen):
        img = np.array(Image.open(img_path))
        label_path = labels_dir / (img_path.stem + ".txt")
        centers = _load_labels(label_path, image_size=img.shape[:2][::-1])

        ax = axes[ax_idx]
        ax.imshow(img)
        if len(centers) > 0:
            ax.scatter(centers[:, 0], centers[:, 1],
                        s=15, c="lime", marker="o",
                        edgecolors="red", linewidths=0.5)
        ax.set_title(f"{img_path.stem}  ({len(centers)} frø)", fontsize=8)
        ax.axis("off")

    # Skjul ubrugte axes
    for j in range(ax_idx + 1, len(axes)):
        axes[j].axis("off")

    if output is None:
        output = dataset_dir / "preview.png"
    plt.tight_layout()
    plt.savefig(output, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--split", default="train", choices=["train", "val"])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    output = make_preview(args.dataset, args.n, args.output, args.split)
    print(f"Preview gemt: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7.2: Tilføj auto-kald i `generate_dataset.py`**

I `scripts/02_yolo_pipeline/composite_generator/generate_dataset.py`, find linjen `print(f"\nFærdig. Output: {config.output_dir}")` og ret blokken til:

```python
    print(f"\nFærdig. Output: {config.output_dir}")

    # Auto-genereret preview
    try:
        import visualize
        preview_path = visualize.make_preview(config.output_dir, n_samples=8)
        print(f"Preview: {preview_path}")
    except Exception as e:
        print(f"Kunne ikke lave preview: {e}")
```

- [ ] **Step 7.3: Smoke-test visualize.py manuelt på mini-dataset**

Først lav et mini-dataset med 4 composites:
```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, 'scripts/02_yolo_pipeline/composite_generator')
import generate_dataset
from pathlib import Path
config = generate_dataset.Config(
    seed_image_dir=Path.home()/'Development/dataset/Corn/images',
    background_paths=[Path.home()/'Development/dataset/Corn/Belt1.png',
                       Path.home()/'Development/dataset/Corn/Belt2.png'],
    output_dir=Path('/tmp/composite_smoke_test'),
    num_composites=4, train_val_split=0.75, random_seed=42,
    canvas_size=(800, 600),
    seeds_min=10, seeds_max=20,
    gap_pixels=4, max_attempts=200,
)
generate_dataset.run(config)
"
```

Forventet: 3 train + 1 val billeder, en data.yaml, en preview.png. Tjek at `/tmp/composite_smoke_test/preview.png` faktisk eksisterer og kan åbnes.

```bash
open /tmp/composite_smoke_test/preview.png
```

- [ ] **Step 7.4: Commit**

```bash
git add scripts/02_yolo_pipeline/composite_generator/visualize.py \
        scripts/02_yolo_pipeline/composite_generator/generate_dataset.py
git commit -m "composite_generator: visualize.py + auto-preview efter generation"
```

---

## Task 8: 🚦 GATE 1 — Manuel review af MVP-output

**Dette er ikke et kode-task. Det er en manuel gennemgang inden vi går videre.**

- [ ] **Step 8.1: Kør MVP på 20 composites i fuld plade-størrelse**

```bash
.venv/bin/python scripts/02_yolo_pipeline/composite_generator/generate_dataset.py \
  --num-composites 20 \
  --output-dir /tmp/composite_mvp_gate1
```

Forventet: cirka 1-3 minutter pr. 20 composites på fuld 2000x1500. Output i `/tmp/composite_mvp_gate1/`.

- [ ] **Step 8.2: Åbn preview.png og gennemgå checklisten**

```bash
open /tmp/composite_mvp_gate1/preview.png
```

Tjek hvert punkt:

- [ ] GT-markører ligger på frø-centre, ikke ved siden af
- [ ] Ingen frø placeret på brun kant (background_loader virker)
- [ ] Frø-rotation virker — der er frø i flere orientering
- [ ] Bbox-størrelser varierer pr. frø (ikke alle ens) — verificér ved at åbne et par label-filer manuelt:
  ```bash
  head /tmp/composite_mvp_gate1/labels/train/composite_00000.txt
  ```
- [ ] Composites ser overordnet rimelige ud (ingen åbenlys bug)

- [ ] **Step 8.3: Hvis noget fejler — diagnose-trin**

| Symptom | Mulig årsag | Næste step |
|---|---|---|
| GT-markører er forskudt | Bbox-beregning i composer.py | Tjek `_rotate_stamp` og center-offsets |
| Frø ligger på brun kant | HSV-tærskel for løs i background_loader | Justér `BLUE_HUE_MIN/MAX/SAT/VAL` |
| Alle frø samme rotation | RNG ikke seedet | Tjek `rng` propagation i scatter() |
| Bbox-størrelser ens | bbox_sizes ikke beregnet fra roteret stamp | Tjek composer.render returnerer (sw, sh) fra `rotated` |

- [ ] **Step 8.4: Når Gate 1 er bestået**

Skriv noget i terminalen som "Gate 1 ✅" og forsæt til Task 9. Hvis Gate 1 fejlede gentagne gange og du ikke kan finde root cause, stop og bed om input fra Andreas.

---

## Task 9: `composer.py` — tilføj dropshadow + edge-feather + scale

Realisme-features. TDD: skriv tests først der verificerer at output ændres når features tændes.

**Files:**
- Modify: `scripts/02_yolo_pipeline/composite_generator/composer.py`
- Modify: `scripts/02_yolo_pipeline/composite_generator/tests/test_composer.py`

- [ ] **Step 9.1: Tilføj failing tests**

I `scripts/02_yolo_pipeline/composite_generator/tests/test_composer.py`, tilføj nederst:

```python
def test_render_applies_shadow():
    """Når shadow.enabled=True, skal pixels under skygge være mørkere end uden skygge."""
    bg = _make_bg(200, 200)
    stamps = [_make_stamp(40, 40)]
    placements = [Placement(center_x=100, center_y=100, rotation_deg=0, seed_index=0)]

    no_shadow = composer.render(bg, stamps, placements,
                                  shadow_config=None, feather_px=0.0)
    with_shadow = composer.render(bg, stamps, placements,
                                    shadow_config={"enabled": True,
                                                    "offset_px": [10, 10],
                                                    "blur_sigma": 3.0,
                                                    "opacity": 0.5},
                                    feather_px=0.0)
    # Pixel nedenfor og til højre for stampen skal være mørkere med skygge
    no_sh_val = no_shadow.image[125, 125, :].mean()
    with_sh_val = with_shadow.image[125, 125, :].mean()
    assert with_sh_val < no_sh_val - 5, \
        f"Skygge ikke synlig: uden={no_sh_val:.1f}, med={with_sh_val:.1f}"


def test_render_applies_feather():
    """Med feather > 0, skal kanten af paste-området være blødere."""
    bg = _make_bg(200, 200)
    stamps = [_make_stamp(40, 40)]
    placements = [Placement(center_x=100, center_y=100, rotation_deg=0, seed_index=0)]

    no_feather = composer.render(bg, stamps, placements,
                                   shadow_config=None, feather_px=0.0)
    with_feather = composer.render(bg, stamps, placements,
                                     shadow_config=None, feather_px=3.0)

    # Lige uden for stamp-grænsen: med feather skal pixelen være en blanding,
    # uden feather er pixelen ren baggrund. Tjek gradient.
    edge_no = no_feather.image[80, 100, 0]      # ren baggrund
    edge_with = with_feather.image[80, 100, 0]  # bør være blandet
    assert edge_with != edge_no, "Feather har ingen effekt"


def test_render_applies_scale_jitter():
    """Med scale > 1.0 i placement, skal bbox være større."""
    bg = _make_bg(400, 400)
    stamps = [_make_stamp(30, 30)]

    placement_small = [Placement(center_x=200, center_y=200,
                                   rotation_deg=0, seed_index=0, scale=1.0)]
    placement_large = [Placement(center_x=200, center_y=200,
                                   rotation_deg=0, seed_index=0, scale=1.5)]

    result_small = composer.render(bg, stamps, placement_small,
                                     shadow_config=None, feather_px=0.0)
    result_large = composer.render(bg, stamps, placement_large,
                                     shadow_config=None, feather_px=0.0)

    w_small, h_small = result_small.bbox_sizes[0]
    w_large, h_large = result_large.bbox_sizes[0]
    assert w_large > w_small * 1.3
    assert h_large > h_small * 1.3
```

- [ ] **Step 9.2: Kør tests, verificer FAIL**

```bash
.venv/bin/pytest scripts/02_yolo_pipeline/composite_generator/tests/test_composer.py -v
```

Forventet: 3 nye tests fejler (de 4 eksisterende fortsætter med at passere).

- [ ] **Step 9.3: Opdater `composer.py` med shadow + feather + scale**

I `scripts/02_yolo_pipeline/composite_generator/composer.py`, erstat funktionen `_rotate_stamp` og `render` med:

```python
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
    matrix[0, 2] += (new_w_rot - w) / 2
    matrix[1, 2] += (new_h_rot - h) / 2

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

    # Blur alpha-masken som basis for skyggen
    blurred = cv2.GaussianBlur(alpha_mask.astype(np.float32),
                                 ksize=(0, 0), sigmaX=blur_sigma)
    blurred = blurred / 255.0 * opacity   # 0-1

    canvas_h, canvas_w = canvas.shape[:2]
    sh, sw = blurred.shape
    x0, y0 = top_left[0] + offset_x, top_left[1] + offset_y
    x1, y1 = x0 + sw, y0 + sh

    # Clip
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(canvas_w, x1), min(canvas_h, y1)
    if cx0 >= cx1 or cy0 >= cy1:
        return

    sx0 = cx0 - x0
    sy0 = cy0 - y0
    sx1 = sx0 + (cx1 - cx0)
    sy1 = sy0 + (cy1 - cy0)

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

    apply_shadow = shadow_config is not None and shadow_config.get("enabled", False)

    for p in placements:
        stamp = seed_stamps[p.seed_index]
        scale = getattr(p, "scale", 1.0)
        rotated = _scale_and_rotate_stamp(stamp.rgba, scale, p.rotation_deg)
        sh, sw = rotated.shape[:2]

        x0 = p.center_x - sw // 2
        y0 = p.center_y - sh // 2
        x1 = x0 + sw
        y1 = y0 + sh

        # Skygge først (under frø'et)
        if apply_shadow:
            alpha_mask = rotated[:, :, 3]
            _apply_shadow(canvas, alpha_mask, (x0, y0), shadow_config)

        # Clipping mod canvas
        cx0, cy0 = max(0, x0), max(0, y0)
        cx1, cy1 = min(canvas_w, x1), min(canvas_h, y1)
        if cx0 >= cx1 or cy0 >= cy1:
            continue

        sx0 = cx0 - x0
        sy0 = cy0 - y0
        sx1 = sx0 + (cx1 - cx0)
        sy1 = sy0 + (cy1 - cy0)

        stamp_rgb = rotated[sy0:sy1, sx0:sx1, :3]
        alpha = rotated[sy0:sy1, sx0:sx1, 3].astype(np.float32)

        # Edge feather: gaussian blur af alpha
        if feather_px > 0:
            alpha = cv2.GaussianBlur(alpha, ksize=(0, 0), sigmaX=feather_px)

        alpha_norm = (alpha / 255.0)[..., np.newaxis]
        bg_slice = canvas[cy0:cy1, cx0:cx1, :].astype(np.float32)
        blended = stamp_rgb.astype(np.float32) * alpha_norm + bg_slice * (1 - alpha_norm)
        canvas[cy0:cy1, cx0:cx1, :] = np.clip(blended, 0, 255).astype(np.uint8)

        centers.append((p.center_x, p.center_y))
        bbox_sizes.append((sw, sh))

    return CompositeResult(image=canvas, centers=centers, bbox_sizes=bbox_sizes)
```

Slet den gamle `_rotate_stamp`-funktion (den er erstattet af `_scale_and_rotate_stamp`).

- [ ] **Step 9.4: Kør alle composer-tests, verificer PASS**

```bash
.venv/bin/pytest scripts/02_yolo_pipeline/composite_generator/tests/test_composer.py -v
```

Forventet: alle 7 tests PASS.

- [ ] **Step 9.5: Opdater `generate_dataset.py` så den bruger features fra config**

I `generate_dataset.py`, opdater `Config` dataclass (tilføj felter) og `from_yaml`-metoden samt `run()`-funktionen:

I `Config`-dataclass, tilføj:
```python
    shadow_config: Optional[dict] = None
    feather_px: float = 0.0
    scale_jitter: tuple[float, float] = (1.0, 1.0)
```

I `from_yaml`, tilføj efter de eksisterende felter:
```python
            shadow_config=data["shadow"],
            feather_px=data["edge_feather_px"],
            scale_jitter=tuple(data["seed_scale_jitter"]),
```

I `run()`, find linjen:
```python
        placements = packing.scatter(
            canvas_size=config.canvas_size, seed_stamps=stamps,
            gap_pixels=config.gap_pixels, max_attempts=config.max_attempts,
            rng=rng)
```

Tilføj direkte efter scale-jitter til hvert placement:
```python
        for pl in placements:
            pl.scale = float(rng.uniform(*config.scale_jitter))
```

Erstat composer-kaldet:
```python
        result = composer.render(bg, stamps, placements,
                                  shadow_config=config.shadow_config,
                                  feather_px=config.feather_px)
```

- [ ] **Step 9.6: Commit**

```bash
git add scripts/02_yolo_pipeline/composite_generator/composer.py \
        scripts/02_yolo_pipeline/composite_generator/generate_dataset.py \
        scripts/02_yolo_pipeline/composite_generator/tests/test_composer.py
git commit -m "composite_generator: dropshadow + edge-feather + scale-jitter (TDD)"
```

---

## Task 10: `packing.py` — dense-strategi

Som scatter men med gap_pixels=0 (eller næsten 0). Tæt pakkede områder.

**Files:**
- Modify: `scripts/02_yolo_pipeline/composite_generator/packing.py`
- Modify: `scripts/02_yolo_pipeline/composite_generator/tests/test_packing.py`

- [ ] **Step 10.1: Skriv failing test**

I `scripts/02_yolo_pipeline/composite_generator/tests/test_packing.py`, tilføj:

```python
def test_dense_packs_tighter_than_scatter():
    """dense-strategi placerer frø tættere end scatter med samme antal."""
    stamps = [_make_stamp(25, 25, i) for i in range(15)]

    p_scatter = packing.scatter(
        canvas_size=(400, 400), seed_stamps=stamps,
        gap_pixels=10, max_attempts=200,
        rng=np.random.default_rng(0))
    p_dense = packing.dense(
        canvas_size=(400, 400), seed_stamps=stamps,
        max_attempts=200, rng=np.random.default_rng(0))

    # Gennemsnitlig minimum-afstand mellem nabofrø
    def avg_min_dist(placements):
        if len(placements) < 2:
            return float('inf')
        dists = []
        for i, p1 in enumerate(placements):
            min_d = min(
                ((p1.center_x - p2.center_x) ** 2 +
                 (p1.center_y - p2.center_y) ** 2) ** 0.5
                for j, p2 in enumerate(placements) if i != j)
            dists.append(min_d)
        return sum(dists) / len(dists)

    assert avg_min_dist(p_dense) < avg_min_dist(p_scatter)
```

- [ ] **Step 10.2: Kør test, verificer FAIL**

```bash
.venv/bin/pytest scripts/02_yolo_pipeline/composite_generator/tests/test_packing.py::test_dense_packs_tighter_than_scatter -v
```

Forventet: FAIL med `AttributeError: module 'packing' has no attribute 'dense'`.

- [ ] **Step 10.3: Tilføj `dense()` i `packing.py`**

I `scripts/02_yolo_pipeline/composite_generator/packing.py`, tilføj efter `scatter()`:

```python
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
```

- [ ] **Step 10.4: Kør tests, verificer PASS**

```bash
.venv/bin/pytest scripts/02_yolo_pipeline/composite_generator/tests/test_packing.py -v
```

Forventet: alle tests PASS (inkl. den nye).

- [ ] **Step 10.5: Commit**

```bash
git add scripts/02_yolo_pipeline/composite_generator/packing.py \
        scripts/02_yolo_pipeline/composite_generator/tests/test_packing.py
git commit -m "composite_generator: packing.dense() (gap=0 variant af scatter)"
```

---

## Task 11: `packing.py` — t_config-strategi

Eksplicit T/L-konfigurationer. Sample 2-3 frø placeret så de rør hinanden i en T-eller L-form.

**Files:**
- Modify: `scripts/02_yolo_pipeline/composite_generator/packing.py`
- Modify: `scripts/02_yolo_pipeline/composite_generator/tests/test_packing.py`

- [ ] **Step 11.1: Skriv failing test**

I `scripts/02_yolo_pipeline/composite_generator/tests/test_packing.py`, tilføj:

```python
def test_t_config_creates_touching_seeds():
    """t_config returnerer en cluster af 2-3 frø der rør hinanden."""
    stamps = [_make_stamp(30, 30, i) for i in range(3)]

    placements = packing.t_config(
        canvas_size=(400, 400), seed_stamps=stamps,
        rng=np.random.default_rng(0))

    assert 2 <= len(placements) <= 3

    # Mindst ét par skal være tæt på hinanden (distance < sum af radier + lille gap)
    if len(placements) >= 2:
        radius = 0.5 * (30 * 30 + 30 * 30) ** 0.5  # ~21
        min_dist = min(
            ((p1.center_x - p2.center_x) ** 2 +
             (p1.center_y - p2.center_y) ** 2) ** 0.5
            for i, p1 in enumerate(placements)
            for p2 in placements[i+1:])
        # Frø rør hinanden = distance < 2*radius
        assert min_dist < 2 * radius + 5


def test_t_config_perpendicular_orientation():
    """For ekstra realisme: andet frø står ~vinkelret på første."""
    stamps = [_make_stamp(40, 15, i) for i in range(3)]  # aflange majs

    placements = packing.t_config(
        canvas_size=(400, 400), seed_stamps=stamps,
        rng=np.random.default_rng(0))

    if len(placements) >= 2:
        rot_diff = abs(placements[0].rotation_deg - placements[1].rotation_deg) % 180
        # Vinkelret = 90° ± 30° tolerance
        assert 60 < rot_diff < 120, f"Ikke vinkelret: {rot_diff}°"
```

- [ ] **Step 11.2: Kør tests, verificer FAIL**

```bash
.venv/bin/pytest scripts/02_yolo_pipeline/composite_generator/tests/test_packing.py::test_t_config_creates_touching_seeds -v
```

Forventet: FAIL med `AttributeError: module 'packing' has no attribute 't_config'`.

- [ ] **Step 11.3: Tilføj `t_config()` i `packing.py`**

I `scripts/02_yolo_pipeline/composite_generator/packing.py`, tilføj efter `dense()`:

```python
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
```

- [ ] **Step 11.4: Kør tests, verificer PASS**

```bash
.venv/bin/pytest scripts/02_yolo_pipeline/composite_generator/tests/test_packing.py -v
```

Forventet: alle tests PASS.

- [ ] **Step 11.5: Commit**

```bash
git add scripts/02_yolo_pipeline/composite_generator/packing.py \
        scripts/02_yolo_pipeline/composite_generator/tests/test_packing.py
git commit -m "composite_generator: packing.t_config() med vinkelrette frø-par"
```

---

## Task 12: `packing.py` — `compute()` dispatcher der mixer strategier

Top-level `compute()` der tager strategy_mix dict og blander de tre strategier.

**Files:**
- Modify: `scripts/02_yolo_pipeline/composite_generator/packing.py`
- Modify: `scripts/02_yolo_pipeline/composite_generator/tests/test_packing.py`
- Modify: `scripts/02_yolo_pipeline/composite_generator/generate_dataset.py`

- [ ] **Step 12.1: Skriv failing test**

I `scripts/02_yolo_pipeline/composite_generator/tests/test_packing.py`, tilføj:

```python
def test_compute_blends_strategies():
    """compute() returnerer placements der reflekterer strategy_mix."""
    stamps = [_make_stamp(25, 25, i) for i in range(30)]

    # Pure scatter
    p_scatter_only = packing.compute(
        canvas_size=(800, 600), seed_stamps=stamps,
        strategy_mix={"scatter": 1.0, "dense": 0.0, "t_config": 0.0},
        gap_pixels=4, max_attempts=200,
        rng=np.random.default_rng(0))
    # Pure dense
    p_dense_only = packing.compute(
        canvas_size=(800, 600), seed_stamps=stamps,
        strategy_mix={"scatter": 0.0, "dense": 1.0, "t_config": 0.0},
        gap_pixels=4, max_attempts=200,
        rng=np.random.default_rng(0))

    # Dense passer flere frø ind end scatter på samme canvas
    assert len(p_dense_only) >= len(p_scatter_only)


def test_compute_returns_non_empty_for_reasonable_input():
    """Standard config skal returnere mindst 10 placements."""
    stamps = [_make_stamp(20, 30, i) for i in range(60)]
    p = packing.compute(
        canvas_size=(1000, 800), seed_stamps=stamps,
        strategy_mix={"scatter": 0.5, "dense": 0.3, "t_config": 0.2},
        gap_pixels=4, max_attempts=200,
        rng=np.random.default_rng(0))
    assert len(p) > 10
```

- [ ] **Step 12.2: Kør tests, verificer FAIL**

```bash
.venv/bin/pytest scripts/02_yolo_pipeline/composite_generator/tests/test_packing.py::test_compute_blends_strategies -v
```

Forventet: FAIL med `AttributeError: module 'packing' has no attribute 'compute'`.

- [ ] **Step 12.3: Tilføj `compute()` i `packing.py`**

I `scripts/02_yolo_pipeline/composite_generator/packing.py`, tilføj efter `t_config()`:

```python
def compute(
    canvas_size: tuple[int, int],
    seed_stamps: list[SeedStamp],
    strategy_mix: dict[str, float],
    gap_pixels: int,
    max_attempts: int,
    rng: np.random.Generator,
) -> list[Placement]:
    """Top-level dispatcher: blandt strategy_mix vægt for hvert frø.

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

    # Step 1: hvor mange frø skal i t-config clusters?
    t_weight = strategy_mix.get("t_config", 0.0)
    n_t_config = int(n_total * t_weight)
    n_remaining = n_total - n_t_config

    all_placements: list[Placement] = []
    used_stamps_indices = set()

    # Step 1a: T-config clusters (2-3 frø ad gangen)
    seed_idx = 0
    while n_t_config >= 2 and seed_idx + 1 < n_total:
        cluster_stamps = seed_stamps[seed_idx:seed_idx + 3]
        sub_placements = t_config(canvas_size, cluster_stamps, rng)
        # Remap seed_index til globalt
        for sp in sub_placements:
            sp.seed_index = seed_idx + sp.seed_index
            all_placements.append(sp)
            used_stamps_indices.add(sp.seed_index)
        consumed = len(sub_placements)
        seed_idx += consumed if consumed > 0 else 1
        n_t_config -= consumed

    # Step 2: scatter vs dense for de resterende
    remaining_stamps_with_idx = [
        (i, s) for i, s in enumerate(seed_stamps)
        if i not in used_stamps_indices
    ]

    scatter_weight = strategy_mix.get("scatter", 0.0)
    dense_weight = strategy_mix.get("dense", 0.0)
    total_w = scatter_weight + dense_weight
    if total_w == 0:
        return all_placements

    n_scatter = int(len(remaining_stamps_with_idx) * scatter_weight / total_w)

    scatter_idxs = remaining_stamps_with_idx[:n_scatter]
    dense_idxs = remaining_stamps_with_idx[n_scatter:]

    # Brug en "fake" SeedStamp-liste med oprindelige indices
    if scatter_idxs:
        scatter_stamps = [s for _, s in scatter_idxs]
        scatter_placements = scatter(canvas_size, scatter_stamps,
                                       gap_pixels, max_attempts, rng)
        for sp in scatter_placements:
            sp.seed_index = scatter_idxs[sp.seed_index][0]
            all_placements.append(sp)

    if dense_idxs:
        dense_stamps = [s for _, s in dense_idxs]
        dense_placements = dense(canvas_size, dense_stamps,
                                   max_attempts, rng)
        for dp in dense_placements:
            dp.seed_index = dense_idxs[dp.seed_index][0]
            all_placements.append(dp)

    return all_placements
```

- [ ] **Step 12.4: Kør tests, verificer PASS**

```bash
.venv/bin/pytest scripts/02_yolo_pipeline/composite_generator/tests/test_packing.py -v
```

Forventet: alle tests PASS.

- [ ] **Step 12.5: Opdater `generate_dataset.py` til at bruge `compute()`**

I `generate_dataset.py`, tilføj `strategy_mix` til `Config`:
```python
    strategy_mix: dict = None
```

I `from_yaml`:
```python
            strategy_mix=data["packing_strategy_mix"],
```

I `run()`, erstat scatter-kaldet med:
```python
        placements = packing.compute(
            canvas_size=config.canvas_size, seed_stamps=stamps,
            strategy_mix=config.strategy_mix,
            gap_pixels=config.gap_pixels, max_attempts=config.max_attempts,
            rng=rng)
```

- [ ] **Step 12.6: Commit**

```bash
git add scripts/02_yolo_pipeline/composite_generator/packing.py \
        scripts/02_yolo_pipeline/composite_generator/generate_dataset.py \
        scripts/02_yolo_pipeline/composite_generator/tests/test_packing.py
git commit -m "composite_generator: packing.compute() blender scatter/dense/t_config"
```

---

## Task 13: 🚦 GATE 2 — Manuel review af fuld pipeline

**Anden manuel gate. Tjek at realisme-features faktisk virker.**

- [ ] **Step 13.1: Kør fuld pipeline på 20 composites**

```bash
.venv/bin/python scripts/02_yolo_pipeline/composite_generator/generate_dataset.py \
  --num-composites 20 \
  --output-dir /tmp/composite_gate2
```

- [ ] **Step 13.2: Åbn preview.png og gennemgå Gate 2-checklisten**

```bash
open /tmp/composite_gate2/preview.png
```

Tjek hvert punkt:

- [ ] Skygger ses naturlige (ikke for hårde, ikke usynlige)
- [ ] Ingen synlige hårde alpha-kanter (edge-feather virker)
- [ ] T/L-konfigurationer er reelt til stede (find mindst én)
- [ ] Dense-pakning giver tæt-pakkede områder (find mindst én)
- [ ] Strategi-mix matcher config (visuel sanity, ikke kvantitativ)
- [ ] GT-markører ligger stadig på frø-centre

- [ ] **Step 13.3: Hvis noget fejler — justér config**

| Symptom | Fix |
|---|---|
| Skygger for stærke | Reducer `shadow.opacity` i config.yaml (0.35 → 0.20) |
| Skygger for svage | Øg `opacity` (0.35 → 0.50) |
| Kanter stadig hårde | Øg `edge_feather_px` (1.5 → 3.0) |
| Ingen T-konfig synlig | Øg `packing_strategy_mix.t_config` (0.20 → 0.35) |
| For meget dense, ingen scatter | Justér `strategy_mix` |

Efter ændringer i config.yaml, kør Step 13.1 igen og inspicér nyt preview.

- [ ] **Step 13.4: Når Gate 2 er bestået**

Du er klar til fuld produktion. Hvis Gate 2 fejlede gentagne gange, stop og bed Andreas om input.

---

## Task 14: Fuld kørsel — 2000 composites

Produktions-kørsel. Forventet tid: 1-3 timer på Mac med 12.384 frø.

- [ ] **Step 14.1: Tjek disk-plads**

```bash
df -h ~/Development/school/fcm_project_export
```

Forventet output: mindst 25 GB ledig (vi har brug for ~10-15 GB til 2000 PNG'er).

- [ ] **Step 14.2: Kør generation (kan tage 1-3 timer)**

```bash
.venv/bin/python scripts/02_yolo_pipeline/composite_generator/generate_dataset.py \
  2>&1 | tee /tmp/composite_v1_run.log
```

Bemærk: hvis du afbryder midt i (Ctrl-C), kan du genstarte samme kommando — resume-funktionen springer eksisterende output over.

- [ ] **Step 14.3: Verificer output**

```bash
ls data/composite_v1/images/train/ | wc -l        # forventet: 1600
ls data/composite_v1/images/val/ | wc -l          # forventet: 400
ls data/composite_v1/labels/train/ | wc -l        # forventet: 1600
cat data/composite_v1/data.yaml                   # tjek struktur
du -sh data/composite_v1/                         # forventet: 5-20 GB
open data/composite_v1/preview.png                # endelig sanity check
```

- [ ] **Step 14.4: Commit metadata (ikke billederne — de er for store til git)**

Tilføj først til `.gitignore`:

```bash
echo "data/composite_*/images/" >> .gitignore
echo "data/composite_*/labels/" >> .gitignore
```

Så commit metadata + preview:

```bash
git add .gitignore data/composite_v1/data.yaml data/composite_v1/generation_metadata.json data/composite_v1/preview.png
git commit -m "data: composite_v1 generation metadata + preview (billeder gitignored)"
```

---

## Task 15: Arkiver gammel ellipse-pipeline

Flyt `01b_realistic_synthetic_data.py` og `02_convert_to_yolo_format.py` til arkiv-mappe.

**Files:**
- Move: `scripts/02_yolo_pipeline/01b_realistic_synthetic_data.py` → `scripts/02a_archived_ellipse_synthetic/`
- Move: `scripts/02_yolo_pipeline/02_convert_to_yolo_format.py` → `scripts/02a_archived_ellipse_synthetic/`
- Create: `scripts/02a_archived_ellipse_synthetic/README.md`

- [ ] **Step 15.1: Lav arkiv-mappe og flyt filer**

```bash
mkdir -p scripts/02a_archived_ellipse_synthetic
git mv scripts/02_yolo_pipeline/01b_realistic_synthetic_data.py \
       scripts/02a_archived_ellipse_synthetic/
git mv scripts/02_yolo_pipeline/02_convert_to_yolo_format.py \
       scripts/02a_archived_ellipse_synthetic/
```

- [ ] **Step 15.2: Skriv README.md i arkiv-mappen**

Indhold til `scripts/02a_archived_ellipse_synthetic/README.md`:

```markdown
# Arkiveret: Ellipse-baseret syntetisk pipeline

Denne mappe indeholder den oprindelige syntetiske data-pipeline (v0)
som tegnede aflange ellipser på ensartet grå baggrund. Vi har erstattet
den med `composite_generator/` som bygger composites fra rigtige
Videometer-frø-PNG'er.

## Hvorfor bevaret?

- **Eksamens-fortælling:** progression fra "ellipser når der ikke var
  rigtige data" → "composites når frø-PNG'er kom" → "rigtige Videometer-
  scener efter besøg på fabrik". Den narrative bro er værd at vise.
- **Baseline-sammenligning:** v1/v2/v3 modellerne er trænet på ellipse-
  data. Hvis composite-trænede modeller klarer sig dårligere, kan vi
  diagnosticere ved at sammenligne mod denne pipeline.

## Filer

- `01b_realistic_synthetic_data.py` — generér 300 256×256 syntetiske billeder
- `02_convert_to_yolo_format.py` — konvertér til YOLO-format dataset

## Ikke aktivt brugt

Hvis du genaktiverer denne pipeline, skal du være opmærksom på at
output-stier kan kollidere med composite_generator output.
```

- [ ] **Step 15.3: Tjek at intet andet script peger på de gamle filer**

```bash
grep -rn "01b_realistic_synthetic_data\|02_convert_to_yolo_format" \
  scripts/ docs/ README.md --include="*.py" --include="*.md" \
  | grep -v "02a_archived_ellipse_synthetic"
```

Forventet output: ingen matches (eller kun harmløse historiske referencer i `docs/`).

Hvis der er referencer (fx i `scripts/03_train_yolo.py`), notér dem og ret stier i næste step.

- [ ] **Step 15.4: Commit**

```bash
git add scripts/02a_archived_ellipse_synthetic/
git commit -m "arkivér: flyt ellipse-pipeline til scripts/02a_archived_ellipse_synthetic/"
```

---

## Task 16: Træn v4-model på composites

Genbrug eksisterende `03_train_yolo.py` med ny data.yaml.

**Files:**
- Modify: `scripts/02_yolo_pipeline/03_train_yolo.py` (eventuel sti-opdatering)
- Create: `models/v4_composite/notes.md`
- Modify: `models/active` symlink

- [ ] **Step 16.1: Se hvad 03_train_yolo.py forventer**

```bash
head -50 scripts/02_yolo_pipeline/03_train_yolo.py
```

Notér hvor `data.yaml`-stien er hardkodet eller læst fra arg.

- [ ] **Step 16.2: Træn på Mac MPS eller Colab**

**Option A: Lokalt på Mac MPS** (langsommere men ingen upload):

```bash
.venv/bin/python -c "
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model.train(
    data='data/composite_v1/data.yaml',
    epochs=80,
    imgsz=1280,           # halvdelen af 2000 — kompromis MPS-memory vs detalje
    batch=4,              # juster ned hvis MPS løber tør for hukommelse
    patience=15,
    device='mps',
    project='runs/train',
    name='v4_composite',
    flipud=0.5, fliplr=0.5, degrees=180.0,
    mosaic=1.0, mixup=0.15, copy_paste=0.3,
)
print('Done. Best vægte:', 'runs/train/v4_composite/weights/best.pt')
"
```

**Option B: Colab T4 GPU** (hurtigere, ingen NDA-problemer per Andreas):
- Upload `data/composite_v1.zip` til Colab
- Brug ny notebook `scripts/04_colab/v4_composite/train.ipynb` (kopi af v3's, justér paths)

- [ ] **Step 16.3: Verificer træningen producerede en best.pt**

```bash
ls -la runs/train/v4_composite/weights/best.pt
```

Forventet: filen eksisterer, ~5.4 MB (YOLOv11n typisk).

- [ ] **Step 16.4: Kopier modellen til models/v4_composite/**

```bash
mkdir -p models/v4_composite
cp runs/train/v4_composite/weights/best.pt models/v4_composite/best.pt
```

- [ ] **Step 16.5: Skriv notes.md med resultater**

Indhold til `models/v4_composite/notes.md` (udfyld faktiske tal fra `runs/train/v4_composite/results.csv`):

```markdown
# v4_composite

**Trænet:** 2026-05-XX, [Mac MPS / Colab T4]
**Arkitektur:** YOLOv11n
**Ultralytics:** 8.4.47

## Data (2000 composites, 2000×1500, 80/20 split)

Composite-pipeline fra `scripts/02_yolo_pipeline/composite_generator/`:
- Frø-pool: 12.384 alpha-stansede majs fra Videometer
- Baggrunde: Belt1.png + Belt2.png med HSV-baseret blå-zone-detection
- Strategi-mix: 50% scatter / 30% dense / 20% t_config
- Skygge: offset (3,3), blur=4.0, opacity=0.35
- Edge-feather: 1.5 px
- Frø-skala-jitter: ±15%

## Trænings-config

- imgsz: 1280 (downscale fra 2000)
- epochs: 80, patience: 15
- batch: [4/16 afhængig af device]
- augmentations: flipud/lr=0.5, degrees=180, mosaic, mixup, copy_paste

## Metrics (held-out 400 composites)

| Metric | Værdi | Krav |
|---|---|---|
| Precision | [TBD efter træning] | > 0.5 |
| Recall | [TBD] | > 0.3 |
| mAP@50 | [TBD] | > 0.2 |
| mAP@50-95 | [TBD] | (bonus) |

## v3 vs v4 sammenligning

Hvis relevant, sammenligning af edge cases mellem v3 (syntetiske ellipser)
og v4 (composites med rigtige frø) — vis at v4 generaliserer bedre.

## Limitations

- Kvantitative metrics gælder kun composite-verdenen.
- Generalisering til rigtige Videometer-scener er kun verificeret kvalitativt
  (visuel inspektion af forudsigelser).
- Aggressive YOLO-augmentations (mosaic, mixup) mitigerer noget overfit-risiko.
```

- [ ] **Step 16.6: Skift active-model symlink til v4**

```bash
cd models
ln -sfn v4_composite active
cd ..
ls -la models/active
```

Forventet: `active -> v4_composite`.

- [ ] **Step 16.7: Smoke-test demoen med ny model**

```bash
.venv/bin/streamlit run scripts/03_demo/app.py --server.headless true &
sleep 5
curl -s http://localhost:8501 | head -5
kill %1
```

Forventet: ingen import-fejl, Streamlit svarer. (Hvis du vil bruge demoen interaktivt, kør uden `--headless`.)

- [ ] **Step 16.8: Commit modellen + notes**

```bash
git add models/v4_composite/ models/active
git commit -m "models: v4_composite trænet på 2000 composite-scener (YOLOv11n)"
```

---

## Self-Review

Inden vi afslutter planen, lad mig tjekke spec-coverage:

| Spec-sektion | Implementeret i task |
|---|---|
| §1 Kontekst | Task 1-2 (kontekst i README) |
| §2 Mål | Task 1-14 |
| §3.1 Tilgang B | Task 9 (shadow+feather) + Task 11-12 (T-config) |
| §3.2 Held-out val | Task 6 (train_val_split=0.80) |
| §3.3 Fuld plade-størrelse | Task 1 (config canvas_size=[2000,1500]) |
| §3.4 YOLOv11n A/B YOLO26 | Task 16 (YOLOv11n nu, A/B udskudt) |
| §3.5 tight bbox | Task 5 (bbox_sizes fra rotated stamp) |
| §4 Arkitektur | Task 1-7 strukturen |
| §5.1 seed_loader | Task 2 |
| §5.2 background_loader | Task 3 |
| §5.3 packing | Task 4 (scatter), 10 (dense), 11 (t_config), 12 (compute) |
| §5.4 composer | Task 5 (MVP), 9 (shadow+feather+scale) |
| §5.5 generate_dataset | Task 6 + 9.5 + 12.5 |
| §5.6 visualize | Task 7 |
| §6 config.yaml | Task 1 |
| §7 Dataflow | Task 6, 9.5, 12.5 |
| §8 Fejlhåndtering | Task 2 (skip-on-error), Task 6 (atomic write + resume) |
| §9 Test-strategi | Task 2-12 (TDD), Task 8 + 13 (manual gates) |
| §10 Risici | Task 14 (disk check), Task 16 (augmentations som mitigation) |
| §11 Implementerings-fasering | Hele dette dokument |

**Gaps:** Ingen kritiske. A/B-test mod YOLO26n (§3.4 / §11 step 7) er bevidst udskudt til efter v4 er trænet og evalueret — det er en separat opgave.

**Type-konsistens:**
- `SeedStamp` defineret i Task 2, brugt konsistent i Task 4, 5, 10, 11, 12
- `Placement` defineret i Task 4 (med `scale=1.0` default), brugt i Task 5, 11, 12
- `CompositeResult` defineret i Task 5, brugt i Task 6
- `Config` defineret i Task 6, udvidet i Task 9.5 og 12.5
- `BackgroundPool` defineret i Task 3, brugt i Task 6

**Placeholder-scan:** Søgt for "TBD" — kun ét sted i `models/v4_composite/notes.md` der bevidst skal udfyldes efter træning. Det er ikke en plan-failure men data der først kendes ved kørsel.

---

## Bagefter (uden for denne plan)

- **A/B mod YOLO26n:** ny task — kopi af Task 16 med `YOLO('yolo26n.pt')`, sammenlign mod v4.
- **Manuel evaluering på rigtige Videometer-scener:** når Andreas besøger fabrikken, tag 5-10 testbilleder, kør v4 på dem, lav et lille kvalitativt rapport.
- **Demo-opdatering:** Streamlit-demoen peger nu på v4 — eventuel UI-justering hvis nye edge cases skal vises.
