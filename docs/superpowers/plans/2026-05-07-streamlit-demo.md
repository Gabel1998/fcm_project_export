# Streamlit-demo Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bygge et Streamlit-baseret UI-lag oven på den trænede YOLOv11n-model, der lader Videometer-folk justere fire parametre og se ground truth vs prediction live, klar til demo 2026-05-08.

**Architecture:** Streamlit-app i ny mappe `scripts/03_demo/`. Tre rene Python-moduler (`synthetic.py`, `landmarks.py`, `info_texts.py`) som er TDD-testede; Streamlit-appen `app.py` glues dem sammen med smoke-tests. Modellen loades én gang via `@st.cache_resource`. Eksisterende pipeline-scripts forbliver urørte.

**Tech Stack:** Python 3.11, Streamlit ≥1.30, NumPy, PIL, Ultralytics (YOLOv11n via PyTorch MPS), pytest til unit-tests.

**Bemærk om commits:** Projektet er ikke et git-repo endnu. Spring `git add`/`git commit`-trinene over hvis du ikke har kørt `git init`. Planen virker alligevel.

---

## File Structure

```
scripts/03_demo/
├── __init__.py            # tom — gør mappen til en Python-pakke
├── synthetic.py           # generate_image(num_seeds, img_size, noise_preset, seed_size_preset)
├── landmarks.py           # match_landmarks() + compute_metrics()
├── info_texts.py          # dict med alle danske tooltip-tekster
├── app.py                 # Streamlit entry point
├── styles.css             # custom CSS for visuel polish
└── tests/
    ├── __init__.py
    ├── test_synthetic.py
    └── test_landmarks.py
```

**Ansvar pr. fil:**
- `synthetic.py` — ren funktion: parametre ind, (billede, ground truth-centre, mm/pixel) ud. Ingen Streamlit-afhængigheder.
- `landmarks.py` — ren funktion: matcher predicted vs ground truth, beregner P/R/F1/distance-stats. Ingen Streamlit-afhængigheder.
- `info_texts.py` — kun et dict. Centralt sted for alle danske forklaringer.
- `app.py` — al Streamlit-logik (layout, caching, kald til ovenstående). Eneste fil med UI-kode.
- `styles.css` — visuelle justeringer indlæst via `st.markdown(..., unsafe_allow_html=True)`.

---

## Task 1: Setup + dependencies

**Files:**
- Create: `scripts/03_demo/__init__.py`
- Create: `scripts/03_demo/tests/__init__.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Opret mappestruktur**

```bash
mkdir -p scripts/03_demo/tests
touch scripts/03_demo/__init__.py
touch scripts/03_demo/tests/__init__.py
```

- [ ] **Step 2: Tilføj streamlit + pytest til requirements**

Åbn `requirements.txt` og tilføj nederst (før de udkommenterede linjer):

```
streamlit>=1.30.0
pytest>=8.0.0
```

- [ ] **Step 3: Installer dependencies i venv**

Run: `.venv/bin/pip install streamlit pytest`
Expected: `Successfully installed streamlit-1.x.x ... pytest-8.x.x`

- [ ] **Step 4: Verificér Streamlit + pytest virker**

Run: `.venv/bin/streamlit --version && .venv/bin/pytest --version`
Expected: Begge skal printe versionsnumre uden fejl.

- [ ] **Step 5: Commit (spring over hvis ikke git-repo)**

```bash
git add scripts/03_demo/ requirements.txt
git commit -m "demo: setup pakkestruktur og dependencies"
```

---

## Task 2: synthetic.py — refaktorer datageneratoren (TDD)

**Files:**
- Create: `scripts/03_demo/synthetic.py`
- Create: `scripts/03_demo/tests/test_synthetic.py`

**Mål:** Tag den eksisterende generator-logik fra `scripts/02_yolo_pipeline/01b_realistic_synthetic_data.py` og pak den som en ren funktion `generate_image()` der tager 4 parametre.

- [ ] **Step 1: Skriv testen først**

Mappenavnet `03_demo` starter med et ciffer, så vi kan ikke importere det som en almindelig Python-pakke (`import scripts.03_demo` fejler). Vi bruger derfor et `sys.path`-trick: tilføj demo-mappen direkte til Python-stien, og importér `synthetic` som top-level modul.

Opret `scripts/03_demo/tests/test_synthetic.py`:

```python
"""Tests for synthetic data generator."""
import sys
from pathlib import Path

import numpy as np
import pytest

# Tilføj demo-mappen til sys.path så vi kan importere modulet direkte
DEMO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO_DIR))

import synthetic  # noqa: E402


def test_default_parametre_giver_korrekt_shape():
    """Default-kald skal returnere et 256x256 billede og en liste af centre."""
    img, centers, mm_per_pixel = synthetic.generate_image(
        num_seeds=58, img_size=256, noise_preset="kalibreret",
        seed_size_preset="normal", random_seed=42,
    )
    assert img.shape == (256, 256), f"forventet (256,256), fik {img.shape}"
    assert img.dtype == np.uint8, f"forventet uint8, fik {img.dtype}"
    assert isinstance(centers, list)
    assert all(len(c) == 2 for c in centers), "hvert center skal være (x, y)"
    assert mm_per_pixel == 0.05


def test_lille_billede_virker():
    """128x128 skal også fungere."""
    img, centers, _ = synthetic.generate_image(
        num_seeds=20, img_size=128, noise_preset="kalibreret",
        seed_size_preset="normal", random_seed=1,
    )
    assert img.shape == (128, 128)
    # mindst nogle frø placeret (rejection sampling kan dropper nogle)
    assert len(centers) >= 5


def test_random_seed_er_deterministisk():
    """Samme random_seed → samme output."""
    img1, c1, _ = synthetic.generate_image(
        num_seeds=30, img_size=256, noise_preset="kalibreret",
        seed_size_preset="normal", random_seed=123,
    )
    img2, c2, _ = synthetic.generate_image(
        num_seeds=30, img_size=256, noise_preset="kalibreret",
        seed_size_preset="normal", random_seed=123,
    )
    assert np.array_equal(img1, img2)
    assert c1 == c2


def test_centre_er_inden_for_billedet():
    """Alle centre skal ligge inden for billed-grænserne."""
    img, centers, _ = synthetic.generate_image(
        num_seeds=50, img_size=256, noise_preset="kalibreret",
        seed_size_preset="normal", random_seed=7,
    )
    for x, y in centers:
        assert 0 <= x < 256, f"x={x} ude af billedet"
        assert 0 <= y < 256, f"y={y} ude af billedet"


def test_invalid_preset_giver_fejl():
    """Ukendt preset-navn skal kaste ValueError."""
    with pytest.raises(ValueError):
        synthetic.generate_image(
            num_seeds=30, img_size=256, noise_preset="DETTE_FINDES_IKKE",
            seed_size_preset="normal", random_seed=1,
        )
```

- [ ] **Step 2: Kør testen — den skal fejle**

Run: `.venv/bin/pytest scripts/03_demo/tests/test_synthetic.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'synthetic'`

- [ ] **Step 3: Implementér synthetic.py**

Opret `scripts/03_demo/synthetic.py`:

```python
"""Syntetisk datagenerator — refaktoreret fra 01b_realistic_synthetic_data.py.

En ren funktion: parametre ind, (billede, centre, mm/pixel) ud.
Ingen Streamlit-afhængigheder.
"""
from typing import Literal

import numpy as np

# Mapping fra preset-navn til generator-værdier
NOISE_PRESETS = {
    "kalibreret": 0.02,   # som model blev trænet på
    "realistisk": 0.05,
    "svært":      0.10,
}

SEED_SIZE_PRESETS = {
    # (length_min, length_max, width_min, width_max) i pixels
    "mindre": (8, 12, 3, 5),
    "normal": (12, 18, 5, 7),  # som model blev trænet på
    "større": (18, 24, 7, 9),
}

# Konstanter (samme som det model'en så under træning)
BACKGROUND_REFLECTANCE = 0.85
SEED_REFLECTANCE_MIN = 0.20
SEED_REFLECTANCE_MAX = 0.45
PACKING_GAP_PIXELS = 2
MAX_PLACEMENT_ATTEMPTS = 200
MM_PER_PIXEL = 0.05


def _ellipses_overlap(e1, e2, gap):
    """Konservativ overlap-tjek: afstand mellem centre vs sum af radier."""
    cx1, cy1, L1, _, _ = e1
    cx2, cy2, L2, _, _ = e2
    r1, r2 = L1 / 2.0, L2 / 2.0
    dx, dy = cx1 - cx2, cy1 - cy2
    return np.sqrt(dx * dx + dy * dy) < (r1 + r2 + gap)


def _draw_ellipse(image, cx, cy, length, width, angle_deg, color):
    """Tegn udfyldt ellipse direkte i numpy-arrayet."""
    a, b = length / 2.0, width / 2.0
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    H, W = image.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    x_shifted, y_shifted = xx - cx, yy - cy
    x_rot = cos_a * x_shifted + sin_a * y_shifted
    y_rot = -sin_a * x_shifted + cos_a * y_shifted

    inside = (x_rot / a) ** 2 + (y_rot / b) ** 2 < 1.0
    image[inside] = color


def generate_image(
    num_seeds: int,
    img_size: int,
    noise_preset: Literal["kalibreret", "realistisk", "svært"],
    seed_size_preset: Literal["mindre", "normal", "større"],
    random_seed: int | None = None,
):
    """Generér ét syntetisk billede der ligner Videometer-data.

    Args:
        num_seeds: antal frø generatoren forsøger at placere (rejection sampling
            kan dropper nogle hvis pladen er fuld).
        img_size: kvadratisk billed-størrelse i pixels (fx 128 eller 256).
        noise_preset: navn fra NOISE_PRESETS — styrer reflektans-støj.
        seed_size_preset: navn fra SEED_SIZE_PRESETS — styrer frø-dimensioner.
        random_seed: None = ny tilfældig, ellers determistisk output.

    Returns:
        Tuple (img_uint8, centers, mm_per_pixel):
            img_uint8: (H, W) uint8 array i [0, 255]
            centers: liste af (x, y) tupler — ground truth centre
            mm_per_pixel: skala-faktor (typisk 0.05)

    Raises:
        ValueError: hvis preset-navn ikke findes.
    """
    # Validering først så fejlen er tydelig
    if noise_preset not in NOISE_PRESETS:
        raise ValueError(
            f"noise_preset '{noise_preset}' ukendt. "
            f"Brug en af: {list(NOISE_PRESETS.keys())}"
        )
    if seed_size_preset not in SEED_SIZE_PRESETS:
        raise ValueError(
            f"seed_size_preset '{seed_size_preset}' ukendt. "
            f"Brug en af: {list(SEED_SIZE_PRESETS.keys())}"
        )

    rng = np.random.default_rng(random_seed)
    noise_level = NOISE_PRESETS[noise_preset]
    L_min, L_max, W_min, W_max = SEED_SIZE_PRESETS[seed_size_preset]

    # Start med lys plade-baggrund
    image = np.ones((img_size, img_size), dtype=np.float32) * BACKGROUND_REFLECTANCE

    placed_ellipses = []
    centers = []
    margin = L_max // 2 + 2

    for _ in range(num_seeds):
        length = int(rng.integers(L_min, L_max + 1))
        width = int(rng.integers(W_min, W_max + 1))

        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            cx = int(rng.integers(margin, img_size - margin))
            cy = int(rng.integers(margin, img_size - margin))
            angle = float(rng.uniform(0, 360))

            new_e = (cx, cy, length, width, angle)
            collision = any(
                _ellipses_overlap(new_e, ex, PACKING_GAP_PIXELS)
                for ex in placed_ellipses
            )
            if not collision:
                color = float(rng.uniform(SEED_REFLECTANCE_MIN, SEED_REFLECTANCE_MAX))
                _draw_ellipse(image, cx, cy, length, width, angle, color)
                placed_ellipses.append(new_e)
                centers.append((cx, cy))
                break
        # Hvis vi ikke kunne placere efter MAX forsøg: pladen er fuld, skip
        # (vi forsøger ikke at placere flere — lader bare være)

    # Tilføj reflektans-støj
    noise = rng.standard_normal((img_size, img_size)) * noise_level
    image = np.clip(image + noise, 0.0, 1.0)

    # Konvertér til uint8 for visning og YOLO
    img_uint8 = (image * 255).astype(np.uint8)

    return img_uint8, centers, MM_PER_PIXEL
```

- [ ] **Step 4: Kør tests — alle skal passere**

Run: `.venv/bin/pytest scripts/03_demo/tests/test_synthetic.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/03_demo/synthetic.py scripts/03_demo/tests/test_synthetic.py
git commit -m "demo: refaktorer datageneratoren som ren funktion"
```

---

## Task 3: landmarks.py — matching og metrics (TDD)

**Files:**
- Create: `scripts/03_demo/landmarks.py`
- Create: `scripts/03_demo/tests/test_landmarks.py`

- [ ] **Step 1: Skriv testen først**

Opret `scripts/03_demo/tests/test_landmarks.py`:

```python
"""Tests for landmark matching og metrics."""
import sys
from pathlib import Path

import numpy as np

DEMO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DEMO_DIR))

import landmarks  # noqa: E402


def test_perfekt_match_giver_fulde_metrics():
    """Identiske gt og pred → P=R=F1=1.0."""
    gt = [(10, 10), (50, 50), (100, 100)]
    pred = [(10, 10), (50, 50), (100, 100)]
    result = landmarks.match_landmarks(gt, pred, tolerance_px=5)
    assert result.tp == 3
    assert result.fp == 0
    assert result.fn == 0

    metrics = landmarks.compute_metrics(result, mm_per_pixel=0.05)
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.median_error_mm == 0.0


def test_ingen_predictions_giver_recall_nul():
    """Ground truth findes, ingen predictions → R=0, P udefineret (vi sætter 0)."""
    gt = [(10, 10), (50, 50)]
    pred = []
    result = landmarks.match_landmarks(gt, pred, tolerance_px=5)
    assert result.tp == 0
    assert result.fp == 0
    assert result.fn == 2

    metrics = landmarks.compute_metrics(result, mm_per_pixel=0.05)
    assert metrics.precision == 0.0  # 0 predictions, definér som 0
    assert metrics.recall == 0.0
    assert metrics.f1 == 0.0


def test_kun_falske_predictions():
    """Pred uden ground truth → alle FP."""
    gt = []
    pred = [(10, 10), (50, 50)]
    result = landmarks.match_landmarks(gt, pred, tolerance_px=5)
    assert result.tp == 0
    assert result.fp == 2
    assert result.fn == 0


def test_lille_afvigelse_inden_for_tolerance_matcher():
    """Pred 3 pixels fra gt med tolerance=5 → match."""
    gt = [(50, 50)]
    pred = [(53, 50)]
    result = landmarks.match_landmarks(gt, pred, tolerance_px=5)
    assert result.tp == 1
    assert result.fp == 0
    assert len(result.distances_px) == 1
    assert abs(result.distances_px[0] - 3.0) < 0.001


def test_stor_afvigelse_uden_for_tolerance_matcher_ikke():
    """Pred 10 pixels fra gt med tolerance=5 → ingen match."""
    gt = [(50, 50)]
    pred = [(60, 50)]
    result = landmarks.match_landmarks(gt, pred, tolerance_px=5)
    assert result.tp == 0
    assert result.fp == 1
    assert result.fn == 1


def test_metrics_omregner_til_mm():
    """Pixel-distancer skal omregnes til mm via mm_per_pixel."""
    gt = [(50, 50)]
    pred = [(54, 50)]  # 4 pixels væk
    result = landmarks.match_landmarks(gt, pred, tolerance_px=5)
    metrics = landmarks.compute_metrics(result, mm_per_pixel=0.05)
    # 4 pixels * 0.05 mm/pixel = 0.20 mm
    assert abs(metrics.median_error_mm - 0.20) < 0.001
```

- [ ] **Step 2: Kør testen — den skal fejle**

Run: `.venv/bin/pytest scripts/03_demo/tests/test_landmarks.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'landmarks'`

- [ ] **Step 3: Implementér landmarks.py**

Opret `scripts/03_demo/landmarks.py`:

```python
"""Landmark-matching og metrics — refaktoreret fra 04_predict_and_evaluate_landmarks.py.

To rene funktioner:
- match_landmarks(gt, pred, tolerance) → MatchResult
- compute_metrics(result, mm_per_pixel) → Metrics
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MatchResult:
    """Resultat af greedy nearest-neighbor matching."""
    tp: int                          # antal predictions inden for tolerance
    fp: int                          # antal predictions uden for tolerance
    fn: int                          # antal gt der ikke blev matched
    distances_px: list[float] = field(default_factory=list)  # afstande for TP'er


@dataclass
class Metrics:
    """Sammenfatning af præstation til visning."""
    precision: float
    recall: float
    f1: float
    median_error_px: float
    median_error_mm: float
    mean_error_px: float
    mean_error_mm: float
    max_error_px: float
    max_error_mm: float


def match_landmarks(gt_centers, pred_centers, tolerance_px: float) -> MatchResult:
    """Greedy matching: hver pred får sit nærmeste gt hvis inden for tolerance.

    Args:
        gt_centers: liste af (x, y) ground truth-centre
        pred_centers: liste af (x, y) predicted centre
        tolerance_px: maks afstand for at noget regnes som korrekt match

    Returns:
        MatchResult med TP/FP/FN og distancer for matchede par.
    """
    gt = np.array(gt_centers) if len(gt_centers) > 0 else np.empty((0, 2))
    pred = np.array(pred_centers) if len(pred_centers) > 0 else np.empty((0, 2))

    matched_gt = np.zeros(len(gt), dtype=bool)
    tp, fp = 0, 0
    distances = []

    for p in pred:
        if len(gt) == 0:
            fp += 1
            continue
        # Beregn afstande fra denne pred til alle gt
        dists = np.linalg.norm(gt - p, axis=1)
        # Maskér allerede-matchede gt
        dists_masked = np.where(matched_gt, np.inf, dists)
        nearest_idx = int(np.argmin(dists_masked))
        nearest_dist = float(dists_masked[nearest_idx])

        if nearest_dist <= tolerance_px:
            tp += 1
            matched_gt[nearest_idx] = True
            distances.append(nearest_dist)
        else:
            fp += 1

    fn = int(np.sum(~matched_gt))
    return MatchResult(tp=tp, fp=fp, fn=fn, distances_px=distances)


def compute_metrics(result: MatchResult, mm_per_pixel: float) -> Metrics:
    """Beregn P/R/F1 og distance-statistik fra et match-resultat."""
    # Precision og recall — pas på division ved nul
    p_denom = result.tp + result.fp
    precision = result.tp / p_denom if p_denom > 0 else 0.0

    r_denom = result.tp + result.fn
    recall = result.tp / r_denom if r_denom > 0 else 0.0

    f1_denom = precision + recall
    f1 = 2 * precision * recall / f1_denom if f1_denom > 0 else 0.0

    # Distance-stats — håndtér tomt list
    if len(result.distances_px) > 0:
        d = np.array(result.distances_px)
        median_px = float(np.median(d))
        mean_px = float(np.mean(d))
        max_px = float(np.max(d))
    else:
        median_px = mean_px = max_px = 0.0

    return Metrics(
        precision=precision,
        recall=recall,
        f1=f1,
        median_error_px=median_px,
        median_error_mm=median_px * mm_per_pixel,
        mean_error_px=mean_px,
        mean_error_mm=mean_px * mm_per_pixel,
        max_error_px=max_px,
        max_error_mm=max_px * mm_per_pixel,
    )
```

- [ ] **Step 4: Kør tests — alle skal passere**

Run: `.venv/bin/pytest scripts/03_demo/tests/test_landmarks.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/03_demo/landmarks.py scripts/03_demo/tests/test_landmarks.py
git commit -m "demo: tilføj landmark matching og metrics"
```

---

## Task 4: info_texts.py — danske tooltip-tekster

**Files:**
- Create: `scripts/03_demo/info_texts.py`

- [ ] **Step 1: Opret info_texts.py med alle nøgler**

```python
"""Centrale danske forklaringer til alle tekniske termer i demoen.

Holdt i én fil så de er nemme at redigere uden at jagte dem rundt i app.py.
"""

TOOLTIPS = {
    # Parametre
    "antal_frø": (
        "Hvor mange frø generatoren forsøger at placere på pladen. "
        "Tætpakkede billeder er sværere at detektere."
    ),
    "billed_størrelse": (
        "Modellen er trænet på 256×256. Mindre billeder gør prediction lettere "
        "men giver færre frø."
    ),
    "støj_niveau": (
        "Tilfældige reflektans-variationer i pladen. "
        "Kalibreret = som model blev trænet på."
    ),
    "frø_størrelse": (
        "Modellen er trænet på 'normal' størrelse (12-18 px lange). "
        "Mindre/større frø udfordrer modellen."
    ),

    # Visualiseringer
    "ground_truth": (
        "Den kendte sandhed fra generatoren — hvor frøene faktisk er placeret. "
        "Vores facit."
    ),
    "prediction": (
        "Modellens output: bounding boxes omkring hvert frø. "
        "Centrum af hver box = predicted landmark."
    ),

    # Metrics
    "precision": (
        "Andelen af modellens forudsigelser der rammer rigtigt. "
        "1.0 = ingen falske detektioner."
    ),
    "recall": (
        "Andelen af rigtige frø som modellen faktisk fandt. "
        "1.0 = ingen missede frø."
    ),
    "f1": (
        "Harmonisk gennemsnit af Precision og Recall — "
        "én score der balancerer begge."
    ),
    "median_fejl": (
        "Typisk afstand fra predicted center til rigtigt center. "
        "Rapporteret i mm via Videometers skala (0.05 mm/pixel)."
    ),
}
```

- [ ] **Step 2: Verificér via interaktiv tjek**

Run: `.venv/bin/python -c "from scripts.demo_module import info_texts; print(len(info_texts.TOOLTIPS))"`

Det vil fejle pga. mappenavnet med tal. Brug i stedet:

```bash
cd scripts/03_demo && ../../.venv/bin/python -c "import info_texts; assert len(info_texts.TOOLTIPS) == 10; print('OK -', len(info_texts.TOOLTIPS), 'tooltips')"
```

Expected: `OK - 10 tooltips`

- [ ] **Step 3: Commit**

```bash
git add scripts/03_demo/info_texts.py
git commit -m "demo: tilføj danske tooltip-tekster"
```

---

## Task 5: app.py skeleton + sidebar med parametre

**Files:**
- Create: `scripts/03_demo/app.py`

- [ ] **Step 1: Skriv app.py-skelet med model-loader og sidebar**

```python
"""Streamlit-demo til FCM Landmark Detection.

Kør: streamlit run scripts/03_demo/app.py
Forventer at scripts/02_yolo_pipeline/runs/**/best.pt eksisterer (kør Task 3
fra hovedpipelinen først).
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw

# Tilføj demo-mappen til sys.path så lokale moduler kan importeres
DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_DIR))

import synthetic
import landmarks
from info_texts import TOOLTIPS

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FCM Landmark Detection",
    page_icon="🌾",
    layout="wide",
)

CONFIDENCE_THRESHOLD = 0.25
MATCH_TOLERANCE_PX = 5
BBOX_SIZE_PIXELS = 18


# ---------------------------------------------------------------------------
# Model-loader (caches på tværs af reruns)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Find og indlæs den nyeste best.pt fra runs/-mappen."""
    from ultralytics import YOLO

    # Søg fra projekt-roden (to niveauer op fra denne fil)
    project_root = DEMO_DIR.parent.parent
    pattern = str(project_root / "scripts" / "02_yolo_pipeline" / "runs" / "**" / "best.pt")
    matches = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime, reverse=True)

    if not matches:
        st.error(
            "❌ Ingen trænet model fundet. "
            "Kør først `scripts/02_yolo_pipeline/03_train_yolo.py`."
        )
        st.stop()

    return YOLO(matches[0]), matches[0]


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("FCM Landmark Detection")
st.caption(
    "Find frø-centre på Videometer-plader · YOLOv11n · synthetic data showcase"
)

# Indlæs modellen (caches — kun første gang er langsom)
model, model_path = load_model()


# ---------------------------------------------------------------------------
# Sidebar — parametre
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Generér nyt billede")

    num_seeds = st.slider(
        "Antal frø",
        min_value=20, max_value=80, value=58, step=1,
        help=TOOLTIPS["antal_frø"],
    )

    img_size = st.selectbox(
        "Billed-størrelse",
        options=[256, 128],
        index=0,
        format_func=lambda v: f"{v} × {v} px",
        help=TOOLTIPS["billed_størrelse"],
    )

    noise_preset = st.radio(
        "Støj-niveau",
        options=["kalibreret", "realistisk", "svært"],
        index=0,
        horizontal=True,
        help=TOOLTIPS["støj_niveau"],
    )

    seed_size_preset = st.radio(
        "Frø-størrelse",
        options=["mindre", "normal", "større"],
        index=1,
        horizontal=True,
        help=TOOLTIPS["frø_størrelse"],
    )

    regenerate = st.button("⟳ Generér nyt tilfældigt billede", use_container_width=True)

    st.divider()
    st.caption(
        f"**Model:** YOLOv11n  \n"
        f"**Vægtfil:** `{Path(model_path).name}`  \n"
        f"**Skala:** 0.05 mm/pixel"
    )


# Resten af appen kommer i Task 6+
st.info("Sidebar virker. Side-by-side rendering kommer i Task 6.")
```

- [ ] **Step 2: Smoke-test — appen skal starte**

Run: `cd /Users/andreassogardgabel/Development/school/fcm_project_export && .venv/bin/streamlit run scripts/03_demo/app.py --server.headless=true --server.port=8501 &`

Vent 5 sekunder, så:

Run: `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8501`
Expected: `200`

Stop serveren igen: `pkill -f "streamlit run scripts/03_demo"`

- [ ] **Step 3: Manuel browsertjek (åbn URL'en og se)**

Start igen: `.venv/bin/streamlit run scripts/03_demo/app.py`
Åbn `http://localhost:8501` i en browser.

Verificér:
- Titel og caption vises i toppen
- Sidebar har: slider, dropdown, to radio-grupper, knap, info-blok
- Hver kontrol har en `?`-ikon der viser dansk forklaring ved hover
- Ingen røde fejlbeskeder

Stop med Ctrl+C i terminalen.

- [ ] **Step 4: Commit**

```bash
git add scripts/03_demo/app.py
git commit -m "demo: app skeleton med model-loader og sidebar"
```

---

## Task 6: Generér + render side-by-side

**Files:**
- Modify: `scripts/03_demo/app.py` (erstat placeholder-info med rigtig rendering)

- [ ] **Step 1: Tilføj rendering-funktioner og hovedlogik**

Erstat den sidste linje `st.info("Sidebar virker...")` i `app.py` med:

```python
# ---------------------------------------------------------------------------
# Helper: tegn centre/bboxes oven på billedet
# ---------------------------------------------------------------------------
def render_with_overlay(img_uint8, centers=None, boxes=None,
                       center_color=(0, 255, 0), box_color=(255, 80, 80)):
    """Returnér en RGB-PIL af billedet med valgfrie overlays.

    Args:
        img_uint8: (H, W) gråtone-array
        centers: liste af (x, y) — tegnes som små udfyldte cirkler
        boxes: liste af (x1, y1, x2, y2) — tegnes som rektangler
    """
    rgb = np.stack([img_uint8] * 3, axis=-1)  # gråtone → RGB
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)

    if boxes is not None:
        for x1, y1, x2, y2 in boxes:
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=1)

    if centers is not None:
        for x, y in centers:
            r = 2
            draw.ellipse([x - r, y - r, x + r, y + r], fill=center_color)

    return img


# ---------------------------------------------------------------------------
# Generér + predict
# ---------------------------------------------------------------------------
# Brug session_state til at holde et "regenereringsfrø" der ændres når
# brugeren klikker knappen — så samme parametre giver et nyt billede.
if "rng_seed" not in st.session_state:
    st.session_state.rng_seed = 42
if regenerate:
    st.session_state.rng_seed = int(np.random.randint(0, 1_000_000))

with st.spinner("Genererer billede og kører prediction…"):
    img_uint8, gt_centers, mm_per_pixel = synthetic.generate_image(
        num_seeds=num_seeds,
        img_size=img_size,
        noise_preset=noise_preset,
        seed_size_preset=seed_size_preset,
        random_seed=st.session_state.rng_seed,
    )

    # YOLO predict — modellen accepterer numpy arrays direkte
    results = model.predict(
        img_uint8, conf=CONFIDENCE_THRESHOLD, verbose=False, device="mps",
    )
    r = results[0]

    if r.boxes is not None and len(r.boxes) > 0:
        xyxy = r.boxes.xyxy.cpu().numpy()
        xywh = r.boxes.xywh.cpu().numpy()
        pred_centers = [(float(c[0]), float(c[1])) for c in xywh[:, :2]]
        pred_boxes = [tuple(b) for b in xyxy]
    else:
        pred_centers = []
        pred_boxes = []


# ---------------------------------------------------------------------------
# Side-by-side rendering
# ---------------------------------------------------------------------------
col_gt, col_pred = st.columns(2)

with col_gt:
    st.markdown("### 🟢 Ground Truth", help=TOOLTIPS["ground_truth"])
    img_gt = render_with_overlay(img_uint8, centers=gt_centers)
    st.image(img_gt, use_container_width=True)
    st.caption(f"{len(gt_centers)} frø placeret af generatoren")

with col_pred:
    st.markdown("### 🔴 YOLOv11n Prediction", help=TOOLTIPS["prediction"])
    img_pred = render_with_overlay(
        img_uint8, centers=pred_centers, boxes=pred_boxes,
    )
    st.image(img_pred, use_container_width=True)
    inference_ms = float(r.speed.get("inference", 0.0)) if hasattr(r, "speed") else 0.0
    st.caption(f"{len(pred_centers)} detektioner · {inference_ms:.0f} ms inference")
```

- [ ] **Step 2: Manuel browsertjek**

Run: `.venv/bin/streamlit run scripts/03_demo/app.py`

Verificér:
- To billeder vises side om side
- Venstre har grønne prikker (ground truth)
- Højre har røde rektangler + grønne prikker (prediction)
- Skift `Antal frø` → siden re-renderer med ny tæthed
- Skift `Billed-størrelse` til 128 → mindre billede vises
- Klik "⟳ Generér nyt tilfældigt billede" → nyt billede med samme parametre

Stop med Ctrl+C.

- [ ] **Step 3: Commit**

```bash
git add scripts/03_demo/app.py
git commit -m "demo: side-by-side rendering med ground truth og prediction"
```

---

## Task 7: Metrics-bar

**Files:**
- Modify: `scripts/03_demo/app.py` (tilføj metrics nederst)

- [ ] **Step 1: Tilføj metrics-beregning og visning**

Tilføj nederst i `app.py` (efter `col_pred`-blokken):

```python
# ---------------------------------------------------------------------------
# Metrics-bar
# ---------------------------------------------------------------------------
match = landmarks.match_landmarks(
    gt_centers, pred_centers, tolerance_px=MATCH_TOLERANCE_PX,
)
metrics = landmarks.compute_metrics(match, mm_per_pixel=mm_per_pixel)

st.divider()

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    label="Precision",
    value=f"{metrics.precision:.3f}",
    help=TOOLTIPS["precision"],
)
m2.metric(
    label="Recall",
    value=f"{metrics.recall:.3f}",
    help=TOOLTIPS["recall"],
)
m3.metric(
    label="F1 Score",
    value=f"{metrics.f1:.3f}",
    help=TOOLTIPS["f1"],
)
m4.metric(
    label="Median fejl",
    value=f"{metrics.median_error_mm:.3f} mm",
    delta=f"{metrics.median_error_px:.2f} px",
    delta_color="off",
    help=TOOLTIPS["median_fejl"],
)
```

- [ ] **Step 2: Manuel browsertjek**

Run: `.venv/bin/streamlit run scripts/03_demo/app.py`

Verificér:
- Fire metrics vises under billederne: Precision, Recall, F1, Median fejl
- Værdierne ligger højt (0.9+ for default-parametre)
- Hver har et `?`-ikon med dansk forklaring
- Skift til "svært" støj → metrics falder typisk
- Skift til "større" frø → metrics falder typisk
- 0 detektioner-edge-case crasher ikke (prøv `Antal frø=20`, `Billed-størrelse=128`, `seed_size=større`)

Stop med Ctrl+C.

- [ ] **Step 3: Commit**

```bash
git add scripts/03_demo/app.py
git commit -m "demo: tilføj metrics-bar med precision/recall/f1/median-fejl"
```

---

## Task 8: Forklaringsblok (info panel)

**Files:**
- Modify: `scripts/03_demo/app.py` (tilføj info-blok nederst)

- [ ] **Step 1: Tilføj forklaringsblok**

Tilføj nederst i `app.py` (efter metrics-bar):

```python
# ---------------------------------------------------------------------------
# Forklarings-blok
# ---------------------------------------------------------------------------
st.info(
    "**Hvad ser du?** Modellen detekterer rektangulære regioner "
    "(bounding boxes) omkring hvert frø. Vi tager centrum af hver box som "
    "det predicted landmark. Median-afstanden i mm fortæller hvor præcist "
    "modellen rammer — typisk ~0.5 pixel = sub-pixel præcision."
)
```

- [ ] **Step 2: Manuel browsertjek**

Run: `.venv/bin/streamlit run scripts/03_demo/app.py`

Verificér:
- Blå info-blok vises nederst med forklaringen
- Tekst er læselig og giver mening for en ML-novice

Stop med Ctrl+C.

- [ ] **Step 3: Commit**

```bash
git add scripts/03_demo/app.py
git commit -m "demo: tilføj forklaringsblok under metrics"
```

---

## Task 9: Custom CSS for visuel polish

**Files:**
- Create: `scripts/03_demo/styles.css`
- Modify: `scripts/03_demo/app.py` (load CSS)

- [ ] **Step 1: Skriv CSS-filen**

Opret `scripts/03_demo/styles.css`:

```css
/* Fjern Streamlit-branding for cleaner demo-look */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header [data-testid="stToolbar"] {visibility: hidden;}

/* Smallere padding i toppen — mere indhold på skærmen */
section[data-testid="stMain"] > div {
    padding-top: 1.5rem;
}

/* Større, federe titel */
h1 {
    font-weight: 700 !important;
    letter-spacing: -0.02em;
    margin-bottom: 0.25rem !important;
}

/* Caption under titel — mere pro-look */
.stCaption {
    color: #888 !important;
    font-size: 0.9rem !important;
}

/* Sidebar — mørkere baggrund og lidt mere padding */
section[data-testid="stSidebar"] {
    padding-top: 1rem;
}

/* Metric-værdier større og fede */
[data-testid="stMetricValue"] {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #888 !important;
}

/* Billed-overskrifter (Ground Truth / Prediction) — mere markante */
.stMarkdown h3 {
    font-size: 1.1rem !important;
    margin-bottom: 0.5rem !important;
    margin-top: 0.5rem !important;
}

/* Info-boks — mere subtil */
[data-testid="stAlert"] {
    border-left-width: 3px;
    background-color: rgba(74, 158, 255, 0.08);
}
```

- [ ] **Step 2: Load CSS i app.py**

Tilføj efter `st.set_page_config(...)`-blokken i `app.py`:

```python
# Load custom CSS for visuel polish
_css_path = DEMO_DIR / "styles.css"
if _css_path.exists():
    st.markdown(
        f"<style>{_css_path.read_text()}</style>",
        unsafe_allow_html=True,
    )
```

- [ ] **Step 3: Manuel browsertjek**

Run: `.venv/bin/streamlit run scripts/03_demo/app.py`

Verificér:
- Streamlit-hamburger-menuen og footer er væk
- Titlen er fed og stor
- Metrics har større værdier og uppercase labels
- Generelt mere "produkt-look" end før

Stop med Ctrl+C.

- [ ] **Step 4: Commit**

```bash
git add scripts/03_demo/styles.css scripts/03_demo/app.py
git commit -m "demo: custom CSS for visuel polish"
```

---

## Task 10: Edge cases + final smoke test

**Files:** Ingen ændringer — kun manuel verifikation.

- [ ] **Step 1: Edge case-tjek**

Run: `.venv/bin/streamlit run scripts/03_demo/app.py`

Test hver af disse kombinationer og notér eventuelle problemer:

| Antal frø | Billed-størrelse | Støj | Frø-størrelse | Forventet |
|---|---|---|---|---|
| 20 | 256 | kalibreret | normal | Sparsomt billede, høje metrics |
| 80 | 256 | kalibreret | normal | Tæt billede, høje metrics |
| 58 | 128 | kalibreret | normal | Lille billede, evt. lavere metrics |
| 58 | 256 | svært | normal | Støjet billede, metrics falder |
| 58 | 256 | kalibreret | større | Store frø, metrics falder |
| 20 | 128 | svært | større | Worst-case kombination, ingen crash |

For hver kombination, verificér:
- App rendererer uden røde fejlbeskeder
- Begge billeder vises (selv om predictions kan være tomme)
- Metrics viser tal (kan være 0.000 ved 0 detektioner — det er OK)

- [ ] **Step 2: Tooltip-tjek**

For hver `?`-ikon i appen — hover og verificér at den danske tekst er forståelig.

- [ ] **Step 3: Performance-tjek**

Skift parametre hurtigt (5-6 gange på 10 sekunder). Verificér:
- App holder ikke fast i en gammel state
- Hver ændring giver respons inden for ~3 sekunder
- Spinner ("Genererer billede og kører prediction…") vises mens modellen kører

- [ ] **Step 4: Run alle unit-tests én sidste gang**

Run: `.venv/bin/pytest scripts/03_demo/tests/ -v`
Expected: Alle 11 tests passer (5 i synthetic, 6 i landmarks).

- [ ] **Step 5: Final commit + tag**

```bash
git add -A
git commit -m "demo: færdig — klar til Videometer-præsentation 2026-05-08"
git tag demo-v1
```

---

## Self-review checklist

Efter implementering, kontrollér:

**Spec-coverage:**
- ✅ Side-by-side ground truth vs prediction (Task 6)
- ✅ Fire parametre med kontroller (Task 5)
- ✅ Tooltips på alle termer (Task 5, 6, 7 + info_texts.py i Task 4)
- ✅ Metrics-bar med P/R/F1/Median fejl (Task 7)
- ✅ Custom CSS (Task 9)
- ✅ Tests for synthetic (Task 2) og landmarks (Task 3)
- ✅ Edge case-håndtering (Task 10)

**Hvis noget ikke er dækket — tilføj en task til at fixe det.**

**Demo-klar:** Når Task 10 passerer alle edge cases, er du klar til Videometer.

---

## Hvad sker der efter demoen

- `scripts/03_demo/`-koden bevares og kan genbruges til eksamen.
- Hvis Videometer-data ankommer, tilføjes en "rigtige data"-tilstand som ekstra modus (uden for denne plan).
- Eksamen i juni: samme app, opdateret med rigtige data hvis muligt.
