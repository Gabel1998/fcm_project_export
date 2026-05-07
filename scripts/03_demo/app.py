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

# Load custom CSS for visuel polish (hvis filen findes)
_css_path = DEMO_DIR / "styles.css"
if _css_path.exists():
    st.markdown(
        f"<style>{_css_path.read_text()}</style>",
        unsafe_allow_html=True,
    )

CONFIDENCE_THRESHOLD = 0.25
MATCH_TOLERANCE_PX = 5


# ---------------------------------------------------------------------------
# Model-loader (caches på tværs af reruns)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Find og indlæs den nyeste best.pt fra runs/-mappen."""
    from ultralytics import YOLO

    # Søg fra projekt-roden (to niveauer op fra denne fil)
    project_root = DEMO_DIR.parent.parent
    pattern = str(
        project_root / "scripts" / "02_yolo_pipeline" / "runs" / "**" / "best.pt"
    )
    matches = sorted(
        glob.glob(pattern, recursive=True), key=os.path.getmtime, reverse=True
    )

    if not matches:
        st.error(
            "❌ Ingen trænet model fundet. "
            "Kør først `scripts/02_yolo_pipeline/03_train_yolo.py`."
        )
        st.stop()

    return YOLO(matches[0]), matches[0]


# ---------------------------------------------------------------------------
# Helper: tegn centre/bboxes oven på billedet
# ---------------------------------------------------------------------------
def render_with_overlay(
    img_uint8, centers=None, boxes=None,
    center_color=(0, 255, 0), box_color=(255, 80, 80),
):
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

    regenerate = st.button(
        "⟳ Generér nyt tilfældigt billede", width="stretch"
    )

    st.divider()
    st.caption(
        f"**Model:** YOLOv11n  \n"
        f"**Vægtfil:** `{Path(model_path).name}`  \n"
        f"**Skala:** 0.05 mm/pixel"
    )


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

    # YOLO forventer 3-kanal RGB. Generatoren laver gråtone, så vi stabler
    # samme kanal tre gange — modellen har set præcis sådan input under træning
    # (PNG-filerne fra script 2 blev læst som 3-kanal af YOLO's loader).
    img_rgb = np.stack([img_uint8] * 3, axis=-1)
    results = model.predict(
        img_rgb, conf=CONFIDENCE_THRESHOLD, verbose=False, device="mps",
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
    st.image(img_gt, width="stretch")
    st.caption(f"{len(gt_centers)} frø placeret af generatoren")

with col_pred:
    st.markdown("### 🔴 YOLOv11n Prediction", help=TOOLTIPS["prediction"])
    img_pred = render_with_overlay(
        img_uint8, centers=pred_centers, boxes=pred_boxes,
    )
    st.image(img_pred, width="stretch")
    inference_ms = (
        float(r.speed.get("inference", 0.0)) if hasattr(r, "speed") else 0.0
    )
    st.caption(f"{len(pred_centers)} detektioner · {inference_ms:.0f} ms inference")


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


# ---------------------------------------------------------------------------
# Forklarings-blok
# ---------------------------------------------------------------------------
st.info(
    "**Hvad ser du?** Modellen detekterer rektangulære regioner "
    "(bounding boxes) omkring hvert frø. Vi tager centrum af hver box som "
    "det predicted landmark. Median-afstanden i mm fortæller hvor præcist "
    "modellen rammer — typisk ~0.5 pixel = sub-pixel præcision."
)
