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
    layout="wide",
)

# Load custom CSS for visuel polish (hvis filen findes)
css_file_path = DEMO_DIR / "styles.css"
if css_file_path.exists():
    st.markdown(
        f"<style>{css_file_path.read_text()}</style>",
        unsafe_allow_html=True,
    )

CONFIDENCE_THRESHOLD = 0.25
MATCH_TOLERANCE_PX = 5


# ---------------------------------------------------------------------------
# Model-loader (caches på tværs af reruns)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Find og indlæs den aktive model.

    Foretrækker models/active/best.pt (versioneret setup, skift via
    ./switch_model.sh). Falder tilbage til runs/-glob hvis active ikke
    findes — så pipelinen virker også før versioneringen er sat op.
    """
    from ultralytics import YOLO

    project_root = DEMO_DIR.parent.parent

    # Primær: versioneret active-model
    active_path = project_root / "models" / "active" / "best.pt"
    if active_path.exists():
        return YOLO(str(active_path)), str(active_path)

    # Fallback: nyeste best.pt under runs/ (legacy-sti)
    pattern = str(
        project_root / "scripts" / "02_yolo_pipeline" / "runs" / "**" / "best.pt"
    )
    matching_files = sorted(
        glob.glob(pattern, recursive=True), key=os.path.getmtime, reverse=True
    )

    if not matching_files:
        st.error(
            "Ingen trænet model fundet. "
            "Forventet sti: models/active/best.pt — kør `./switch_model.sh <version>` "
            "eller træn en model via `scripts/02_yolo_pipeline/03_train_yolo.py`."
        )
        st.stop()

    return YOLO(matching_files[0]), matching_files[0]


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
    rgb_array = np.stack([img_uint8] * 3, axis=-1)  # gråtone → RGB
    overlay_image = Image.fromarray(rgb_array)
    drawer = ImageDraw.Draw(overlay_image)

    if boxes is not None:
        for x1, y1, x2, y2 in boxes:
            drawer.rectangle([x1, y1, x2, y2], outline=box_color, width=1)

    if centers is not None:
        center_dot_radius = 2
        for x, y in centers:
            drawer.ellipse(
                [x - center_dot_radius, y - center_dot_radius,
                 x + center_dot_radius, y + center_dot_radius],
                fill=center_color,
            )

    return overlay_image


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
    st.header("Parametre")

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
        "Generér nyt billede", width="stretch"
    )

    st.divider()
    # Følg symlinks så vi får den faktiske versions-mappe (fx "v1_kalibreret"),
    # ikke bare "active" som er symlink-navnet.
    resolved_model_path = Path(model_path).resolve()
    version_name = (
        resolved_model_path.parent.name
        if resolved_model_path.parent.name != "active" else "ukendt"
    )
    st.caption(
        f"**Model:** YOLOv11n  \n"
        f"**Version:** `{version_name}`  \n"
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
    img_uint8, ground_truth_centers, mm_per_pixel = synthetic.generate_image(
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
    prediction_results = model.predict(
        img_rgb, conf=CONFIDENCE_THRESHOLD, verbose=False, device="mps",
    )
    first_result = prediction_results[0]

    if first_result.boxes is not None and len(first_result.boxes) > 0:
        boxes_xyxy = first_result.boxes.xyxy.cpu().numpy()
        boxes_xywh = first_result.boxes.xywh.cpu().numpy()
        predicted_centers = [
            (float(center_xy[0]), float(center_xy[1]))
            for center_xy in boxes_xywh[:, :2]
        ]
        predicted_boxes = [tuple(box_coords) for box_coords in boxes_xyxy]
    else:
        predicted_centers = []
        predicted_boxes = []


# ---------------------------------------------------------------------------
# Side-by-side rendering
# ---------------------------------------------------------------------------
column_truth, column_prediction = st.columns(2)

with column_truth:
    st.markdown("### Ground truth", help=TOOLTIPS["ground_truth"])
    image_with_truth = render_with_overlay(img_uint8, centers=ground_truth_centers)
    st.image(image_with_truth, width="stretch")
    st.caption(f"{len(ground_truth_centers)} frø placeret af generatoren")

with column_prediction:
    st.markdown("### Model output", help=TOOLTIPS["prediction"])
    image_with_prediction = render_with_overlay(
        img_uint8, centers=predicted_centers, boxes=predicted_boxes,
    )
    st.image(image_with_prediction, width="stretch")
    inference_ms = (
        float(first_result.speed.get("inference", 0.0))
        if hasattr(first_result, "speed") else 0.0
    )
    st.caption(f"{len(predicted_centers)} detektioner · {inference_ms:.0f} ms inference")


# ---------------------------------------------------------------------------
# Metrics-bar
# ---------------------------------------------------------------------------
match_result = landmarks.match_landmarks(
    ground_truth_centers, predicted_centers, tolerance_px=MATCH_TOLERANCE_PX,
)
metrics = landmarks.compute_metrics(match_result, mm_per_pixel=mm_per_pixel)

st.divider()

precision_column, recall_column, f1_column, error_column = st.columns(4)
precision_column.metric(
    label="Precision",
    value=f"{metrics.precision:.3f}",
    help=TOOLTIPS["precision"],
)
recall_column.metric(
    label="Recall",
    value=f"{metrics.recall:.3f}",
    help=TOOLTIPS["recall"],
)
f1_column.metric(
    label="F1 Score",
    value=f"{metrics.f1_score:.3f}",
    help=TOOLTIPS["f1"],
)
error_column.metric(
    label="Median fejl",
    value=f"{metrics.median_error_mm:.3f} mm",
    delta=f"{metrics.median_error_px:.2f} px",
    delta_color="off",
    help=TOOLTIPS["median_fejl"],
)


# ---------------------------------------------------------------------------
# Forklarings-blok (custom — ikke Streamlit's default alert-look)
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="callout">'
    '<span class="callout-label">Note</span>'
    'Modellen detekterer rektangulære regioner (bounding boxes) omkring hvert '
    'frø. Vi tager centrum af hver box som det predicted landmark. '
    'Median-afstanden i mm fortæller hvor præcist modellen rammer — typisk '
    '~0.5 pixel, dvs. sub-pixel præcision.'
    '</div>',
    unsafe_allow_html=True,
)
