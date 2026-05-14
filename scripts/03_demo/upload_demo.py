"""Upload-demo til FCM Landmark Detection.

Lader brugeren uploade et rigtigt billede og se modellen's predictions.
Beregnet til validering på rigtige Videometer-scener efter fabrik-besøg.

Kør:
    .venv/bin/streamlit run scripts/03_demo/upload_demo.py
"""
import io
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# Projekt-root så vi kan finde models/active/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ACTIVE_MODEL = PROJECT_ROOT / "models" / "active" / "best.pt"


# ---------------------------------------------------------------------------
# Sidens setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FCM Upload-test",
    layout="wide",
)

st.title("FCM Landmark Detection — Upload-test")
st.markdown(
    "Upload et rigtigt billede fra Videometer-maskinen og se hvor godt "
    "modellen detekterer majs-frø."
)


# ---------------------------------------------------------------------------
# Model-load (caches mellem reruns så vi kun loader én gang)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load YOLO-modellen fra models/active/best.pt (kun én gang).

    Bruger st.cache_resource så modellen ikke gen-loades ved hver Streamlit-rerun.
    """
    from ultralytics import YOLO
    if not ACTIVE_MODEL.exists():
        st.error(f"❌ Mangler model på {ACTIVE_MODEL}")
        st.stop()
    return YOLO(str(ACTIVE_MODEL))


model = load_model()
active_version = ACTIVE_MODEL.parent.resolve().name
st.sidebar.markdown(f"**Aktiv model:** `{active_version}`")
st.sidebar.markdown(f"**Vægt-fil:** `{ACTIVE_MODEL.relative_to(PROJECT_ROOT)}`")


# ---------------------------------------------------------------------------
# Sidebar-kontroller
# ---------------------------------------------------------------------------
st.sidebar.header("Detection-indstillinger")

confidence_threshold = st.sidebar.slider(
    "Confidence-tærskel",
    min_value=0.05, max_value=0.95, value=0.25, step=0.05,
    help="Minimum confidence for at acceptere en detection (lavere = flere men "
         "usikrere bbox'er; højere = færre men sikrere)."
)

iou_threshold = st.sidebar.slider(
    "IoU-tærskel (NMS)",
    min_value=0.1, max_value=0.95, value=0.7, step=0.05,
    help="Hvor meget to bbox'er må overlappe før den ene fjernes som duplikat."
)

imgsz = st.sidebar.select_slider(
    "Input-størrelse (px)",
    options=[640, 960, 1280, 1920, 2560],
    value=1280,
    help="YOLO ned-skalerer dit billede til denne størrelse internt. "
         "Større = mere detalje på små frø, men langsommere."
)

show_centers = st.sidebar.checkbox("Vis center-markører", value=True)
show_bboxes = st.sidebar.checkbox("Vis bounding boxes", value=True)
show_confidence = st.sidebar.checkbox("Vis confidence-tal", value=False)


# ---------------------------------------------------------------------------
# Upload-widget
# ---------------------------------------------------------------------------
uploaded = st.file_uploader(
    "Vælg et billede",
    type=["png", "jpg", "jpeg", "bmp"],
    help="PNG eller JPG fra Videometer-kameraet. Den må gerne være stor — "
         "modellen ned-skalerer selv.",
)

if uploaded is None:
    st.info("👆 Upload et billede for at se predictions. "
            "Modellen er trænet på composite-billeder med 50-80 majs på blå "
            "plade-baggrund.")
    st.stop()


# ---------------------------------------------------------------------------
# Vis original
# ---------------------------------------------------------------------------
image_pil = Image.open(uploaded).convert("RGB")
img_w, img_h = image_pil.size

col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Original")
    st.image(image_pil, use_container_width=True)
    st.caption(f"Størrelse: {img_w} × {img_h} px")


# ---------------------------------------------------------------------------
# Kør prediction
# ---------------------------------------------------------------------------
with st.spinner("Detekterer frø…"):
    start = time.time()
    results = model.predict(
        source=np.array(image_pil),
        conf=confidence_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        device="mps",
        verbose=False,
    )
    elapsed_ms = (time.time() - start) * 1000

result = results[0]
boxes = result.boxes  # ultralytics Boxes-objekt

# Hver detection: (x1, y1, x2, y2, conf, cls)
xyxy = boxes.xyxy.cpu().numpy() if boxes is not None else np.zeros((0, 4))
confs = boxes.conf.cpu().numpy() if boxes is not None else np.zeros(0)

n_detections = len(xyxy)


# ---------------------------------------------------------------------------
# Tegn prediction-overlay
# ---------------------------------------------------------------------------
overlay = image_pil.copy()
drawer = ImageDraw.Draw(overlay)

# Skalér linje-tykkelse til billed-størrelsen så det ser ens ud
line_width = max(2, int(min(img_w, img_h) / 500))
marker_radius = max(4, int(min(img_w, img_h) / 200))

for (x1, y1, x2, y2), conf in zip(xyxy, confs):
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    if show_bboxes:
        drawer.rectangle([(x1, y1), (x2, y2)], outline="red", width=line_width)

    if show_centers:
        drawer.ellipse(
            [(cx - marker_radius, cy - marker_radius),
             (cx + marker_radius, cy + marker_radius)],
            fill="lime", outline="red", width=max(1, line_width // 2),
        )

    if show_confidence:
        text_offset = marker_radius + 4
        drawer.text((cx + text_offset, cy - text_offset),
                     f"{conf:.2f}", fill="yellow")


with col_right:
    st.subheader("Prediction")
    st.image(overlay, use_container_width=True)
    st.caption(f"{n_detections} frø detekteret  ·  {elapsed_ms:.0f} ms")


# ---------------------------------------------------------------------------
# Metrics og statistik
# ---------------------------------------------------------------------------
st.divider()

col_a, col_b, col_c, col_d = st.columns(4)

col_a.metric("Frø detekteret", n_detections)
col_b.metric("Inference-tid", f"{elapsed_ms:.0f} ms")
if n_detections > 0:
    col_c.metric("Avg. confidence", f"{confs.mean():.2f}")
    col_d.metric("Min confidence", f"{confs.min():.2f}")
else:
    col_c.metric("Avg. confidence", "—")
    col_d.metric("Min confidence", "—")


# Confidence-fordeling som histogram (kun hvis mindst 1 detection)
if n_detections > 0:
    st.subheader("Confidence-fordeling")
    import pandas as pd
    df = pd.DataFrame({"confidence": confs})
    st.bar_chart(df["confidence"].value_counts(bins=10).sort_index())

    st.subheader("Bbox-størrelser")
    widths = xyxy[:, 2] - xyxy[:, 0]
    heights = xyxy[:, 3] - xyxy[:, 1]
    df_sizes = pd.DataFrame({"bredde (px)": widths, "højde (px)": heights})
    st.dataframe(df_sizes.describe().round(1), use_container_width=True)


# ---------------------------------------------------------------------------
# Download-knap til overlay-billede
# ---------------------------------------------------------------------------
buf = io.BytesIO()
overlay.save(buf, format="PNG")
st.download_button(
    label="📥 Download prediction-overlay",
    data=buf.getvalue(),
    file_name=f"prediction_{uploaded.name.rsplit('.', 1)[0]}.png",
    mime="image/png",
)


# ---------------------------------------------------------------------------
# Tips
# ---------------------------------------------------------------------------
with st.expander("💡 Tips til at fortolke resultaterne"):
    st.markdown("""
- **Få detections (<10):** modellen ser ikke frøene. Prøv at sænke confidence-tærsklen i sidebar.
- **For mange detections:** højere confidence-tærskel filtrerer svage forslag væk.
- **Detections på baggrund:** modellen er ikke trænet på den belysning eller plade. Composite-pipelinen brugte kun blå plade — andre baggrunde kan forvirre.
- **Mistede frø midt i grupper:** dette er det Videometer specifikt sagde var svært — touching/overlap. Det er hovedudfordringen modellen blev trænet på.
- **Konsistens på tværs af confidence-niveauer:** prøv at justere slider'en og se ved hvilket niveau du har "ægte" detections vs noise.
""")
