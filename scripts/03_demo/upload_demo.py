"""Upload-demo til FCM Landmark Detection.

Lader brugeren uploade et rigtigt billede og se modellen's predictions.
Beregnet til validering på rigtige Videometer-scener efter fabrik-besøg.

Kør:
    .venv/bin/streamlit run scripts/03_demo/upload_demo.py
"""
import csv
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yaml
from PIL import Image, ImageDraw, ImageFont

# Projekt-root så vi kan finde models/active/
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ACTIVE_MODEL = PROJECT_ROOT / "models" / "active" / "best.pt"

# Gør sibling-modulerne importérbare (Streamlit kører scriptet direkte)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import seed_morphology  # noqa: E402  — segmentering + center-of-mass + orientering
import overlays  # noqa: E402  — lag-baseret overlay-render


# ---------------------------------------------------------------------------
# Sidens setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FCM Upload-test",
    layout="wide",
)

st.title("FCM Landmark Detection — Upload-test")
st.markdown(
    "Upload et billede fra Videometer og se hvor godt "
    "modellen detekterer majs."
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
        st.error(f"Mangler model på {ACTIVE_MODEL}")
        st.stop()
    return YOLO(str(ACTIVE_MODEL))


model = load_model()
active_version = ACTIVE_MODEL.parent.resolve().name
st.sidebar.markdown(f"**Aktiv model:** `{active_version}`")
st.sidebar.markdown(f"**Vægt-fil:** `{ACTIVE_MODEL.relative_to(PROJECT_ROOT)}`")


@st.cache_data(show_spinner=False)
def load_model_metadata(version_name: str, _model) -> dict:
    """Samler alt metadata om aktiv model i ét dict.

    Kilder: best.pt (filstørrelse), Ultralytics' info-summary (arkitektur,
    layers, params, gradients), results.csv (training-metrikker), args.yaml
    (training-config), data.yaml (dataset-info).
    Cachet på version_name så vi kun læser én gang pr. session.
    """
    meta: dict = {"version": version_name}
    model_path = PROJECT_ROOT / "models" / version_name / "best.pt"
    if model_path.exists():
        meta["file_size_mb"] = model_path.stat().st_size / (1024 * 1024)

    # Arkitektur: tæl direkte fra modellen (Ultralytics' "layers"-tal er
    # antallet af leaf-modules, dvs. modules uden children). Det er mere
    # robust end at parse logger-output, som ikke altid bliver fanget når
    # Streamlit kører.
    for param in _model.model.parameters():
        param.requires_grad_(True)
    meta["layers"] = sum(
        1 for _, mod in _model.model.named_modules() if len(mod._modules) == 0
    )
    meta["parameters"] = sum(p.numel() for p in _model.model.parameters())
    meta["gradients"] = sum(
        p.numel() for p in _model.model.parameters() if p.requires_grad
    )

    # Arkitektur-navn via detection-head'en (v10Detect = YOLOv10, Detect = YOLO11)
    for module in _model.model.modules():
        cls_name = module.__class__.__name__
        if cls_name == "v10Detect":
            meta["arch"] = "YOLOv10n"
            break
        if cls_name == "Detect":
            meta["arch"] = "YOLO11n"
            break

    # Training metrics: sidste række i results.csv
    results_csv = PROJECT_ROOT / "models" / version_name / "results.csv"
    if results_csv.exists():
        with open(results_csv) as f:
            rows = list(csv.DictReader(f))
        if rows:
            last = rows[-1]
            meta["mAP50"] = float(last.get("metrics/mAP50(B)", 0))
            meta["mAP50_95"] = float(last.get("metrics/mAP50-95(B)", 0))
            meta["precision"] = float(last.get("metrics/precision(B)", 0))
            meta["recall"] = float(last.get("metrics/recall(B)", 0))

    # Training-config fra args.yaml (ligger i runs/, ikke models/)
    args_path = (
        PROJECT_ROOT / "runs" / "detect" / "runs" / "train" / version_name / "args.yaml"
    )
    if args_path.exists():
        with open(args_path) as f:
            args = yaml.safe_load(f) or {}
        for key in ("epochs", "imgsz", "batch", "patience", "device"):
            if key in args:
                meta[key] = args[key]
        meta["data_yaml"] = args.get("data")

    # Dataset-info: tæl filer i train/val mapperne
    data_yaml_str = meta.get("data_yaml")
    if data_yaml_str:
        data_yaml_path = Path(data_yaml_str)
        if data_yaml_path.exists():
            with open(data_yaml_path) as f:
                dcfg = yaml.safe_load(f) or {}
            dataset_root = data_yaml_path.parent
            meta["dataset_name"] = dataset_root.name
            for split_key, count_key in (("train", "train_count"), ("val", "val_count")):
                split_dir = dataset_root / dcfg.get(split_key, "")
                if split_dir.exists():
                    meta[count_key] = sum(
                        1 for p in split_dir.iterdir()
                        if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                    )

    return meta


def compute_objectness_heatmaps(_model, image_array: np.ndarray, imgsz: int, device: str):
    """Hook YOLO's detection-head og fang klassifikations-sandsynligheder pr. scale.

    YOLO11 outputter 3 detection-skalaer (P3, P4, P5) med hver sin grid-opløsning.
    For 1-klasse-modeller (kun "seed") er sigmoid af klassifikations-logit'en =
    pixel-baseret confidence for at der er et frø i den grid-celle. Det er
    modellens egen "heatmap" før NMS og bbox-decoding.

    Returnerer liste af 3 [H, W] numpy-arrays i [0, 1] sorteret fra fineste
    (P3) til groveste (P5) opløsning.
    """
    import torch

    # Dict så seneste capture pr. scale overskriver tidligere. Ultralytics
    # kører en warmup-pass med dummy-tensor før den rigtige inference, så
    # hooks fanger BÅDE warmup (output ≈ 0) og det reelle output. Vi vil
    # have det sidste (rigtige) for hver scale.
    captured: dict[int, np.ndarray] = {}

    def make_hook(scale_idx: int):
        def hook(module, inp, out):
            prob = torch.sigmoid(out).detach().cpu().numpy()[0, 0]
            captured[scale_idx] = prob
        return hook

    detect = _model.model.model[-1]
    hooks = []
    for i, cv3_branch in enumerate(detect.cv3):
        last_conv = cv3_branch[-1]
        hooks.append(last_conv.register_forward_hook(make_hook(i)))

    try:
        _model.predict(
            image_array, imgsz=imgsz, device=device, conf=0.01, verbose=False
        )
    finally:
        for h in hooks:
            h.remove()

    return [captured[i] for i in sorted(captured.keys())]


def make_heatmap_overlay(base_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Læg 2D heatmap som farve-overlay på et PIL-billede.

    Heatmappet upsamples til billed-størrelse (bilinear) og farves med
    matplotlib's 'jet'-colormap (blå = lav, rød = høj). Alpha styrer
    blanding: 0.0 = kun base, 1.0 = kun heatmap.
    """
    from matplotlib import colormaps

    target_w, target_h = base_img.size

    # Upsample heatmap til samme størrelse som input-billedet
    hm_pil = Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8), mode="L")
    hm_resized = hm_pil.resize((target_w, target_h), resample=Image.BILINEAR)
    hm_norm = np.array(hm_resized).astype(np.float32) / 255.0

    # Apply jet colormap → [H, W, 4] RGBA, drop alpha
    cmap = colormaps["jet"]
    colored = (cmap(hm_norm)[:, :, :3] * 255).astype(np.uint8)

    base_array = np.array(base_img.convert("RGB")).astype(np.float32)
    blended = (1 - alpha) * base_array + alpha * colored.astype(np.float32)
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def merge_close_centers(xyxy: np.ndarray, confs: np.ndarray, min_dist_px: float):
    """Center-distance NMS: behold detections hvis center er langt nok fra alle
    allerede accepterede. Højeste confidence vinder ved konflikt.

    Argumenter:
        xyxy: (N, 4) array af bbox-hjørner
        confs: (N,) array af confidence-værdier
        min_dist_px: minimum center-til-center afstand i pixels

    Returnerer (filtreret_xyxy, filtreret_confs, antal_fjernet).
    """
    if len(xyxy) == 0:
        return xyxy, confs, 0
    centers = np.stack(
        [(xyxy[:, 0] + xyxy[:, 2]) / 2,
         (xyxy[:, 1] + xyxy[:, 3]) / 2],
        axis=1,
    )
    # Sortér efter confidence (højeste først) — vinder ved konflikt
    order = np.argsort(-confs)
    kept: list[int] = []
    for idx in order:
        c = centers[idx]
        if all(np.linalg.norm(c - centers[k]) >= min_dist_px for k in kept):
            kept.append(int(idx))
    n_removed = len(xyxy) - len(kept)
    return xyxy[kept], confs[kept], n_removed


# ---------------------------------------------------------------------------
# IoU mod ground truth (kun for composite-billeder med kendte labels)
# ---------------------------------------------------------------------------
# (split-navn, images-dir, labels-dir) — vi tjekker val før train
GT_SPLITS = [
    ("val",
     PROJECT_ROOT / "data" / "composite_v1_no_leak" / "images" / "val",
     PROJECT_ROOT / "data" / "composite_v1_no_leak" / "labels" / "val"),
    ("train",
     PROJECT_ROOT / "data" / "composite_v1_no_leak" / "images" / "train",
     PROJECT_ROOT / "data" / "composite_v1_no_leak" / "labels" / "train"),
]


def find_matching_gt(image_name: str, uploaded_rgb: np.ndarray):
    """Find label-filen der matcher det uploadede billede.

    VIGTIGT: matcher ikke kun på filnavn — filnavne kolliderer (test_images og
    træningssættet har begge fx 'composite_00000.png', men med FORSKELLIGT
    indhold). Vi verificerer derfor at datasæt-billedet er pixel-identisk med
    det uploadede, før vi bruger dets labels. Ellers sammenligner man mod et
    helt andet billedes ground truth og får vrøvl-IoU.
    """
    stem = Path(image_name).stem
    for _split, img_dir, lbl_dir in GT_SPLITS:
        label_path = lbl_dir / f"{stem}.txt"
        image_path = img_dir / f"{stem}.png"
        if not (label_path.exists() and image_path.exists()):
            continue
        dataset_rgb = np.array(Image.open(image_path).convert("RGB"))
        if (dataset_rgb.shape == uploaded_rgb.shape
                and np.array_equal(dataset_rgb, uploaded_rgb)):
            return label_path
    return None


def load_gt_boxes(label_path: Path, image_w: int, image_h: int) -> np.ndarray:
    """Læs YOLO-labels (class cx cy w h, normaliseret) → pixel-xyxy (N, 4)."""
    boxes = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue  # spring tomme/ugyldige linjer over
        # YOLO gemmer normaliseret center + størrelse → absolutte hjørner
        _, cx, cy, w, h = (float(v) for v in parts)
        bx, by, bw, bh = cx * image_w, cy * image_h, w * image_w, h * image_h
        boxes.append([bx - bw / 2, by - bh / 2, bx + bw / 2, by + bh / 2])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), np.float32)


def iou_one_vs_many(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    """IoU mellem én box (4,) og N bokse (N, 4), alt i pixel-xyxy."""
    if len(others) == 0:
        return np.zeros(0, np.float32)
    # Skæringsrektanglets hjørner
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)   # overlap-areal
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_oth = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / np.clip(area_box + area_oth - inter, 1e-9, None)  # overlap / forening


def match_by_iou(pred_xyxy: np.ndarray, gt_xyxy: np.ndarray):
    """Greedy: hver prediction tager sit bedste ledige ground-truth.

    Returnerer liste af (pred_idx, gt_idx, iou) for de matchede par.
    """
    used = np.zeros(len(gt_xyxy), dtype=bool)
    matches = []
    for i, pbox in enumerate(pred_xyxy):
        scores = iou_one_vs_many(pbox, gt_xyxy)
        if len(scores) == 0:
            break
        scores = np.where(used, -1.0, scores)   # bloker allerede-matchede GT
        j = int(np.argmax(scores))
        if scores[j] > 0:
            used[j] = True
            matches.append((i, j, float(scores[j])))
    return matches


def draw_gt_and_iou(canvas_rgb: np.ndarray, gt_xyxy: np.ndarray,
                    pred_xyxy: np.ndarray, matches) -> np.ndarray:
    """Tegn ground-truth-bokse (grøn) + IoU-tal pr. par, farvet efter værdi."""
    import cv2

    canvas = canvas_rgb.copy()
    short_side = min(canvas.shape[:2])
    line = max(2, int(short_side / 500))            # skalér til opløsning
    font_scale = max(0.5, short_side / 1400)

    # Ground-truth-bokse i grøn (modellens egne bokse er allerede røde i overlay'et)
    for (x1, y1, x2, y2) in gt_xyxy:
        cv2.rectangle(canvas, (int(x1), int(y1)), (int(x2), int(y2)),
                      (0, 255, 0), line)

    # IoU-tal ved hver matchet prediction, farvekodet efter hvor godt den sidder
    for (pred_idx, _gt_idx, iou) in matches:
        x1, y1, x2, y2 = pred_xyxy[pred_idx]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        color = ((0, 200, 0) if iou >= 0.90        # grøn = sidder stramt
                 else (255, 200, 0) if iou >= 0.75  # gul = ok
                 else (255, 60, 60))                # rød = løst
        cv2.putText(canvas, f"{iou:.2f}", (cx + line, cy - line),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                    max(1, line - 1), cv2.LINE_AA)
    return canvas


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
    help="YOLO ned-skalerer input-billedet til denne størrelse internt. "
         "Større = mere detalje på små frø, men langsommere."
)

st.sidebar.markdown("**Lag** (stables oven på hinanden)")
show_bboxes = st.sidebar.checkbox("Bounding boxes", value=True)
show_centers = st.sidebar.checkbox("Center-markør (bbox-midte)", value=True)
show_masks = st.sidebar.checkbox(
    "Instans-masker",
    value=False,
    help="Segmenterer frø-pixels og farver hvert frø — viser instans-splittingen "
         "af rørende frø (det Videometers blob-analyse laver).",
)
show_com = st.sidebar.checkbox(
    "Center-of-mass (+)",
    value=False,
    help="Frøets faktiske tyngdepunkt (segmenteret), ikke bbox-midten. Det er "
         "den størrelse Videometer rapporterer som X/Y Center mass.",
)
show_orientation = st.sidebar.checkbox(
    "Orienterings-pil",
    value=False,
    help="Frøets hovedakse via 2.-ordens momenter. Meningsfuld for aflange frø; "
         "udefineret for runde.",
)
show_confidence = st.sidebar.checkbox("Confidence-tal", value=False)

# Center-merge: post-processing der erstatter IoU-baseret NMS med en
# afstands-baseret regel. Effektivt mod touching frø hvor bbox'er
# overlapper meget, men centre er langt fra hinanden
st.sidebar.markdown("---")
use_center_merge = st.sidebar.checkbox(
    "Brug center-distance NMS (post-processing)",
    value=False,
    help="Erstatter standard IoU-NMS med en regel baseret på center-afstand. "
         "Når slået til kører inference med iou=0.95 (lader duplikater slippe "
         "igennem) og merger derefter detections hvis deres centre er tæt på "
         "hinanden. Bedre til touching frø — én parameter (afstand) i stedet "
         "for to skidte valg (lav vs høj IoU).",
)
min_center_distance_px = st.sidebar.slider(
    "Min center-afstand (px)",
    min_value=5, max_value=80, value=20, step=1,
    help="To detections merges hvis deres centre er tættere end dette. "
         "Typisk ~halvdelen af en frø-diameter. Kun aktiv når center-merge "
         "er slået til.",
    disabled=not use_center_merge,
)

# Objectness heatmap: hook detection head'en og visualiser klassifikations-
# sandsynligheder pr. grid-celle som farve-overlay — modellens egen "heatmap"
show_heatmap = st.sidebar.checkbox(
    "Vis objectness heatmap",
    value=False,
    help="Henter klassifikations-scores fra YOLO's 3 detection-heads og "
         "visualiserer hvor modellen ser frø som pixel-confidence. Rød = "
         "høj confidence, blå = lav. Tilføjer én ekstra inference-pass når "
         "slået til.",
)

# Ekstra: histogram-fordeling før filtrering — kører en anden inference med
# conf=0.01 så vi kan se ALLE kandidater, ikke kun dem over slideren
show_unfiltered_hist = st.sidebar.checkbox(
    "Vis ufiltreret confidence-fordeling",
    value=False,
    help="Kører en ekstra inference med conf=0.01 og viser histogrammet over "
         "ALLE detections (ikke kun dem over threshold). Fordobler "
         "inference-tiden når slået til. Nuværende threshold markeres "
         "som rød stiplet linje.",
)

# IoU mod ground truth: sammenlign modellens bokse med de sande labels
show_iou = st.sidebar.checkbox(
    "Vis IoU mod ground truth",
    value=False,
    help="Sammenligner modellens bokse med de sande YOLO-labels og tegner "
         "ground-truth + IoU pr. frø. Virker kun for billeder fra "
         "composite_v1_no_leak (fx composite_01600.png), hvor der findes "
         "en .txt-label med samme navn.",
)


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
    st.info("Upload et billede for at se predictions. "
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
    # Hvis center-merge er slået til, vil vi have ALLE dubletter med —
    # derfor høj IoU. Slideren ignoreres i denne mode.
    effective_iou = 0.95 if use_center_merge else iou_threshold

    start = time.time()
    results = model.predict(
        source=np.array(image_pil),
        conf=confidence_threshold,
        iou=effective_iou,
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

# Post-processing: center-distance NMS hvis slået til
n_merged = 0
n_before_merge = len(xyxy)
if use_center_merge and n_before_merge > 0:
    xyxy, confs, n_merged = merge_close_centers(
        xyxy, confs, min_center_distance_px
    )

n_detections = len(xyxy)


# ---------------------------------------------------------------------------
# Tegn prediction-overlay som en stak af valgte lag
# ---------------------------------------------------------------------------
# Segmenteringen (center-of-mass, orientering, masker) er kun nødvendig hvis
# mindst ét af de lag der bruger den er tændt — så beregner vi den kun da.
need_segmentation = show_masks or show_com or show_orientation
seg_labels = None
seg_measurements = None
if need_segmentation and n_detections > 0:
    with st.spinner("Segmenterer frø (center-of-mass + orientering)…"):
        seg_labels, seg_measurements = seed_morphology.analyze(
            np.array(image_pil), xyxy
        )

# Stabl de valgte lag oven på billedet (masker nederst, stregtegning øverst)
overlay = Image.fromarray(
    overlays.render_overlay(
        np.array(image_pil), xyxy, confs,
        show_masks=show_masks,
        show_boxes=show_bboxes,
        show_bbox_center=show_centers,
        show_com=show_com,
        show_orientation=show_orientation,
        show_confidence=show_confidence,
        labels=seg_labels,
        measurements=seg_measurements,
    )
)


with col_right:
    st.subheader("Prediction")
    st.image(overlay, use_container_width=True)
    caption = f"{n_detections} frø detekteret  ·  {elapsed_ms:.0f} ms"
    if use_center_merge:
        caption += (
            f"  ·  {n_merged} dubletter fjernet via center-merge "
            f"(før: {n_before_merge}, afstand ≥ {min_center_distance_px} px)"
        )
    st.caption(caption)


# ---------------------------------------------------------------------------
# Objectness heatmap (modellens interne pixel-confidence)
# ---------------------------------------------------------------------------
if show_heatmap:
    st.divider()
    st.subheader("Objectness heatmap")
    st.caption(
        "Modellens klassifikations-sandsynlighed pr. grid-celle, før NMS og "
        "bbox-decoding. Rød = høj confidence, blå = lav."
    )

    with st.spinner("Beregner heatmap..."):
        heatmaps = compute_objectness_heatmaps(
            model, np.array(image_pil), imgsz=imgsz, device="mps"
        )

    # Kombineret view: upsample alle 3 til P3-størrelsen, så pixel-wise max
    target_h, target_w = heatmaps[0].shape
    combined = heatmaps[0].copy()
    for hm in heatmaps[1:]:
        hm_resized = np.array(
            Image.fromarray((np.clip(hm, 0, 1) * 255).astype(np.uint8), mode="L")
            .resize((target_w, target_h), resample=Image.BILINEAR)
        ).astype(np.float32) / 255.0
        combined = np.maximum(combined, hm_resized)

    combined_overlay = make_heatmap_overlay(image_pil, combined, alpha=0.5)
    st.image(
        combined_overlay,
        use_container_width=True,
        caption=f"Kombineret (max af P3+P4+P5) · grid {target_w}×{target_h}",
    )

    with st.expander("Pr. detection-scale (P3 fineste → P5 groveste)"):
        cols = st.columns(3)
        for col, hm, scale_label in zip(cols, heatmaps, ["P3", "P4", "P5"]):
            scale_overlay = make_heatmap_overlay(image_pil, hm, alpha=0.5)
            with col:
                st.image(
                    scale_overlay,
                    use_container_width=True,
                    caption=f"{scale_label}: {hm.shape[1]}×{hm.shape[0]} grid",
                )


# ---------------------------------------------------------------------------
# IoU mod ground truth (kun når billedet har kendte labels)
# ---------------------------------------------------------------------------
if show_iou:
    st.divider()
    st.subheader("IoU mod ground truth")
    gt_label_path = find_matching_gt(uploaded.name, np.array(image_pil))
    if gt_label_path is None:
        st.warning(
            f"Intet matchende ground-truth fundet for `{uploaded.name}`. IoU "
            "kræver et billede der er **pixel-identisk** med et billede i "
            "composite_v1_no_leak (val/train) — fx `composite_01600.png` "
            "taget direkte fra datasættet. Et billede fra `test_images/` med "
            "samme navn som et træningsbillede tæller IKKE — det er et andet "
            "billede."
        )
    elif n_detections == 0:
        st.warning("Ingen detektioner at sammenligne mod ground truth.")
    else:
        gt_xyxy = load_gt_boxes(gt_label_path, img_w, img_h)
        matches = match_by_iou(xyxy, gt_xyxy)
        ious = np.array([m[2] for m in matches], dtype=np.float32)

        iou_image = draw_gt_and_iou(np.array(overlay), gt_xyxy, xyxy, matches)
        st.image(
            iou_image,
            use_container_width=True,
            caption="Grøn = ground truth · rød = modellens box · tal = IoU pr. par "
                    "(grøn ≥ 0.90, gul ≥ 0.75, rød < 0.75)",
        )

        if len(ious) > 0:
            iou_a, iou_b, iou_c, iou_d = st.columns(4)
            iou_a.metric("Median IoU", f"{np.median(ious):.3f}")
            iou_b.metric("Matchede par", f"{len(ious)} / {len(gt_xyxy)}")
            iou_c.metric("IoU ≥ 0.50", f"{100 * (ious >= 0.50).mean():.0f}%")
            iou_d.metric("IoU ≥ 0.95", f"{100 * (ious >= 0.95).mean():.0f}%")
            st.caption(
                f"Ground truth: {len(gt_xyxy)} frø · detektioner: {n_detections} · "
                f"mean IoU {ious.mean():.3f} · min/max "
                f"{ious.min():.3f}/{ious.max():.3f}. Andelen falder ved højere "
                "tærskel — netop derfor er mAP@50 ~1,0 mens mAP@50-95 = 0,97."
            )
        else:
            st.warning("Ingen detektioner overlappede med ground truth.")


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
    import altair as alt

    st.subheader("Confidence-fordeling")

    # Vælg datakilde: enten de filtrerede detections eller en ny inference
    # med lav threshold for at se den fulde fordeling
    if show_unfiltered_hist:
        with st.spinner("Kører ufiltreret inference…"):
            results_full = model.predict(
                source=np.array(image_pil),
                conf=0.01,
                iou=iou_threshold,
                imgsz=imgsz,
                device="mps",
                verbose=False,
            )
        boxes_full = results_full[0].boxes
        confs_for_hist = (
            boxes_full.conf.cpu().numpy()
            if boxes_full is not None else np.zeros(0)
        )
        st.caption(
            f"Viser {len(confs_for_hist)} detections med conf ≥ 0.01 "
            f"(rød stiplet linje = threshold {confidence_threshold:.2f})"
        )
    else:
        confs_for_hist = confs
        st.caption(
            f"Viser {n_detections} detections over threshold "
            f"({confidence_threshold:.2f})"
        )

    # Lav histogrammet med altair så x-aksen får ordentlige numeriske labels
    n_bins = 20 if show_unfiltered_hist else 10
    df_hist = pd.DataFrame({"Confidence": confs_for_hist})
    hist_chart = (
        alt.Chart(df_hist)
        .mark_bar(color="#3b82f6")
        .encode(
            x=alt.X(
                "Confidence:Q",
                bin=alt.Bin(maxbins=n_bins),
                title="Confidence",
                axis=alt.Axis(format=".2f"),
            ),
            y=alt.Y("count():Q", title="Antal detections"),
            tooltip=[
                alt.Tooltip("Confidence:Q", bin=alt.Bin(maxbins=n_bins), format=".2f"),
                alt.Tooltip("count():Q", title="Antal"),
            ],
        )
    )

    # Tilføj rød stiplet linje ved den valgte threshold når vi viser ufiltreret
    if show_unfiltered_hist:
        threshold_rule = (
            alt.Chart(pd.DataFrame({"x": [confidence_threshold]}))
            .mark_rule(color="red", strokeDash=[6, 4], size=2)
            .encode(x="x:Q")
        )
        hist_chart = hist_chart + threshold_rule

    st.altair_chart(hist_chart, use_container_width=True)

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
    label="Download prediction-overlay",
    data=buf.getvalue(),
    file_name=f"prediction_{uploaded.name.rsplit('.', 1)[0]}.png",
    mime="image/png",
)


# ---------------------------------------------------------------------------
# Footer: model-detaljer (altid synlig)
# ---------------------------------------------------------------------------
st.divider()

meta = load_model_metadata(active_version, model)

st.markdown(
    f"### Model: `{meta['version']}` · "
    f"{meta.get('arch', 'YOLO')} · "
    f"Trænet på `{meta.get('dataset_name', '?')}`"
)

# To kolonner: performance og arkitektur side om side
col_perf, col_arch = st.columns(2)

with col_perf:
    st.markdown("**Performance**")
    perf_data = [
        ("mAP50",     f"{meta.get('mAP50', 0):.4f}"),
        ("mAP50-95",  f"{meta.get('mAP50_95', 0):.4f}"),
        ("Precision", f"{meta.get('precision', 0):.4f}"),
        ("Recall",    f"{meta.get('recall', 0):.4f}"),
    ]
    perf_df = pd.DataFrame(perf_data, columns=["Metric", "Value"]).set_index("Metric")
    st.table(perf_df)

with col_arch:
    st.markdown("**Arkitektur**")
    params_m = meta.get("parameters", 0) / 1e6
    grads_m = meta.get("gradients", 0) / 1e6
    arch_data = [
        ("Layers",          f"{meta.get('layers', '?')}"),
        ("Parameters",      f"{params_m:.2f} M"),
        ("Gradients",       f"{grads_m:.2f} M"),
        ("Model-størrelse", f"{meta.get('file_size_mb', 0):.1f} MB"),
    ]
    arch_df = pd.DataFrame(arch_data, columns=["Attribut", "Value"]).set_index("Attribut")
    st.table(arch_df)

# Training-config og dataset-info som tekst-linjer
st.markdown("**Training-config**")
config_parts = [
    f"{meta.get('epochs', '?')} epochs",
    f"imgsz {meta.get('imgsz', '?')}",
    f"batch {meta.get('batch', '?')}",
    f"patience {meta.get('patience', '?')}",
    f"device {meta.get('device', '?')}",
]
st.text(" · ".join(str(p) for p in config_parts))
st.text(
    f"Dataset: {meta.get('dataset_name', '?')} "
    f"({meta.get('train_count', '?')} train / {meta.get('val_count', '?')} val)"
)
