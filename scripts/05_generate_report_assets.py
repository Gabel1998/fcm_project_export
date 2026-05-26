#!/usr/bin/env python3
"""Generér alle billeder brugt i REPORT.md.

Henter:
- Træningskurver, PR-kurve, confusion matrix fra runs/detect/runs/train/v5_no_leak/
- Validation-predictions fra samme mappe
- Et nedskaleret syntetisk eksempel-billede

Genererer:
- Prediction-overlay på composite_00000.png med v5_no_leak-modellen
- Objectness heatmap-overlay (kombineret P3+P4+P5)

Kør:
    .venv/bin/python scripts/05_generate_report_assets.py

Bemærk: heatmap-logikken er bevidst en kopi af compute_objectness_heatmaps()
i scripts/03_demo/upload_demo.py — sidstnævnte er en Streamlit-app der ikke
kan importeres uden side-effekter, så funktionen er duplikeret her for at
holde dette script selvstændigt.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from matplotlib import colormaps
from ultralytics import YOLO

# Stier løses relativt til repo-roden så scriptet kan kaldes fra hvor som helst
ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs" / "detect" / "runs" / "train" / "v5_no_leak"
MODEL_PATH = ROOT / "models" / "v5_no_leak" / "best.pt"
V4_MODEL_PATH = ROOT / "models" / "v4_composite" / "best.pt"
TEST_IMG = ROOT / "test_images" / "composite_00000.png"
OUT_DIR = ROOT / "docs" / "screenshots"

IMGSZ = 1280  # Samme input-størrelse som under træning


def device_str() -> str:
    """Vælg MPS hvis tilgængelig (Mac M-chip), ellers CPU."""
    return "mps" if torch.backends.mps.is_available() else "cpu"


def copy_training_assets() -> None:
    """Kopiér eksisterende træningsbilleder fra runs/ med pænere navne."""
    pairs = [
        ("results.png", "training_curves.png"),
        ("BoxPR_curve.png", "pr_curve.png"),
        ("confusion_matrix.png", "confusion_matrix.png"),
        ("val_batch0_pred.jpg", "val_predictions.jpg"),
    ]
    for src_name, dst_name in pairs:
        src = RUNS_DIR / src_name
        dst = OUT_DIR / dst_name
        if not src.exists():
            print(f"  ADVARSEL: {src} findes ikke — springes over")
            continue
        shutil.copy(src, dst)
        print(f"  kopieret: {src_name} -> {dst_name}")


def generate_synthetic_example() -> None:
    """Halvér opløsningen af composite_00000.png så filen ikke er 12 MB i docs/."""
    img = Image.open(TEST_IMG)
    new_size = (img.width // 2, img.height // 2)
    img.resize(new_size, Image.LANCZOS).save(
        OUT_DIR / "synthetic_example.png", optimize=True
    )
    print(f"  genereret: synthetic_example.png ({new_size[0]}x{new_size[1]})")


def generate_prediction_overlay(model: YOLO, device: str) -> None:
    """Tegn bounding boxes på test-billedet med Ultralytics' indbyggede plot()."""
    results = model.predict(
        source=str(TEST_IMG),
        imgsz=IMGSZ,
        conf=0.25,
        device=device,
        save=False,
        verbose=False,
    )
    plotted_bgr = results[0].plot()  # numpy array, BGR
    plotted_rgb = plotted_bgr[:, :, ::-1]
    img = Image.fromarray(plotted_rgb)
    # Halvér til konsistens med synthetic_example
    img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
    img.save(OUT_DIR / "demo_predictions.png", optimize=True)
    n_dets = len(results[0].boxes) if results[0].boxes is not None else 0
    print(f"  genereret: demo_predictions.png ({n_dets} detektioner)")


def compute_objectness_heatmaps(
    model: YOLO, image_array: np.ndarray, device: str
) -> list[np.ndarray]:
    """Hook YOLO's detection-head og fang klassifikations-sandsynligheder pr. scale.

    Logik kopieret 1:1 fra scripts/03_demo/upload_demo.py — samme model,
    samme hooks, samme sigmoid på sidste cv3-conv pr. skala (P3, P4, P5).
    """
    # Dict så seneste capture pr. scale overskriver tidligere (warmup-pass
    # giver ellers et output ≈ 0)
    captured: dict[int, np.ndarray] = {}

    def make_hook(scale_idx: int):
        def hook(_module, _inp, out):
            prob = torch.sigmoid(out).detach().cpu().numpy()[0, 0]
            captured[scale_idx] = prob
        return hook

    detect = model.model.model[-1]
    hooks = []
    for i, cv3_branch in enumerate(detect.cv3):
        last_conv = cv3_branch[-1]
        hooks.append(last_conv.register_forward_hook(make_hook(i)))

    try:
        model.predict(image_array, imgsz=IMGSZ, device=device, conf=0.01, verbose=False)
    finally:
        for h in hooks:
            h.remove()

    return [captured[i] for i in sorted(captured.keys())]


def make_heatmap_overlay(
    base_img: Image.Image, heatmap: np.ndarray, alpha: float = 0.5
) -> Image.Image:
    """Læg 2D heatmap som jet-colormap-overlay på et PIL-billede.

    Logik kopieret 1:1 fra scripts/03_demo/upload_demo.py.
    """
    target_w, target_h = base_img.size

    hm_pil = Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8), mode="L")
    hm_resized = hm_pil.resize((target_w, target_h), resample=Image.BILINEAR)
    hm_norm = np.array(hm_resized).astype(np.float32) / 255.0

    cmap = colormaps["jet"]
    colored = (cmap(hm_norm)[:, :, :3] * 255).astype(np.uint8)

    base_array = np.array(base_img.convert("RGB")).astype(np.float32)
    blended = (1 - alpha) * base_array + alpha * colored.astype(np.float32)
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def generate_heatmap_overlay(model: YOLO, device: str) -> None:
    """Generér samlet objectness heatmap-overlay (P3+P4+P5 summeret)."""
    base = Image.open(TEST_IMG).convert("RGB")
    image_array = np.array(base)

    heatmaps = compute_objectness_heatmaps(model, image_array, device)

    # Upsample alle 3 skalaer til samme størrelse (P3's størrelse) og summér
    target_h, target_w = heatmaps[0].shape
    combined = heatmaps[0].copy()
    for hm in heatmaps[1:]:
        hm_pil = Image.fromarray((np.clip(hm, 0, 1) * 255).astype(np.uint8), mode="L")
        hm_up = np.array(
            hm_pil.resize((target_w, target_h), resample=Image.BILINEAR)
        ).astype(np.float32) / 255.0
        combined += hm_up
    combined = combined / max(combined.max(), 1e-6)

    overlay = make_heatmap_overlay(base, combined, alpha=0.5)
    overlay = overlay.resize((overlay.width // 2, overlay.height // 2), Image.LANCZOS)
    overlay.save(OUT_DIR / "objectness_heatmap.png", optimize=True)
    print("  genereret: objectness_heatmap.png")


def _get_font(size: int) -> ImageFont.ImageFont:
    """Hent en system-font med graceful fallback til default."""
    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _predict_and_plot(model: YOLO, device: str) -> Image.Image:
    """Kør prediction på testbilledet og returnér plot som PIL-billede."""
    results = model.predict(
        source=str(TEST_IMG),
        imgsz=IMGSZ,
        conf=0.25,
        device=device,
        save=False,
        verbose=False,
    )
    plotted_bgr = results[0].plot()
    return Image.fromarray(plotted_bgr[:, :, ::-1]), len(results[0].boxes) if results[0].boxes is not None else 0


def generate_v4_vs_v5_comparison(device: str) -> None:
    """Side-by-side sammenligning af v4_composite og v5_no_leak på samme billede.

    Viser visuelt hvordan data leakage påvirkede v4's predictions sammenlignet
    med den leakage-frie v5-træning. Layoutet er HORISONTALT (v4 til venstre,
    v5 til højre) så aspect ratio passer på en A4-side uden at fylde hele
    højden.
    """
    if not V4_MODEL_PATH.exists():
        print(f"  ADVARSEL: {V4_MODEL_PATH} findes ikke — v4-sammenligning skippes")
        return

    v4 = YOLO(str(V4_MODEL_PATH))
    v5 = YOLO(str(MODEL_PATH))

    v4_img, v4_dets = _predict_and_plot(v4, device)
    v5_img, v5_dets = _predict_and_plot(v5, device)

    # Begge billeder har samme størrelse — halvér dem først
    new_size = (v4_img.width // 2, v4_img.height // 2)
    v4_img = v4_img.resize(new_size, Image.LANCZOS)
    v5_img = v5_img.resize(new_size, Image.LANCZOS)

    # Banner over hvert billede med label + antal detektioner
    banner_h = 40
    font = _get_font(22)
    img_w, img_h = new_size
    canvas_w = img_w * 2
    canvas_h = img_h + banner_h

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    # Venstre: v4
    draw.rectangle([0, 0, img_w, banner_h], fill=(40, 40, 40))
    draw.text((12, 8), f"v4_composite  (data leakage)  —  {v4_dets} detektioner",
              fill=(255, 220, 100), font=font)
    canvas.paste(v4_img, (0, banner_h))

    # Højre: v5
    draw.rectangle([img_w, 0, canvas_w, banner_h], fill=(40, 40, 40))
    draw.text((img_w + 12, 8), f"v5_no_leak  (leakage-fri splits)  —  {v5_dets} detektioner",
              fill=(120, 230, 120), font=font)
    canvas.paste(v5_img, (img_w, banner_h))

    canvas.save(OUT_DIR / "v4_vs_v5_comparison.png", optimize=True)
    print(f"  genereret: v4_vs_v5_comparison.png (v4={v4_dets}, v5={v5_dets} detektioner)")


def generate_heatmaps_per_scale(model: YOLO, device: str) -> None:
    """3 heatmaps side-by-side (P3, P4, P5) som viser multi-scale-arkitekturen.

    P3 = fineste grid (små objekter), P5 = groveste (store objekter).
    """
    base = Image.open(TEST_IMG).convert("RGB")
    image_array = np.array(base)
    heatmaps = compute_objectness_heatmaps(model, image_array, device)

    # Halvér base først — alle 3 overlays skal samme størrelse
    target_size = (base.width // 2, base.height // 2)
    base_small = base.resize(target_size, Image.LANCZOS)

    overlays = []
    for hm in heatmaps:
        overlay = make_heatmap_overlay(base_small, hm, alpha=0.55)
        overlays.append(overlay)

    # Banner-info pr. heatmap
    labels = [
        ("P3", "fineste skala — små objekter", heatmaps[0].shape),
        ("P4", "mellem skala", heatmaps[1].shape),
        ("P5", "groveste skala — store objekter", heatmaps[2].shape),
    ]
    banner_h = 40
    font_big = _get_font(22)
    font_small = _get_font(15)

    w, h = target_size
    canvas_w = w * 3
    canvas_h = h + banner_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    for i, (overlay, (scale, desc, shape)) in enumerate(zip(overlays, labels)):
        x = i * w
        draw.rectangle([x, 0, x + w, banner_h], fill=(40, 40, 40))
        draw.text((x + 10, 4),  f"{scale}", fill=(255, 255, 255), font=font_big)
        draw.text((x + 50, 8),  f"({shape[1]}x{shape[0]} grid) — {desc}",
                  fill=(200, 200, 200), font=font_small)
        canvas.paste(overlay, (x, banner_h))

    canvas.save(OUT_DIR / "heatmaps_per_scale.png", optimize=True)
    print(f"  genereret: heatmaps_per_scale.png (grid-størrelser: "
          f"{heatmaps[0].shape}, {heatmaps[1].shape}, {heatmaps[2].shape})")


def copy_train_batch() -> None:
    """Kopiér train_batch0.jpg fra runs/ — viser et batch med augmentation."""
    src = RUNS_DIR / "train_batch0.jpg"
    if not src.exists():
        print(f"  ADVARSEL: {src} findes ikke — train_batch springes over")
        return
    shutil.copy(src, OUT_DIR / "train_batch_augmented.jpg")
    print("  kopieret: train_batch0.jpg -> train_batch_augmented.jpg")


def main() -> None:
    if not MODEL_PATH.exists():
        print(f"FEJL: modellen findes ikke på {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)
    if not TEST_IMG.exists():
        print(f"FEJL: testbilledet findes ikke på {TEST_IMG}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = device_str()
    print(f"Device: {device}")

    print("[1/5] Kopierer træningsoutputs...")
    copy_training_assets()
    copy_train_batch()

    print("[2/5] Genererer syntetisk eksempel-billede...")
    generate_synthetic_example()

    print("[3/5] Loader v5_no_leak-modellen og genererer predictions + heatmap...")
    model = YOLO(str(MODEL_PATH))
    generate_prediction_overlay(model, device)
    generate_heatmap_overlay(model, device)

    print("[4/5] Genererer heatmaps pr. skala (P3, P4, P5)...")
    generate_heatmaps_per_scale(model, device)

    print("[5/5] Genererer v4 vs. v5 sammenligning...")
    generate_v4_vs_v5_comparison(device)

    print(f"\nFærdig. Alle assets ligger i {OUT_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
