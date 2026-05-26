"""Resize composite-datasæt til upload-venlig størrelse før Colab-træning.

Original composite-billeder er 4096×2704 RGB PNGs (~11 MB hver, 23 GB samlet
for 2001 billeder). Det er for stort til at uploade praktisk. Da Ultralytics
alligevel ned-skalerer til imgsz=1280 internt under træning, kan vi resize
på Mac'en uden tab.

Bilinear interpolation matcher Ultralytics' LetterBox-resize, så de
output-billeder denne pipeline producerer er pixel-identiske med de
tensorer v5_no_leak's pipeline genererede internt. Sammenligningen mellem
v5 (YOLO11n) og v6 (YOLOv10n) er derfor 100% ren mht. input-data.

Labels kopieres uændret fordi YOLO-format er normaliseret (cx/W, cy/H,
w/W, h/H — alle 0-1), og resize bevarer aspect ratio.

Brug:
    .venv/bin/python scripts/02_yolo_pipeline/resize_for_colab.py \\
        --src data/composite_v1_no_leak \\
        --dst data/composite_v1_no_leak_1280

Output: ~2-3 GB datasæt klar til at pakke + uploade.
"""
import argparse
import shutil
import sys
import time
from pathlib import Path

import yaml
from PIL import Image


def parse_args() -> argparse.Namespace:
    """CLI-args med fornuftige defaults."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--src", type=Path, required=True,
        help="Kilde-datasæt (skal indeholde images/train, images/val, labels/, data.yaml)",
    )
    parser.add_argument(
        "--dst", type=Path, required=True,
        help="Destinations-datasæt — oprettes hvis den ikke findes, fejler hvis den allerede gør",
    )
    parser.add_argument(
        "--max-size", type=int, default=1280,
        help="Største dimension efter resize. Default 1280 matcher Ultralytics' imgsz.",
    )
    return parser.parse_args()


def resize_image(src_path: Path, dst_path: Path, max_size: int) -> None:
    """Resize ét PNG og bevar aspect ratio. Bilinear matcher Ultralytics LetterBox."""
    img = Image.open(src_path)
    width, height = img.size
    scale = max_size / max(width, height)
    if scale >= 1.0:
        # Billedet er allerede mindre end target — bare kopier filen
        shutil.copyfile(src_path, dst_path)
        return
    new_size = (int(round(width * scale)), int(round(height * scale)))
    resized = img.resize(new_size, resample=Image.BILINEAR)
    # optimize=True kan tage lidt ekstra tid men sparer ~5-10% disk
    resized.save(dst_path, format="PNG", optimize=True)


def main() -> int:
    args = parse_args()
    src = args.src.resolve()
    dst = args.dst.resolve()

    if not src.exists():
        print(f"Kilde-datasæt findes ikke: {src}", file=sys.stderr)
        return 1
    if dst.exists():
        print(f"Destinations-mappe findes allerede: {dst}", file=sys.stderr)
        print("  Slet den først eller vælg en anden sti.", file=sys.stderr)
        return 1

    # Forbered destinations-struktur
    for split in ("train", "val"):
        (dst / "images" / split).mkdir(parents=True, exist_ok=True)

    # Resize alle billeder, split for split
    total_images = 0
    start = time.time()
    for split in ("train", "val"):
        src_dir = src / "images" / split
        dst_dir = dst / "images" / split
        if not src_dir.exists():
            print(f"Springer {split} over — {src_dir} findes ikke")
            continue
        images = sorted(src_dir.glob("*.png"))
        print(f"Resizing {len(images)} billeder fra {split}...")
        for i, img_path in enumerate(images):
            resize_image(img_path, dst_dir / img_path.name, args.max_size)
            total_images += 1
            # Progress hver 100 billeder
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start
                rate = total_images / elapsed
                remaining = (len(images) - i - 1) / rate / 60
                print(f"  {i+1}/{len(images)} ({rate:.1f}/sek, ETA {remaining:.1f} min)")

    # Labels kopieres uændret — YOLO-format er normaliseret, uafhængigt af opløsning
    print("Kopierer labels (uændret)...")
    shutil.copytree(src / "labels", dst / "labels")

    # data.yaml: opdater path så den peger på den nye dataset-rod
    print("Skriver opdateret data.yaml...")
    with open(src / "data.yaml") as f:
        cfg = yaml.safe_load(f) or {}
    # Brug den nye datasæt-mappes relative sti til projekt-rod
    project_root = Path(__file__).resolve().parent.parent.parent
    cfg["path"] = str(dst.relative_to(project_root))
    with open(dst / "data.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    # Rapportér resultat
    total_bytes = sum(p.stat().st_size for p in dst.rglob("*") if p.is_file())
    total_gb = total_bytes / 1024**3
    elapsed_min = (time.time() - start) / 60
    print()
    print("=== FÆRDIG ===")
    print(f"Resized {total_images} billeder på {elapsed_min:.1f} min")
    print(f"Total størrelse: {total_gb:.2f} GB")
    print(f"Output: {dst}")
    print()
    print("Næste skridt — pak op til upload:")
    print(f"  tar czf {dst.name}.tar.gz {dst.relative_to(project_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
