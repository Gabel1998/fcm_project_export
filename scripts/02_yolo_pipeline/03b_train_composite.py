"""
=====================================================================
 Step 3b: Træn YOLOv11n på composite-datasæt (rigtige Videometer-frø)
=====================================================================

 Adskilt fra 03_train_yolo.py fordi:
   - Composite-billeder er 4096x2704 (ellipse var 256x256)
   - Vi bruger imgsz=1280 som kompromis mellem detalje og MPS-memory
   - Træningstime augmentations skal være andre (composites har egen variation)

 Brug:
   .venv/bin/python scripts/02_yolo_pipeline/03b_train_composite.py
   .venv/bin/python scripts/02_yolo_pipeline/03b_train_composite.py --dataset v2_dense --name v5_dense

 CLI args (alle optional):
   --dataset NAME      Dataset-mappe under data/ (default: composite_v1)
   --name NAME         Run-navn under runs/train/ (default: v4_composite)
   --epochs N          Antal epochs (default: 80)
   --imgsz N           YOLO input-størrelse (default: 1280)
   --batch N           Batch-størrelse (default: 4 — bump op til 8 hvis du har RAM)
   --patience N        Early stopping patience (default: 15)
   --device DEV        mps / cpu / cuda (default: mps)
"""
import argparse
import sys
from pathlib import Path
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    """CLI-args med fornuftige defaults til composite-træning."""
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="composite_v1",
                         help="dataset-mappe under data/")
    parser.add_argument("--name", default="v4_composite",
                         help="run-navn under runs/train/")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=1280,
                         help="YOLO input-størrelse (composites ned-skaleres internt)")
    parser.add_argument("--batch", type=int, default=4,
                         help="batch-størrelse — øg til 8 hvis MPS har RAM nok")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="mps",
                         help="mps (Apple Silicon), cuda, eller cpu")
    parser.add_argument("--pretrained", default="yolo11n.pt",
                         help="pretrained weights — auto-downloades hvis ikke lokalt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent.parent
    data_yaml = project_root / "data" / args.dataset / "data.yaml"

    if not data_yaml.exists():
        print(f"❌ Mangler {data_yaml}", file=sys.stderr)
        print(f"   Kør 'make gen DATASET_NAME={args.dataset}' først.", file=sys.stderr)
        return 1

    print(f"Datasæt:    {data_yaml}")
    print(f"Run-navn:   runs/train/{args.name}")
    print(f"Epochs:     {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch:      {args.batch}")
    print(f"Device:     {args.device}")
    print()

    model = YOLO(args.pretrained)

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        project="runs/train",
        name=args.name,
        exist_ok=True,
        verbose=True,
        # Augmentations — composite-data har allerede god variation,
        # men disse hjælper modellen blive robust mod fuld-billede-transformer
        flipud=0.5,
        fliplr=0.5,
        degrees=180.0,       # frø kan ligge i alle retninger
        mosaic=1.0,          # YOLOv11's mosaic-augmentation
        mixup=0.15,
        copy_paste=0.3,
    )

    print("\n" + "=" * 60)
    print("TRÆNING FÆRDIG")
    print("=" * 60)
    print(f"\nFinal metrics:")
    print(f"  Precision: {results.box.mp:.4f}    (Jons krav: > 0.5)")
    print(f"  Recall:    {results.box.mr:.4f}    (Jons krav: > 0.3)")
    print(f"  mAP@50:    {results.box.map50:.4f}    (Jons krav: > 0.2)")
    print(f"  mAP@50-95: {results.box.map:.4f}    (bonus)")
    print(f"\nBest model: runs/train/{args.name}/weights/best.pt")
    print(f"Plots:      runs/train/{args.name}/results.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
