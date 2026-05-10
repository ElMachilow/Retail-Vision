from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def train(args: argparse.Namespace) -> Path:
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise SystemExit("No se pudo importar ultralytics. Instala requirements.txt primero.") from exc

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        exist_ok=args.exist_ok,
        patience=args.patience,
        seed=args.seed,
    )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise SystemExit(f"No se encontró best.pt en {best}")

    args.export_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, args.export_path)
    print(f"Mejor modelo YOLO: {best}")
    print(f"Modelo copiado para la API: {args.export_path}")
    print(f"Configura YOLO_MODEL_PATH={args.export_path.as_posix()} para usarlo en FastAPI.")
    return args.export_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena YOLOv8n con el dataset demo.")
    parser.add_argument("--data", type=Path, default=Path("data/demo_yolo/dataset.yaml"))
    parser.add_argument("--model", default="yolov8n.pt", help="Modelo base YOLO.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", type=Path, default=Path("models/runs"))
    parser.add_argument("--name", default="demo_product_label")
    parser.add_argument("--exist-ok", action="store_true", default=True)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--export-path", type=Path, default=Path("models/product_label_demo.pt"))
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()

