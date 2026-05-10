"""Train a YOLO model for opponent red slot detection.

Run from backend after building the dataset:
    python scripts/train_yolo_slot_detector.py

The trained model is written under:
    backend/data/cv/models/slot_detector/
"""

from __future__ import annotations

import argparse
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_YAML = BACKEND_DIR / "data" / "training_dataset" / "yolo_slot_detector" / "data.yaml"
DEFAULT_PROJECT_DIR = BACKEND_DIR / "data" / "cv" / "models" / "slot_detector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO opponent slot detector.")
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_YAML,
        help="YOLO data.yaml path.",
    )
    parser.add_argument(
        "--model",
        default="yolov8n.pt",
        help="Base YOLO weights, for example yolov8n.pt or a local .pt file.",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None, help="Use cpu, 0, 0,1, etc.")
    parser.add_argument(
        "--project",
        type=Path,
        default=DEFAULT_PROJECT_DIR,
        help="Training output root.",
    )
    parser.add_argument("--name", default="yolov8n_slots")
    parser.add_argument("--workers", type=int, default=0, help="Use 0 on Windows for fewer loader issues.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = args.data if args.data.is_absolute() else BACKEND_DIR / args.data
    project_dir = args.project if args.project.is_absolute() else BACKEND_DIR / args.project

    if not data_path.exists():
        raise FileNotFoundError(
            f"YOLO data file not found: {data_path}\n"
            "Run scripts/build_yolo_slot_dataset.py first."
        )

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError(
            "ultralytics is required for YOLO training. "
            "Install backend requirements first: pip install -r requirements.txt"
        ) from error

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(project_dir),
        "name": args.name,
        "workers": args.workers,
        "exist_ok": True,
    }

    if args.device is not None:
        train_kwargs["device"] = args.device

    result = model.train(**train_kwargs)

    print()
    print("Training complete.")
    print(f"Project: {project_dir}")
    print(f"Run name: {args.name}")
    print(f"Best weights: {project_dir / args.name / 'weights' / 'best.pt'}")
    print(result)


if __name__ == "__main__":
    main()
