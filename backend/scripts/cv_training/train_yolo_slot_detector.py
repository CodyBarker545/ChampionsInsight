"""Train a YOLO model for opponent red slot detection.

Run from backend after building the dataset:
    python scripts/train_yolo_slot_detector.py

The trained model is written under:
    backend/data/cv/models/slot_detector/
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_YAML = BACKEND_DIR / "data" / "training_dataset" / "yolo_slot_detector" / "data.yaml"
DEFAULT_PROJECT_DIR = BACKEND_DIR / "data" / "cv" / "models" / "slot_detector"
DEFAULT_YOLO_CONFIG_DIR = BACKEND_DIR / "Ultralytics"


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
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Stop early if validation metrics do not improve for this many epochs.",
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--device",
        default="auto",
        help="Use auto, cpu, 0, 0,1, etc. auto selects CUDA device 0 when available.",
    )
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
    os.environ.setdefault("YOLO_CONFIG_DIR", str(DEFAULT_YOLO_CONFIG_DIR))

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

    device = args.device
    if device == "auto":
        try:
            import torch

            device = "0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    if device == "cpu":
        print("WARNING: Training is using CPU. Install CUDA-enabled PyTorch or pass --device 0 for GPU.")
    else:
        print(f"Training device: {device}")

    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "patience": args.patience,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": str(project_dir),
        "name": args.name,
        "workers": args.workers,
        "exist_ok": True,
    }

    if device is not None:
        train_kwargs["device"] = device

    result = model.train(**train_kwargs)

    print()
    print("Training complete.")
    print(f"Project: {project_dir}")
    print(f"Run name: {args.name}")
    print(f"Best weights: {project_dir / args.name / 'weights' / 'best.pt'}")
    print(result)


if __name__ == "__main__":
    main()

