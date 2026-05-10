"""Build a YOLO dataset from real team-preview originals.

This script takes full original team-preview images, uses an existing YOLO slot
detector to auto-label six opponent_slot boxes per image, and writes a standard
YOLO dataset:

    data/training_dataset/yolo_slot_detector_real/
      images/train
      images/val
      labels/train
      labels/val
      data.yaml

Run from backend:
    python scripts/build_yolo_real_slot_dataset.py ^
      --model data/cv/models/slot_detector/yolov8n_slots_cpu_smoke/weights/best.pt
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = BACKEND_DIR / "data" / "training_dataset" / "yolo_slot_detector_real_originals"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "data" / "training_dataset" / "yolo_slot_detector_real"
DEFAULT_MODEL_PATH = (
    BACKEND_DIR
    / "data"
    / "cv"
    / "models"
    / "slot_detector"
    / "yolov8n_slots_cpu_smoke"
    / "weights"
    / "best.pt"
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_ID = 0
CLASS_NAME = "opponent_slot"


def resolve_backend_path(path: Path) -> Path:
    return path if path.is_absolute() else BACKEND_DIR / path


def list_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "overlays" / split).mkdir(parents=True, exist_ok=True)


def box_center_y(detection: dict) -> float:
    box = detection["box"]
    return box["y"] + box["height"] / 2


def filter_vertical_slot_boxes(detections: list[dict], max_boxes: int) -> list[dict]:
    if len(detections) <= max_boxes:
        return sorted(detections, key=lambda item: item["box"]["y"])

    sorted_by_y = sorted(detections, key=box_center_y)
    median_height = sorted(item["box"]["height"] for item in sorted_by_y)[len(sorted_by_y) // 2]
    same_row_threshold = max(24, median_height * 0.55)

    groups: list[list[dict]] = []
    for detection in sorted_by_y:
        center_y = box_center_y(detection)
        matching_group = None

        for group in groups:
            group_center = sum(box_center_y(item) for item in group) / len(group)
            if abs(center_y - group_center) <= same_row_threshold:
                matching_group = group
                break

        if matching_group is None:
            groups.append([detection])
        else:
            matching_group.append(detection)

    best_per_group = [
        max(group, key=lambda item: item["confidence"])
        for group in groups
    ]

    if len(best_per_group) > max_boxes:
        best_per_group = sorted(
            best_per_group,
            key=lambda item: item["confidence"],
            reverse=True,
        )[:max_boxes]

    return sorted(best_per_group, key=lambda item: item["box"]["y"])


def yolo_label_line(box: dict, image_width: int, image_height: int) -> str:
    x_center = (box["x"] + box["width"] / 2) / image_width
    y_center = (box["y"] + box["height"] / 2) / image_height
    width = box["width"] / image_width
    height = box["height"] / image_height

    return f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def detect_slots(model, image_path: Path, conf: float, imgsz: int, max_boxes: int) -> tuple[list[dict], int]:
    results = model.predict(source=str(image_path), conf=conf, imgsz=imgsz, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            detections.append(
                {
                    "classId": class_id,
                    "confidence": round(confidence, 4),
                    "box": {
                        "x": x1,
                        "y": y1,
                        "width": max(0, x2 - x1),
                        "height": max(0, y2 - y1),
                    },
                }
            )

    raw_count = len(detections)
    return filter_vertical_slot_boxes(detections, max_boxes), raw_count


def draw_overlay(image_path: Path, detections: list[dict], output_path: Path) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return

    for index, detection in enumerate(detections, start=1):
        box = detection["box"]
        x1 = box["x"]
        y1 = box["y"]
        x2 = x1 + box["width"]
        y2 = y1 + box["height"]

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 4)
        cv2.putText(
            image,
            f"{index} {detection['confidence']:.2f}",
            (x1, max(32, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), image)


def write_data_yaml(output_dir: Path) -> None:
    (output_dir / "data.yaml").write_text(
        "\n".join(
            [
                f"path: {output_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                f"  0: {CLASS_NAME}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_dataset(args: argparse.Namespace) -> None:
    input_dir = resolve_backend_path(args.input_dir)
    output_dir = resolve_backend_path(args.output)
    model_path = resolve_backend_path(args.model)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError("Install ultralytics before building the real YOLO dataset.") from error

    image_paths = list_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {input_dir}")

    rng = random.Random(args.seed)
    shuffled_paths = image_paths[:]
    rng.shuffle(shuffled_paths)
    val_count = int(round(len(shuffled_paths) * args.val_ratio))
    if args.val_ratio > 0 and len(shuffled_paths) > 1:
        val_count = max(1, min(len(shuffled_paths) - 1, val_count))
    val_paths = set(shuffled_paths[:val_count])

    reset_output_dir(output_dir)
    model = YOLO(str(model_path))
    manifest = []

    for index, image_path in enumerate(image_paths, start=1):
        split = "val" if image_path in val_paths else "train"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] Could not read: {image_path}")
            continue

        image_height, image_width = image.shape[:2]
        detections, raw_count = detect_slots(
            model=model,
            image_path=image_path,
            conf=args.conf,
            imgsz=args.imgsz,
            max_boxes=args.max_boxes,
        )

        if len(detections) != args.max_boxes and not args.keep_incomplete:
            print(f"[WARN] Skipping {image_path.name}: kept {len(detections)} boxes, raw {raw_count}")
            continue

        destination_image = output_dir / "images" / split / image_path.name
        destination_label = output_dir / "labels" / split / f"{image_path.stem}.txt"
        destination_overlay = output_dir / "overlays" / split / f"{image_path.stem}_overlay.jpg"

        shutil.copy2(image_path, destination_image)
        destination_label.write_text(
            "\n".join(
                yolo_label_line(detection["box"], image_width, image_height)
                for detection in detections
            )
            + "\n",
            encoding="utf-8",
        )
        draw_overlay(image_path, detections, destination_overlay)

        manifest.append(
            {
                "source": str(image_path),
                "split": split,
                "image": str(destination_image),
                "label": str(destination_label),
                "overlay": str(destination_overlay),
                "rawDetectionCount": raw_count,
                "detectionCount": len(detections),
            }
        )

        if index % 25 == 0 or index == len(image_paths):
            print(f"Processed {index}/{len(image_paths)}")

    write_data_yaml(output_dir)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print()
    print(f"Source images: {len(image_paths)}")
    print(f"YOLO real images written: {len(manifest)}")
    print(f"Output: {output_dir}")
    print(f"Data YAML: {output_dir / 'data.yaml'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build real-image YOLO slot dataset.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-boxes", type=int, default=6)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    build_dataset(parse_args())


if __name__ == "__main__":
    main()
