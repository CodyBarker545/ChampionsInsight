"""Run a trained YOLO slot detector and save debug overlays."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    BACKEND_DIR
    / "data"
    / "cv"
    / "models"
    / "slot_detector"
    / "yolov8n_slots"
    / "weights"
    / "best.pt"
)
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "data" / "cv" / "debug" / "yolo_slot_detector"


def resolve_backend_path(path: Path) -> Path:
    return path if path.is_absolute() else BACKEND_DIR / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug YOLO red slot detection.")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", type=Path, help="One image to inspect.")
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Folder to scan recursively for images matching --pattern.",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Trained YOLO .pt file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", default="original.jpg", help="Pattern used with --input-dir.")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument(
        "--max-boxes",
        type=int,
        default=None,
        help="Keep at most this many vertically distinct slot boxes.",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save one cropped image for each kept slot detection.",
    )
    return parser.parse_args()


def list_input_images(args: argparse.Namespace) -> list[Path]:
    if args.image:
        image_path = resolve_backend_path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        return [image_path]

    input_dir = resolve_backend_path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    image_paths = sorted(path for path in input_dir.rglob(args.pattern) if path.is_file())
    if not image_paths:
        raise FileNotFoundError(f"No images matched {args.pattern} under {input_dir}")

    return image_paths


def box_center_y(detection: dict) -> float:
    box = detection["box"]
    return box["y"] + box["height"] / 2


def filter_vertical_slot_boxes(detections: list[dict], max_boxes: int | None) -> list[dict]:
    if max_boxes is None or len(detections) <= max_boxes:
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


def detect_image(
    model,
    image_path: Path,
    output_dir: Path,
    conf: float,
    imgsz: int,
    max_boxes: int | None,
    save_crops: bool,
) -> dict:
    results = model.predict(
        source=str(image_path),
        conf=conf,
        imgsz=imgsz,
        verbose=False,
    )

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    clean_image = image.copy()

    detections = []
    for result in results:
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            x1, y1, x2, y2 = [int(round(value)) for value in xyxy]

            detections.append(
                {
                    "classId": class_id,
                    "className": result.names.get(class_id, str(class_id)),
                    "confidence": round(confidence, 4),
                    "box": {
                        "x": x1,
                        "y": y1,
                        "width": max(0, x2 - x1),
                        "height": max(0, y2 - y1),
                    },
                }
            )

    raw_detection_count = len(detections)
    detections = filter_vertical_slot_boxes(detections, max_boxes)

    safe_parent = image_path.parent.name
    output_stem = f"{safe_parent}__{image_path.stem}"
    overlay_path = output_dir / f"{output_stem}_yolo_slots.jpg"
    json_path = output_dir / f"{output_stem}_yolo_slots.json"
    crop_paths = []

    if save_crops:
        crop_dir = output_dir / "crops" / output_stem
        crop_dir.mkdir(parents=True, exist_ok=True)

    for index, detection in enumerate(detections, start=1):
        box = detection["box"]
        x1 = box["x"]
        y1 = box["y"]
        x2 = x1 + box["width"]
        y2 = y1 + box["height"]
        confidence = detection["confidence"]

        if save_crops:
            crop = clean_image[y1:y2, x1:x2]
            crop_path = crop_dir / f"opponent-slot-{index}.jpg"
            cv2.imwrite(str(crop_path), crop)
            crop_paths.append(str(crop_path))

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(
            image,
            f"slot {confidence:.2f}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(overlay_path), image)
    json_path.write_text(
        json.dumps(
            {
                "image": str(image_path),
                "rawDetectionCount": raw_detection_count,
                "detectionCount": len(detections),
                "detections": detections,
                "cropPaths": crop_paths,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "image": str(image_path),
        "rawDetectionCount": raw_detection_count,
        "detectionCount": len(detections),
        "overlayPath": str(overlay_path),
        "jsonPath": str(json_path),
        "cropPaths": crop_paths,
        "detections": detections,
    }


def write_summary(output_dir: Path, rows: list[dict]) -> None:
    summary_json = output_dir / "yolo_slot_detection_summary.json"
    summary_csv = output_dir / "yolo_slot_detection_summary.csv"

    summary_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    with summary_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "rawDetectionCount", "detectionCount", "overlayPath", "jsonPath"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image": row["image"],
                    "rawDetectionCount": row["rawDetectionCount"],
                    "detectionCount": row["detectionCount"],
                    "overlayPath": row["overlayPath"],
                    "jsonPath": row["jsonPath"],
                }
            )

    print(f"Summary JSON: {summary_json}")
    print(f"Summary CSV: {summary_csv}")


def main() -> None:
    args = parse_args()
    model_path = resolve_backend_path(args.model)
    output_dir = resolve_backend_path(args.output_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError(
            "ultralytics is required for YOLO detection. "
            "Install backend requirements first: pip install -r requirements.txt"
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(model_path))
    image_paths = list_input_images(args)

    rows = []
    for index, image_path in enumerate(image_paths, start=1):
        row = detect_image(
            model,
            image_path,
            output_dir,
            args.conf,
            args.imgsz,
            args.max_boxes,
            args.save_crops,
        )
        rows.append(row)
        print(
            f"[{index}/{len(image_paths)}] "
            f"{row['detectionCount']} boxes "
            f"(raw {row['rawDetectionCount']}) - {image_path}"
        )

    write_summary(output_dir, rows)

    if len(rows) == 1:
        print(json.dumps(rows[0]["detections"], indent=2))


if __name__ == "__main__":
    main()

