"""Refine rough YOLO slot labels to tighter red-card boxes.

This script is for the situation where auto-labels are close but too loose.
For each existing YOLO box, it searches inside that region for the red/pink
slot card pixels and writes a tighter replacement label.

Run from backend:
    python scripts/refine_yolo_slot_labels.py ^
      --input data/training_dataset/yolo_slot_detector_real_clean ^
      --output data/training_dataset/yolo_slot_detector_real_refined
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = BACKEND_DIR / "data" / "training_dataset" / "yolo_slot_detector_real_clean"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "data" / "training_dataset" / "yolo_slot_detector_real_refined"

CLASS_ID = 0
CLASS_NAME = "opponent_slot"
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


def resolve_backend_path(path: Path) -> Path:
    return path if path.is_absolute() else BACKEND_DIR / path


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "overlays" / split).mkdir(parents=True, exist_ok=True)


def read_yolo_labels(label_path: Path) -> list[dict]:
    labels = []
    if not label_path.exists():
        return labels

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        labels.append(
            {
                "classId": int(float(parts[0])),
                "xCenter": float(parts[1]),
                "yCenter": float(parts[2]),
                "width": float(parts[3]),
                "height": float(parts[4]),
            }
        )

    return labels


def yolo_to_pixel_box(label: dict, image_width: int, image_height: int) -> dict:
    width = label["width"] * image_width
    height = label["height"] * image_height
    x = (label["xCenter"] * image_width) - width / 2
    y = (label["yCenter"] * image_height) - height / 2
    return clamp_box(
        {
            "x": int(round(x)),
            "y": int(round(y)),
            "width": int(round(width)),
            "height": int(round(height)),
        },
        image_width,
        image_height,
    )


def pixel_box_to_yolo_line(box: dict, image_width: int, image_height: int) -> str:
    x_center = (box["x"] + box["width"] / 2) / image_width
    y_center = (box["y"] + box["height"] / 2) / image_height
    width = box["width"] / image_width
    height = box["height"] / image_height
    return f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def clamp_box(box: dict, image_width: int, image_height: int) -> dict:
    x = max(0, min(int(round(box["x"])), image_width - 1))
    y = max(0, min(int(round(box["y"])), image_height - 1))
    width = max(1, min(int(round(box["width"])), image_width - x))
    height = max(1, min(int(round(box["height"])), image_height - y))
    return {"x": x, "y": y, "width": width, "height": height}


def expand_box(box: dict, image_width: int, image_height: int, pad_ratio: float) -> dict:
    x_pad = int(box["width"] * pad_ratio)
    y_pad = int(box["height"] * pad_ratio)
    return clamp_box(
        {
            "x": box["x"] - x_pad,
            "y": box["y"] - y_pad,
            "width": box["width"] + x_pad * 2,
            "height": box["height"] + y_pad * 2,
        },
        image_width,
        image_height,
    )


def build_red_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 45, 45], dtype=np.uint8)
    upper_red = np.array([14, 255, 255], dtype=np.uint8)
    lower_pink = np.array([145, 30, 45], dtype=np.uint8)
    upper_pink = np.array([179, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_pink, upper_pink))

    height, width = image.shape[:2]
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(11, width // 24), max(9, height // 18)),
    )
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(3, width // 160), max(3, height // 160)),
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    return mask


def refine_box(image: np.ndarray, rough_box: dict, pad_ratio: float) -> tuple[dict, str]:
    image_height, image_width = image.shape[:2]
    search_box = expand_box(rough_box, image_width, image_height, pad_ratio)
    crop = image[
        search_box["y"] : search_box["y"] + search_box["height"],
        search_box["x"] : search_box["x"] + search_box["width"],
    ]

    if crop.size == 0:
        return rough_box, "empty_crop"

    mask = build_red_mask(crop)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    crop_area = max(1, search_box["width"] * search_box["height"])
    rough_center_x = (rough_box["x"] + rough_box["width"] / 2) - search_box["x"]
    rough_center_y = (rough_box["y"] + rough_box["height"] / 2) - search_box["y"]

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height
        area_ratio = area / crop_area
        aspect = width / max(1, height)

        if area_ratio < 0.12:
            continue

        if not (1.2 <= aspect <= 8.5 or 0.10 <= aspect <= 0.95):
            continue

        center_x = x + width / 2
        center_y = y + height / 2
        center_distance = (
            abs(center_x - rough_center_x) / max(1, search_box["width"])
            + abs(center_y - rough_center_y) / max(1, search_box["height"])
        )
        score = area_ratio - center_distance * 0.25
        candidates.append(
            {
                "box": {
                    "x": search_box["x"] + x,
                    "y": search_box["y"] + y,
                    "width": width,
                    "height": height,
                },
                "score": score,
            }
        )

    if not candidates:
        return rough_box, "kept_original_no_red_candidate"

    best = max(candidates, key=lambda item: item["score"])["box"]
    refined = expand_box(
        best,
        image_width,
        image_height,
        pad_ratio=0.018,
    )

    return refined, "refined"


def draw_overlay(image: np.ndarray, rough_boxes: list[dict], refined_boxes: list[dict], output_path: Path) -> None:
    overlay = image.copy()

    for index, box in enumerate(rough_boxes, start=1):
        x1 = box["x"]
        y1 = box["y"]
        x2 = x1 + box["width"]
        y2 = y1 + box["height"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(
            overlay,
            f"old {index}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    for index, box in enumerate(refined_boxes, start=1):
        x1 = box["x"]
        y1 = box["y"]
        x2 = x1 + box["width"]
        y2 = y1 + box["height"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(
            overlay,
            f"new {index}",
            (x1, min(image.shape[0] - 10, y1 + y2 - y1 + 28)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), overlay)


def find_image_for_label(image_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        path = image_dir / f"{stem}{extension}"
        if path.exists():
            return path
    return None


def refine_split(input_dir: Path, output_dir: Path, split: str, pad_ratio: float) -> list[dict]:
    records = []
    input_label_dir = input_dir / "labels" / split
    input_image_dir = input_dir / "images" / split
    output_label_dir = output_dir / "labels" / split
    output_image_dir = output_dir / "images" / split
    output_overlay_dir = output_dir / "overlays" / split

    for label_path in sorted(input_label_dir.glob("*.txt")):
        image_path = find_image_for_label(input_image_dir, label_path.stem)
        if image_path is None:
            records.append({"label": str(label_path), "status": "missing_image"})
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            records.append({"image": str(image_path), "status": "unreadable_image"})
            continue

        image_height, image_width = image.shape[:2]
        rough_boxes = [
            yolo_to_pixel_box(label, image_width, image_height)
            for label in read_yolo_labels(label_path)
        ]
        refined_boxes = []
        statuses = []

        for rough_box in rough_boxes:
            refined_box, status = refine_box(image, rough_box, pad_ratio)
            refined_boxes.append(refined_box)
            statuses.append(status)

        output_image_path = output_image_dir / f"{label_path.stem}.jpg"
        output_label_path = output_label_dir / f"{label_path.stem}.txt"
        output_overlay_path = output_overlay_dir / f"{label_path.stem}_refined_overlay.jpg"

        cv2.imwrite(str(output_image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        output_label_path.write_text(
            "\n".join(
                pixel_box_to_yolo_line(box, image_width, image_height)
                for box in refined_boxes
            )
            + "\n",
            encoding="utf-8",
        )
        draw_overlay(image, rough_boxes, refined_boxes, output_overlay_path)

        records.append(
            {
                "sourceImage": str(image_path),
                "sourceLabel": str(label_path),
                "outputImage": str(output_image_path),
                "outputLabel": str(output_label_path),
                "overlay": str(output_overlay_path),
                "boxCount": len(refined_boxes),
                "statuses": statuses,
            }
        )

    return records


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine rough YOLO red slot labels.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--search-pad-ratio", type=float, default=0.04)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = resolve_backend_path(args.input)
    output_dir = resolve_backend_path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input YOLO dataset not found: {input_dir}")

    reset_output_dir(output_dir)

    records = []
    for split in ["train", "val"]:
        records.extend(refine_split(input_dir, output_dir, split, args.search_pad_ratio))

    write_data_yaml(output_dir)
    (output_dir / "refine_manifest.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )

    refined_count = sum(
        1
        for record in records
        for status in record.get("statuses", [])
        if status == "refined"
    )
    kept_count = sum(
        1
        for record in records
        for status in record.get("statuses", [])
        if status != "refined"
    )

    print(f"Images processed: {len(records)}")
    print(f"Boxes refined:    {refined_count}")
    print(f"Boxes kept:       {kept_count}")
    print(f"Output:           {output_dir}")
    print(f"Data YAML:        {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
