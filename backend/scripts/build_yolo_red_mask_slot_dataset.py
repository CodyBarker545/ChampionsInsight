"""Build YOLO slot labels from red-card masks instead of YOLO self-labels.

Use this to create tighter labels for full team-preview images before final
YOLO training. It detects the red/pink slot card shapes with OpenCV, writes
YOLO labels, and saves overlays for review.

Run from backend:
    python scripts/build_yolo_red_mask_slot_dataset.py
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = BACKEND_DIR / "data" / "training_dataset" / "yolo_slot_detector_real_originals"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "data" / "training_dataset" / "yolo_slot_detector_red_mask"

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

    (output_dir / "needs_review").mkdir(parents=True, exist_ok=True)


def build_red_slot_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Scarlet/red plus magenta/pink card tones from Switch team preview panels.
    lower_red_1 = np.array([0, 45, 45], dtype=np.uint8)
    upper_red_1 = np.array([14, 255, 255], dtype=np.uint8)
    lower_red_2 = np.array([150, 35, 45], dtype=np.uint8)
    upper_red_2 = np.array([179, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_red_2, upper_red_2))

    height, width = image.shape[:2]
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(21, int(width * 0.025)),
            max(21, int(height * 0.018)),
        ),
    )
    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            max(5, int(width * 0.004)),
            max(5, int(height * 0.004)),
        ),
    )

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

    return mask


def clamp_box(box: dict, image_width: int, image_height: int) -> dict:
    x = max(0, min(int(round(box["x"])), image_width - 1))
    y = max(0, min(int(round(box["y"])), image_height - 1))
    width = max(1, min(int(round(box["width"])), image_width - x))
    height = max(1, min(int(round(box["height"])), image_height - y))
    return {"x": x, "y": y, "width": width, "height": height}


def expand_box(box: dict, image_width: int, image_height: int, x_pad_ratio=0.015, y_pad_ratio=0.018) -> dict:
    x_pad = int(box["width"] * x_pad_ratio)
    y_pad = int(box["height"] * y_pad_ratio)
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


def box_center(box: dict) -> tuple[float, float]:
    return box["x"] + box["width"] / 2, box["y"] + box["height"] / 2


def boxes_overlap_ratio(first: dict, second: dict) -> float:
    fx2 = first["x"] + first["width"]
    fy2 = first["y"] + first["height"]
    sx2 = second["x"] + second["width"]
    sy2 = second["y"] + second["height"]

    overlap_w = max(0, min(fx2, sx2) - max(first["x"], second["x"]))
    overlap_h = max(0, min(fy2, sy2) - max(first["y"], second["y"]))
    overlap = overlap_w * overlap_h
    smaller = min(first["width"] * first["height"], second["width"] * second["height"])

    return overlap / max(1, smaller)


def dedupe_boxes(boxes: list[dict]) -> list[dict]:
    kept: list[dict] = []

    for box in sorted(boxes, key=lambda item: item["score"], reverse=True):
        if any(boxes_overlap_ratio(box, old) > 0.55 for old in kept):
            continue
        kept.append(box)

    return kept


def detect_red_slot_boxes(image: np.ndarray, expected_count: int) -> tuple[list[dict], np.ndarray]:
    height, width = image.shape[:2]
    image_area = width * height
    mask = build_red_slot_mask(image)
    contours, _hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[dict] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        area_ratio = area / max(1, image_area)
        aspect = w / max(1, h)

        if area_ratio < 0.010:
            continue
        if w < width * 0.08 or h < height * 0.055:
            continue

        looks_like_slot = (
            1.8 <= aspect <= 8.0
            or 0.12 <= aspect <= 0.85
        )
        if not looks_like_slot:
            continue

        box = expand_box(
            {"x": x, "y": y, "width": w, "height": h},
            width,
            height,
        )
        box["score"] = area_ratio
        box["aspect"] = round(aspect, 4)
        boxes.append(box)

    boxes = dedupe_boxes(boxes)

    if len(boxes) > expected_count:
        boxes = sorted(boxes, key=lambda item: item["score"], reverse=True)[:expected_count]

    if boxes:
        wide_count = sum(1 for box in boxes if box["width"] >= box["height"])
        if wide_count >= len(boxes) / 2:
            boxes = sorted(boxes, key=lambda item: item["y"])
        else:
            boxes = sorted(boxes, key=lambda item: item["x"])

    return boxes, mask


def yolo_label_line(box: dict, image_width: int, image_height: int) -> str:
    x_center = (box["x"] + box["width"] / 2) / image_width
    y_center = (box["y"] + box["height"] / 2) / image_height
    width = box["width"] / image_width
    height = box["height"] / image_height

    return f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def draw_overlay(image: np.ndarray, boxes: list[dict], output_path: Path) -> None:
    overlay = image.copy()

    for index, box in enumerate(boxes, start=1):
        x1 = box["x"]
        y1 = box["y"]
        x2 = x1 + box["width"]
        y2 = y1 + box["height"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 4)
        cv2.putText(
            overlay,
            str(index),
            (x1, max(32, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), overlay)


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
    parser = argparse.ArgumentParser(description="Build YOLO labels from red slot masks.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-count", type=int, default=6)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = resolve_backend_path(args.input_dir)
    output_dir = resolve_backend_path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    image_paths = list_images(input_dir)
    if not image_paths:
        raise FileNotFoundError(f"No training images found in: {input_dir}")

    reset_output_dir(output_dir)

    rng = random.Random(args.seed)
    shuffled_paths = image_paths[:]
    rng.shuffle(shuffled_paths)
    val_count = int(round(len(shuffled_paths) * args.val_ratio))
    if args.val_ratio > 0 and len(shuffled_paths) > 1:
        val_count = max(1, min(len(shuffled_paths) - 1, val_count))
    val_paths = set(shuffled_paths[:val_count])

    manifest = []
    skipped = []

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            skipped.append({"source": str(image_path), "reason": "unreadable"})
            continue

        boxes, mask = detect_red_slot_boxes(image, args.expected_count)
        split = "val" if image_path in val_paths else "train"

        if len(boxes) != args.expected_count and not args.keep_incomplete:
            review_path = output_dir / "needs_review" / f"{image_path.stem}_overlay.jpg"
            draw_overlay(image, boxes, review_path)
            cv2.imwrite(str(output_dir / "needs_review" / f"{image_path.stem}_mask.jpg"), mask)
            skipped.append(
                {
                    "source": str(image_path),
                    "reason": f"detected_{len(boxes)}_boxes",
                    "overlay": str(review_path),
                }
            )
            continue

        output_image = output_dir / "images" / split / f"{image_path.stem}.jpg"
        output_label = output_dir / "labels" / split / f"{image_path.stem}.txt"
        output_overlay = output_dir / "overlays" / split / f"{image_path.stem}_overlay.jpg"

        cv2.imwrite(str(output_image), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        output_label.write_text(
            "\n".join(
                yolo_label_line(box, image.shape[1], image.shape[0])
                for box in boxes
            )
            + "\n",
            encoding="utf-8",
        )
        draw_overlay(image, boxes, output_overlay)

        manifest.append(
            {
                "source": str(image_path),
                "image": str(output_image),
                "label": str(output_label),
                "overlay": str(output_overlay),
                "split": split,
                "boxCount": len(boxes),
            }
        )

        if index % 25 == 0 or index == len(image_paths):
            print(f"Processed {index}/{len(image_paths)}")

    write_data_yaml(output_dir)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "skipped.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")

    print()
    print(f"Input images:    {len(image_paths)}")
    print(f"Dataset images:  {len(manifest)}")
    print(f"Needs review:    {len(skipped)}")
    print(f"Output:          {output_dir}")
    print(f"Data YAML:       {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
