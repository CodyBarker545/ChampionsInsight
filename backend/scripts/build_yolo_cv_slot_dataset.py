"""Build YOLO slot labels using the existing OpenCV red-card detector.

This is intended to replace rough YOLO self-labels. It uses the app's current
red-card detector to find six opponent slot boxes in full team-preview images,
then writes a standard YOLO train/val dataset and review overlays.

Run from backend:
    python scripts/build_yolo_cv_slot_dataset.py
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services import cv_service


DEFAULT_INPUT_DIR = BACKEND_DIR / "data" / "training_dataset" / "yolo_slot_detector_real_originals"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "data" / "training_dataset" / "yolo_slot_detector_cv_labels"

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
        (output_dir / "crops" / split).mkdir(parents=True, exist_ok=True)

    (output_dir / "needs_review").mkdir(parents=True, exist_ok=True)


def box_to_yolo_line(box: dict, image_width: int, image_height: int) -> str:
    x_center = (box["x"] + box["width"] / 2) / image_width
    y_center = (box["y"] + box["height"] / 2) / image_height
    width = box["width"] / image_width
    height = box["height"] / image_height
    return f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def draw_overlay(image, boxes: list[dict], output_path: Path) -> None:
    overlay = image.copy()

    for index, box in enumerate(boxes, start=1):
        x1 = int(box["x"])
        y1 = int(box["y"])
        x2 = x1 + int(box["width"])
        y2 = y1 + int(box["height"])
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


def save_crops(image, boxes: list[dict], crop_dir: Path) -> list[str]:
    crop_paths = []
    crop_dir.mkdir(parents=True, exist_ok=True)

    for index, box in enumerate(boxes, start=1):
        x = int(box["x"])
        y = int(box["y"])
        width = int(box["width"])
        height = int(box["height"])
        crop = image[y : y + height, x : x + width]
        crop_path = crop_dir / f"opponent-slot-{index}.jpg"
        cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        crop_paths.append(str(crop_path))

    return crop_paths


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

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

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

    manifest = []
    skipped = []

    for index, image_path in enumerate(image_paths, start=1):
        try:
            image = cv_service.read_cv_image(image_path)
        except Exception as error:
            skipped.append(
                {
                    "source": str(image_path),
                    "reason": f"unreadable: {error}",
                }
            )
            continue

        image_height, image_width = image.shape[:2]
        boxes = cv_service.detect_opponent_card_boxes(image)

        if len(boxes) != args.expected_count and not args.keep_incomplete:
            review_overlay = output_dir / "needs_review" / f"{image_path.stem}_overlay.jpg"
            draw_overlay(image, boxes, review_overlay)
            skipped.append(
                {
                    "source": str(image_path),
                    "reason": f"detected_{len(boxes)}_boxes",
                    "overlay": str(review_overlay),
                }
            )
            continue

        split = "val" if image_path in val_paths else "train"
        output_image = output_dir / "images" / split / f"{image_path.stem}.jpg"
        output_label = output_dir / "labels" / split / f"{image_path.stem}.txt"
        output_overlay = output_dir / "overlays" / split / f"{image_path.stem}_overlay.jpg"
        output_crop_dir = output_dir / "crops" / split / image_path.stem

        cv2.imwrite(str(output_image), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        output_label.write_text(
            "\n".join(box_to_yolo_line(box, image_width, image_height) for box in boxes)
            + "\n",
            encoding="utf-8",
        )
        draw_overlay(image, boxes, output_overlay)
        crop_paths = save_crops(image, boxes, output_crop_dir)

        manifest.append(
            {
                "source": str(image_path),
                "image": str(output_image),
                "label": str(output_label),
                "overlay": str(output_overlay),
                "crops": crop_paths,
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
    print(f"Input images:   {len(image_paths)}")
    print(f"Dataset images: {len(manifest)}")
    print(f"Needs review:   {len(skipped)}")
    print(f"Output:         {output_dir}")
    print(f"Data YAML:      {output_dir / 'data.yaml'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO dataset from CV red slot detections.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-count", type=int, default=6)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--keep-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    build_dataset(parse_args())


if __name__ == "__main__":
    main()
