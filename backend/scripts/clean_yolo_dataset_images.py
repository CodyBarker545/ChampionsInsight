"""Re-save YOLO dataset images as clean JPEG files.

Use this when Ultralytics reports warnings like:
    corrupt JPEG restored and saved

The script reads images from a YOLO dataset folder, re-encodes them as normal
JPEG files, and copies matching label files with the same stem.

Run from backend:
    python scripts/clean_yolo_dataset_images.py ^
      --input data/training_dataset/yolo_slot_detector_real ^
      --output data/training_dataset/yolo_slot_detector_real_clean
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2


BACKEND_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = BACKEND_DIR / "data" / "training_dataset" / "yolo_slot_detector_real"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "data" / "training_dataset" / "yolo_slot_detector_real_clean"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def resolve_backend_path(path: Path) -> Path:
    return path if path.is_absolute() else BACKEND_DIR / path


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def clean_split(input_dir: Path, output_dir: Path, split: str, jpeg_quality: int) -> list[dict]:
    input_image_dir = input_dir / "images" / split
    input_label_dir = input_dir / "labels" / split
    output_image_dir = output_dir / "images" / split
    output_label_dir = output_dir / "labels" / split

    records = []
    image_paths = sorted(
        path
        for path in input_image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            records.append(
                {
                    "source": str(image_path),
                    "status": "skipped_unreadable",
                }
            )
            print(f"[WARN] Could not read {image_path}")
            continue

        output_image_path = output_image_dir / f"{image_path.stem}.jpg"
        output_label_path = output_label_dir / f"{image_path.stem}.txt"
        input_label_path = input_label_dir / f"{image_path.stem}.txt"

        ok = cv2.imwrite(
            str(output_image_path),
            image,
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
        if not ok:
            records.append(
                {
                    "source": str(image_path),
                    "status": "failed_write",
                }
            )
            print(f"[WARN] Could not write {output_image_path}")
            continue

        if input_label_path.exists():
            shutil.copy2(input_label_path, output_label_path)
        else:
            output_label_path.write_text("", encoding="utf-8")
            print(f"[WARN] Missing label for {image_path.name}")

        records.append(
            {
                "source": str(image_path),
                "cleanImage": str(output_image_path),
                "cleanLabel": str(output_label_path),
                "status": "cleaned",
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
                "  0: opponent_slot",
                "",
            ]
        ),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean YOLO dataset image files.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--jpeg-quality", type=int, default=95)
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
        records.extend(clean_split(input_dir, output_dir, split, args.jpeg_quality))

    write_data_yaml(output_dir)
    (output_dir / "clean_manifest.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )

    cleaned_count = sum(1 for record in records if record["status"] == "cleaned")
    skipped_count = len(records) - cleaned_count

    print(f"Cleaned images: {cleaned_count}")
    print(f"Skipped/failed:  {skipped_count}")
    print(f"Output:          {output_dir}")
    print(f"Data YAML:       {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
