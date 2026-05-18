"""Combine multiple YOLO slot-detection datasets into one train/val dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "data" / "training_dataset" / "yolo_slot_detector_mixed"


def resolve_backend_path(path: Path) -> Path:
    return path if path.is_absolute() else BACKEND_DIR / path


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def copy_split(source_dir: Path, output_dir: Path, dataset_tag: str, split: str) -> int:
    count = 0
    image_dir = source_dir / "images" / split
    label_dir = source_dir / "labels" / split

    for image_path in sorted(image_dir.glob("*.jpg")):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue

        destination_stem = f"{dataset_tag}__{image_path.stem}"
        shutil.copy2(image_path, output_dir / "images" / split / f"{destination_stem}.jpg")
        shutil.copy2(label_path, output_dir / "labels" / split / f"{destination_stem}.txt")
        count += 1

    return count


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


def parse_dataset_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("datasets must be passed as tag=path")
    tag, raw_path = value.split("=", 1)
    tag = tag.strip()
    if not tag:
        raise argparse.ArgumentTypeError("dataset tag cannot be empty")
    return tag, Path(raw_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine YOLO slot datasets.")
    parser.add_argument(
        "--dataset",
        action="append",
        type=parse_dataset_arg,
        required=True,
        help="Dataset in tag=path form. Repeat for each source dataset.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_backend_path(args.output)
    reset_output_dir(output_dir)

    summary: dict[str, dict[str, int]] = {}
    for tag, raw_path in args.dataset:
        source_dir = resolve_backend_path(raw_path)
        if not source_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {source_dir}")

        summary[tag] = {
            "train": copy_split(source_dir, output_dir, tag, "train"),
            "val": copy_split(source_dir, output_dir, tag, "val"),
        }

    write_data_yaml(output_dir)
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Output: {output_dir}")
    print(f"Data YAML: {output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()

