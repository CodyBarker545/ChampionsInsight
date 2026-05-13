"""Build a YOLO dataset for objects inside opponent red slot crops.

This is the Stage 2 detector dataset:
    red slot crop -> pokemon_sprite and type_cluster boxes

It uses the existing CV object-layer code as an auto-label draft generator,
then writes review overlays so bad or loose labels can be fixed before final
training.

Run from backend:
    python scripts/build_yolo_slot_object_dataset.py
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


DEFAULT_INPUT_DIRS = [
    BACKEND_DIR
    / "data"
    / "training_dataset"
    / "yolo_slot_detector_cv_labels"
    / "crops",
    BACKEND_DIR
    / "data"
    / "cv"
    / "debug"
    / "yolo_slot_detector_cv_labels_test"
    / "crops",
]
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "data" / "training_dataset" / "yolo_slot_object_detector"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_NAMES = {
    0: "pokemon_sprite",
    1: "type_cluster",
}


def resolve_backend_path(path: Path) -> Path:
    return path if path.is_absolute() else BACKEND_DIR / path


def list_slot_images(input_dirs: list[Path]) -> list[Path]:
    image_paths: list[Path] = []
    seen: set[Path] = set()

    for input_dir in input_dirs:
        input_dir = resolve_backend_path(input_dir)
        if not input_dir.exists():
            continue

        for path in sorted(input_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            # Keep this focused on red-card crops. The recursive debug folders
            # can contain overlays and other images that should not train Stage 2.
            if not path.name.lower().startswith("opponent-slot-"):
                continue

            resolved = path.resolve()
            if resolved in seen:
                continue

            seen.add(resolved)
            image_paths.append(path)

    return image_paths


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "overlays" / split).mkdir(parents=True, exist_ok=True)

    (output_dir / "needs_review").mkdir(parents=True, exist_ok=True)


def safe_stem(path: Path) -> str:
    parent = path.parent.name
    grandparent = path.parent.parent.name
    return f"{grandparent}__{parent}__{path.stem}".replace(" ", "_")


def clamp_object_box(box: dict, image_width: int, image_height: int) -> dict:
    return cv_service.clamp_box_to_image(
        {
            "x": int(round(box.get("x", 0))),
            "y": int(round(box.get("y", 0))),
            "width": int(round(box.get("width", 0))),
            "height": int(round(box.get("height", 0))),
        },
        image_width,
        image_height,
    )


def get_object_box(detected_object: dict) -> dict:
    box = detected_object.get("box")
    if box:
        return box

    return {
        "x": detected_object.get("x", 0),
        "y": detected_object.get("y", 0),
        "width": detected_object.get("width", 0),
        "height": detected_object.get("height", 0),
    }


def box_to_yolo_line(class_id: int, box: dict, image_width: int, image_height: int) -> str:
    x_center = (box["x"] + box["width"] / 2) / image_width
    y_center = (box["y"] + box["height"] / 2) / image_height
    width = box["width"] / image_width
    height = box["height"] / image_height
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def expand_labeled_box(
    box: dict,
    image_width: int,
    image_height: int,
    x_ratio: float,
    y_ratio: float,
) -> dict:
    x_padding = max(2, int(round(box["width"] * x_ratio)))
    y_padding = max(2, int(round(box["height"] * y_ratio)))
    return cv_service.expand_box_xy(box, x_padding, y_padding, image_width, image_height)


def object_quality_notes(label: str, box: dict, image_width: int, image_height: int) -> list[str]:
    notes: list[str] = []
    width_ratio = box["width"] / max(1, image_width)
    height_ratio = box["height"] / max(1, image_height)
    area_ratio = (box["width"] * box["height"]) / max(1, image_width * image_height)

    if label == "pokemon_sprite":
        if width_ratio < 0.08 or height_ratio < 0.18:
            notes.append("pokemon_box_too_small")
        if width_ratio > 0.74 or height_ratio > 1.00:
            notes.append("pokemon_box_too_large")
        if area_ratio < 0.025:
            notes.append("pokemon_area_tiny")

    if label == "type_cluster":
        aspect = box["width"] / max(1, box["height"])
        if width_ratio < 0.08 or height_ratio < 0.16:
            notes.append("type_cluster_too_small")
        if width_ratio > 0.62 or height_ratio > 0.72:
            notes.append("type_cluster_too_large")
        if aspect < 0.50 or aspect > 3.25:
            notes.append("type_cluster_bad_aspect")

    return notes


def collect_slot_objects(
    slot_image,
    pokemon_padding_x: float,
    pokemon_padding_y: float,
    type_padding_x: float,
    type_padding_y: float,
) -> tuple[list[dict], list[str]]:
    image_height, image_width = slot_image.shape[:2]
    object_layer = cv_service.detect_slot_object_layer(slot_image)
    objects: list[dict] = []
    notes: list[str] = []

    pokemon_object = object_layer.get("pokemon_sprite")
    if pokemon_object and not cv_service.is_empty_image(pokemon_object.get("image")):
        box = clamp_object_box(get_object_box(pokemon_object), image_width, image_height)
        box = expand_labeled_box(
            box,
            image_width,
            image_height,
            pokemon_padding_x,
            pokemon_padding_y,
        )
        object_notes = object_quality_notes("pokemon_sprite", box, image_width, image_height)
        objects.append(
            {
                "classId": 0,
                "className": "pokemon_sprite",
                "box": box,
                "source": pokemon_object.get("source", ""),
                "confidence": pokemon_object.get("confidence"),
                "cropQuality": pokemon_object.get("cropQuality"),
                "notes": object_notes,
            }
        )
        notes.extend(object_notes)
    else:
        notes.append("missing_pokemon_sprite")

    type_cluster_box = cv_service.extract_detected_type_icon_cluster_box(slot_image)
    if type_cluster_box:
        box = clamp_object_box(type_cluster_box, image_width, image_height)
        box = expand_labeled_box(
            box,
            image_width,
            image_height,
            type_padding_x,
            type_padding_y,
        )
        object_notes = object_quality_notes("type_cluster", box, image_width, image_height)
        objects.append(
            {
                "classId": 1,
                "className": "type_cluster",
                "box": box,
                "source": "type_icon_cluster",
                "confidence": None,
                "cropQuality": None,
                "notes": object_notes,
            }
        )
        notes.extend(object_notes)
    else:
        notes.append("missing_type_cluster")

    return objects, sorted(set(notes))


def draw_overlay(image, objects: list[dict], notes: list[str], output_path: Path) -> None:
    overlay = image.copy()
    colors = {
        "pokemon_sprite": (0, 220, 255),
        "type_cluster": (255, 0, 255),
    }

    for item in objects:
        box = item["box"]
        label = item["className"]
        color = colors[label]
        x1 = int(box["x"])
        y1 = int(box["y"])
        x2 = x1 + int(box["width"])
        y2 = y1 + int(box["height"])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            overlay,
            label,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    if notes:
        cv2.putText(
            overlay,
            "REVIEW: " + ",".join(notes[:3]),
            (8, max(18, image.shape[0] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(str(output_path), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def write_data_yaml(output_dir: Path) -> None:
    lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for class_id, name in CLASS_NAMES.items():
        lines.append(f"  {class_id}: {name}")
    lines.append("")
    (output_dir / "data.yaml").write_text("\n".join(lines), encoding="utf-8")


def copy_review_image(
    image,
    output_dir: Path,
    stem: str,
    objects: list[dict],
    notes: list[str],
    source_path: Path,
) -> dict:
    review_image_path = output_dir / "needs_review" / f"{stem}.jpg"
    review_overlay_path = output_dir / "needs_review" / f"{stem}_overlay.jpg"
    cv2.imwrite(str(review_image_path), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    draw_overlay(image, objects, notes, review_overlay_path)
    return {
        "source": str(source_path),
        "image": str(review_image_path),
        "overlay": str(review_overlay_path),
        "objects": objects,
        "notes": notes,
    }


def build_dataset(args: argparse.Namespace) -> None:
    input_dirs = [resolve_backend_path(path) for path in args.input_dir]
    output_dir = resolve_backend_path(args.output)

    image_paths = list_slot_images(input_dirs)
    if not image_paths:
        searched = "\n".join(str(path) for path in input_dirs)
        raise FileNotFoundError(f"No opponent-slot images found in:\n{searched}")

    rng = random.Random(args.seed)
    shuffled_paths = image_paths[:]
    rng.shuffle(shuffled_paths)
    val_count = int(round(len(shuffled_paths) * args.val_ratio))
    if args.val_ratio > 0 and len(shuffled_paths) > 1:
        val_count = max(1, min(len(shuffled_paths) - 1, val_count))
    val_paths = set(shuffled_paths[:val_count])

    reset_output_dir(output_dir)

    manifest: list[dict] = []
    review_items: list[dict] = []

    for index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or cv_service.is_empty_image(image):
            review_items.append(
                {
                    "source": str(image_path),
                    "notes": ["unreadable_or_empty_image"],
                }
            )
            continue

        image_height, image_width = image.shape[:2]
        stem = safe_stem(image_path)
        objects, notes = collect_slot_objects(
            image,
            args.pokemon_padding_x,
            args.pokemon_padding_y,
            args.type_padding_x,
            args.type_padding_y,
        )

        should_review = bool(notes)
        if args.include_review:
            should_review = False

        if should_review:
            review_items.append(
                copy_review_image(image, output_dir, stem, objects, notes, image_path)
            )
            continue

        split = "val" if image_path in val_paths else "train"
        output_image = output_dir / "images" / split / f"{stem}.jpg"
        output_label = output_dir / "labels" / split / f"{stem}.txt"
        output_overlay = output_dir / "overlays" / split / f"{stem}_overlay.jpg"

        cv2.imwrite(str(output_image), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        output_label.write_text(
            "\n".join(
                box_to_yolo_line(item["classId"], item["box"], image_width, image_height)
                for item in objects
            )
            + "\n",
            encoding="utf-8",
        )
        draw_overlay(image, objects, notes, output_overlay)

        manifest.append(
            {
                "source": str(image_path),
                "image": str(output_image),
                "label": str(output_label),
                "overlay": str(output_overlay),
                "split": split,
                "objects": objects,
                "notes": notes,
            }
        )

        if index % 100 == 0 or index == len(image_paths):
            print(f"Processed {index}/{len(image_paths)}")

    write_data_yaml(output_dir)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "needs_review.json").write_text(json.dumps(review_items, indent=2), encoding="utf-8")

    pokemon_count = sum(
        1
        for item in manifest
        for obj in item["objects"]
        if obj["className"] == "pokemon_sprite"
    )
    type_cluster_count = sum(
        1
        for item in manifest
        for obj in item["objects"]
        if obj["className"] == "type_cluster"
    )

    print()
    print(f"Input slot crops: {len(image_paths)}")
    print(f"Dataset images:   {len(manifest)}")
    print(f"Needs review:     {len(review_items)}")
    print(f"Pokemon boxes:    {pokemon_count}")
    print(f"Type clusters:    {type_cluster_count}")
    print(f"Output:           {output_dir}")
    print(f"Data YAML:        {output_dir / 'data.yaml'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build YOLO data for Pokemon/type objects inside red slot crops."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        default=None,
        help="Folder containing opponent-slot-*.jpg crops. Can be passed more than once.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pokemon-padding-x",
        type=float,
        default=0.08,
        help="Horizontal padding ratio applied to auto-labeled Pokemon boxes.",
    )
    parser.add_argument(
        "--pokemon-padding-y",
        type=float,
        default=0.12,
        help="Vertical padding ratio applied to auto-labeled Pokemon boxes.",
    )
    parser.add_argument(
        "--type-padding-x",
        type=float,
        default=0.04,
        help="Horizontal padding ratio applied to type-cluster boxes.",
    )
    parser.add_argument(
        "--type-padding-y",
        type=float,
        default=0.06,
        help="Vertical padding ratio applied to type-cluster boxes.",
    )
    parser.add_argument(
        "--include-review",
        action="store_true",
        help="Keep suspicious auto-labels in the dataset instead of needs_review.",
    )
    args = parser.parse_args()
    if args.input_dir is None:
        args.input_dir = DEFAULT_INPUT_DIRS
    return args


def main() -> None:
    build_dataset(parse_args())


if __name__ == "__main__":
    main()
