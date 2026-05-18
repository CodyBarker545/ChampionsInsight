r"""Incrementally prepare only new upload images for inner slot-object YOLO training.

Pipeline:
    data/uploads/
      -> detect six red opponent slots with the trained outer YOLO model
      -> save originals, overlays, labels, and per-slot crops for only new uploads
      -> rebuild the inner-object dataset using the historical slot crops plus
         the newly extracted upload slot crops

Run from the project root:
    .\.venv\Scripts\python.exe backend\scripts\prepare_new_uploads_for_slot_object_training.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2


BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from paths import UPLOAD_DIR
from scripts.cv_datasets import build_yolo_slot_object_dataset as slot_object_builder
from scripts.cv_datasets.build_yolo_real_slot_dataset import (
    detect_slots,
    draw_overlay,
    yolo_label_line,
)
from services import cv_service, slot_object_detection_service


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

DEFAULT_OUTER_MODEL = (
    BACKEND_DIR
    / "data"
    / "cv"
    / "models"
    / "slot_detector"
    / "yolov8n_slots_mixed_synth_real_cv"
    / "weights"
    / "best.pt"
)
DEFAULT_WORK_DIR = (
    BACKEND_DIR / "data" / "training_dataset" / "upload_slot_object_ingest"
)
DEFAULT_OBJECT_DATASET_OUTPUT = (
    BACKEND_DIR
    / "data"
    / "training_dataset"
    / "yolo_slot_object_detector_padded_clean_types"
)
PROCESSED_STATE_NAME = "processed_uploads.json"
REVIEW_DIR = BACKEND_DIR / "data" / "training_dataset" / "review"
UNSORTED_SLOT_DIR = REVIEW_DIR / "unsorted_slots"
UNSORTED_POKEMON_DIR = REVIEW_DIR / "unsorted_pokemon"
UNSORTED_SINGLE_TYPE_DIR = REVIEW_DIR / "unsorted_single_type_icons"


def resolve_backend_path(path: Path) -> Path:
    return path if path.is_absolute() else BACKEND_DIR / path


def list_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def file_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"processed": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def safe_image_stem(path: Path, digest: str) -> str:
    return f"{path.stem}__{digest[:10]}"


def ensure_work_dirs(work_dir: Path) -> dict[str, Path]:
    dirs = {
        "originals": work_dir / "originals",
        "labels": work_dir / "labels",
        "overlays": work_dir / "overlays",
        "slot_crops": work_dir / "slot_crops",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    for directory in (
        UNSORTED_SLOT_DIR,
        UNSORTED_POKEMON_DIR,
        UNSORTED_SINGLE_TYPE_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def write_slot_crops(
    image,
    detections: list[dict],
    crop_root: Path,
    stem: str,
) -> list[str]:
    crop_dir = crop_root / stem
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_paths: list[str] = []

    for index, detection in enumerate(detections, start=1):
        box = detection["box"]
        x1 = box["x"]
        y1 = box["y"]
        x2 = x1 + box["width"]
        y2 = y1 + box["height"]
        crop = image[y1:y2, x1:x2]
        crop_path = crop_dir / f"opponent-slot-{index}.jpg"
        cv2.imwrite(str(crop_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        crop_paths.append(str(crop_path))

    return crop_paths


def write_review_crop(image, destination_dir: Path, stem: str, suffix: str) -> str:
    destination = destination_dir / f"{stem}__{suffix}.jpg"
    counter = 2
    while destination.exists():
        destination = destination_dir / f"{stem}__{suffix}__{counter}.jpg"
        counter += 1
    cv2.imwrite(str(destination), image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return str(destination)


def harvest_review_crops(slot_crop_paths: list[str], upload_stem: str) -> dict[str, list[str]]:
    harvested = {
        "slots": [],
        "pokemon": [],
        "singleTypeIcons": [],
    }

    for slot_index, slot_crop_path in enumerate(slot_crop_paths, start=1):
        slot_path = Path(slot_crop_path)
        slot_image = cv2.imread(str(slot_path), cv2.IMREAD_COLOR)
        if slot_image is None or cv_service.is_empty_image(slot_image):
            continue

        slot_stem = f"{upload_stem}__slot-{slot_index}"
        harvested["slots"].append(
            write_review_crop(slot_image, UNSORTED_SLOT_DIR, slot_stem, "slot")
        )

        object_layer = slot_object_detection_service.detect_slot_objects(slot_image)
        pokemon_object = object_layer.get("pokemon_sprite")
        pokemon_image = pokemon_object.get("image") if pokemon_object else None
        if not cv_service.is_empty_image(pokemon_image):
            harvested["pokemon"].append(
                write_review_crop(
                    pokemon_image,
                    UNSORTED_POKEMON_DIR,
                    slot_stem,
                    "pokemon",
                )
            )

        for type_index, type_crop in enumerate(
            slot_object_detection_service.get_type_icon_crops(object_layer),
            start=1,
        ):
            type_image = type_crop.get("image")
            if cv_service.is_empty_image(type_image):
                continue
            harvested["singleTypeIcons"].append(
                write_review_crop(
                    type_image,
                    UNSORTED_SINGLE_TYPE_DIR,
                    slot_stem,
                    f"type-{type_index}",
                )
            )

    return harvested


def process_new_uploads(args: argparse.Namespace) -> dict:
    input_dir = resolve_backend_path(args.input_dir)
    outer_model_path = resolve_backend_path(args.outer_model)
    work_dir = resolve_backend_path(args.work_dir)
    state_path = work_dir / PROCESSED_STATE_NAME

    if not input_dir.exists():
        raise FileNotFoundError(f"Upload folder not found: {input_dir}")
    if not outer_model_path.exists():
        raise FileNotFoundError(f"Outer slot detector model not found: {outer_model_path}")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise ImportError("Install ultralytics before preparing upload training data.") from error

    work_dirs = ensure_work_dirs(work_dir)
    state = load_state(state_path)
    processed = state.setdefault("processed", {})
    images = list_images(input_dir)

    new_items = []
    skipped_existing = 0
    skipped_incomplete = 0
    model = YOLO(str(outer_model_path))

    for image_path in images:
        digest = file_sha1(image_path)
        if digest in processed:
            skipped_existing += 1
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
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
            skipped_incomplete += 1
            continue

        stem = safe_image_stem(image_path, digest)
        original_path = work_dirs["originals"] / f"{stem}{image_path.suffix.lower()}"
        label_path = work_dirs["labels"] / f"{stem}.txt"
        overlay_path = work_dirs["overlays"] / f"{stem}_overlay.jpg"

        shutil.copy2(image_path, original_path)
        label_path.write_text(
            "\n".join(
                yolo_label_line(item["box"], image_width, image_height)
                for item in detections
            )
            + "\n",
            encoding="utf-8",
        )
        draw_overlay(image_path, detections, overlay_path)
        crop_paths = write_slot_crops(image, detections, work_dirs["slot_crops"], stem)
        harvested = harvest_review_crops(crop_paths, stem)

        record = {
            "source": str(image_path),
            "sha1": digest,
            "original": str(original_path),
            "label": str(label_path),
            "overlay": str(overlay_path),
            "slotCrops": crop_paths,
            "harvestedReviewCrops": harvested,
            "rawDetectionCount": raw_count,
            "detectionCount": len(detections),
        }
        processed[digest] = record
        new_items.append(record)

    save_state(state_path, state)
    return {
        "inputCount": len(images),
        "newCount": len(new_items),
        "skippedExistingCount": skipped_existing,
        "skippedIncompleteCount": skipped_incomplete,
        "statePath": str(state_path),
        "slotCropDir": str(work_dirs["slot_crops"]),
        "newItems": new_items,
        "reviewOutputs": {
            "slots": str(UNSORTED_SLOT_DIR),
            "pokemon": str(UNSORTED_POKEMON_DIR),
            "singleTypeIcons": str(UNSORTED_SINGLE_TYPE_DIR),
        },
    }


def mark_existing_uploads_processed(args: argparse.Namespace) -> dict:
    input_dir = resolve_backend_path(args.input_dir)
    work_dir = resolve_backend_path(args.work_dir)
    state_path = work_dir / PROCESSED_STATE_NAME
    if not input_dir.exists():
        raise FileNotFoundError(f"Upload folder not found: {input_dir}")

    state = load_state(state_path)
    processed = state.setdefault("processed", {})
    marked = 0

    for image_path in list_images(input_dir):
        digest = file_sha1(image_path)
        if digest in processed:
            continue
        processed[digest] = {
            "source": str(image_path),
            "sha1": digest,
            "status": "marked_existing_without_processing",
        }
        marked += 1

    save_state(state_path, state)
    return {
        "markedExistingCount": marked,
        "statePath": str(state_path),
    }


def rebuild_inner_object_dataset(args: argparse.Namespace, slot_crop_dir: Path) -> None:
    builder_args = argparse.Namespace(
        input_dir=[
            *slot_object_builder.DEFAULT_INPUT_DIRS,
            slot_crop_dir,
        ],
        output=resolve_backend_path(args.object_dataset_output),
        val_ratio=args.val_ratio,
        seed=args.seed,
        pokemon_padding_x=args.pokemon_padding_x,
        pokemon_padding_y=args.pokemon_padding_y,
        type_square_padding=args.type_square_padding,
        include_review=args.include_review,
    )
    slot_object_builder.build_dataset(builder_args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare only new uploads and fold them into the inner slot-object YOLO dataset."
    )
    parser.add_argument("--input-dir", type=Path, default=UPLOAD_DIR)
    parser.add_argument("--outer-model", type=Path, default=DEFAULT_OUTER_MODEL)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--object-dataset-output", type=Path, default=DEFAULT_OBJECT_DATASET_OUTPUT)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--max-boxes", type=int, default=6)
    parser.add_argument("--keep-incomplete", action="store_true")
    parser.add_argument(
        "--mark-existing",
        action="store_true",
        help="Record the current uploads as already handled without generating crops; useful once before using the incremental pipeline.",
    )
    parser.add_argument("--skip-rebuild", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pokemon-padding-x", type=float, default=0.14)
    parser.add_argument("--pokemon-padding-y", type=float, default=0.20)
    parser.add_argument("--type-square-padding", type=float, default=0.04)
    parser.add_argument("--include-review", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mark_existing:
        print(json.dumps(mark_existing_uploads_processed(args), indent=2))
        return

    report = process_new_uploads(args)
    print(json.dumps({key: value for key, value in report.items() if key != "newItems"}, indent=2))

    if args.skip_rebuild:
        return

    rebuild_inner_object_dataset(args, Path(report["slotCropDir"]))


if __name__ == "__main__":
    main()

