"""Process uploaded opponent images and stage their debug crops for training review.

Run from ``backend``:

    python scripts/prepare_uploads_for_training.py

This is the one-command path from:

    data/uploads/

to:

    data/training_dataset/review/unsorted_slots/
    data/training_dataset/review/unsorted_pokemon/
    data/training_dataset/review/unsorted_type_combos/
    data/training_dataset/review/unsorted_single_type_icons/

It also copies each image's full debug bundle into:

    data/training_dataset/review/imports/from_uploads/<image-stem>/

so you can inspect the original image, slot crops, Pokemon crops, combined type
areas, type clusters, and individual type icon crops together.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from paths import UPLOAD_DIR
from services.cv_detection_service import detect_opponent_team
from services.cv_service import ComputerVisionError, make_json_safe_cv_value


TRAINING_ROOT = BACKEND_DIR / "data" / "training_dataset"
REVIEW_DIR = TRAINING_ROOT / "review"
IMPORT_ROOT = REVIEW_DIR / "imports" / "from_uploads"
UNSORTED_SLOT_DIR = REVIEW_DIR / "unsorted_slots"
UNSORTED_POKEMON_DIR = REVIEW_DIR / "unsorted_pokemon"
UNSORTED_TYPE_COMBO_DIR = REVIEW_DIR / "unsorted_type_combos"
UNSORTED_SINGLE_TYPE_DIR = REVIEW_DIR / "unsorted_single_type_icons"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def file_hash(path: Path, length: int = 10) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def build_destination_name(source_path: Path, crop_kind: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    digest = file_hash(source_path)
    return f"{crop_kind}__{source_path.stem}__{timestamp}__{digest}{source_path.suffix.lower()}"


def copy_unique(source_path: Path, destination_dir: Path, crop_kind: str) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / build_destination_name(source_path, crop_kind)

    counter = 2
    while destination.exists():
        destination = destination_dir / f"{destination.stem}__{counter}{destination.suffix}"
        counter += 1

    shutil.copy2(source_path, destination)
    return destination


def copy_debug_bundle(result: dict, source_image: Path) -> Path | None:
    debug_original = Path(result.get("debugOriginalPath") or "")
    if not debug_original.exists():
        return None

    source_debug_dir = debug_original.parent
    destination_dir = IMPORT_ROOT / source_image.stem
    destination_dir.mkdir(parents=True, exist_ok=True)

    for path in source_debug_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, destination_dir / path.name)

    return destination_dir


def stage_training_crops(result: dict) -> dict[str, list[str]]:
    copied = {
        "slots": [],
        "pokemon": [],
        "type_combos": [],
        "single_type_icons": [],
    }

    for slot in result.get("detectedTeam", []) or []:
        slot_path = Path(slot.get("debugCropPath") or "")
        pokemon_path = Path(slot.get("debugPokemonCropPath") or "")

        if slot_path.exists():
            copied["slots"].append(str(copy_unique(slot_path, UNSORTED_SLOT_DIR, "slot")))

        if pokemon_path.exists():
            copied["pokemon"].append(str(copy_unique(pokemon_path, UNSORTED_POKEMON_DIR, "pokemon")))

        for crop in slot.get("debugTypeIconCropPaths", []) or []:
            crop_path = Path(crop.get("path") or "")
            crop_source = crop.get("cropSource") or ""
            if not crop_path.exists():
                continue

            if crop_source == "type_icon_cluster_debug":
                copied["type_combos"].append(
                    str(copy_unique(crop_path, UNSORTED_TYPE_COMBO_DIR, "type_combo"))
                )
            else:
                copied["single_type_icons"].append(
                    str(copy_unique(crop_path, UNSORTED_SINGLE_TYPE_DIR, "single_type"))
                )

    return copied


def process_image(image_path: Path) -> dict:
    try:
        result = detect_opponent_team(image_path, save_debug=True)
        import_dir = copy_debug_bundle(result, image_path)
        copied = stage_training_crops(result)
        return {
            "image": str(image_path),
            "status": "success",
            "importDir": str(import_dir) if import_dir else "",
            "slotCount": len(result.get("detectedTeam", []) or []),
            "copied": copied,
        }
    except ComputerVisionError as error:
        return {
            "image": str(image_path),
            "status": "cv_error",
            "error": str(error),
            "copied": {"slots": [], "pokemon": [], "type_combos": [], "single_type_icons": []},
        }
    except Exception as error:
        return {
            "image": str(image_path),
            "status": "error",
            "error": repr(error),
            "copied": {"slots": [], "pokemon": [], "type_combos": [], "single_type_icons": []},
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare uploaded images for training review.")
    parser.add_argument("--input-dir", type=Path, default=UPLOAD_DIR)
    return parser.parse_args()


def resolve_backend_path(path: Path) -> Path:
    return path if path.is_absolute() else BACKEND_DIR / path


def main() -> None:
    args = parse_args()
    input_dir = resolve_backend_path(args.input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    for folder in (
        IMPORT_ROOT,
        UNSORTED_SLOT_DIR,
        UNSORTED_POKEMON_DIR,
        UNSORTED_TYPE_COMBO_DIR,
        UNSORTED_SINGLE_TYPE_DIR,
    ):
        folder.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(input_dir)
    if not image_paths:
        print(f"No images found in: {input_dir}")
        return

    print(f"Found {len(image_paths)} uploaded images.")
    print(f"Input: {input_dir}")
    print()

    results = []
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] {image_path.name}")
        result = process_image(image_path)
        results.append(result)
        copied = result.get("copied", {})
        print(
            "  -> "
            f"{result.get('status')} | "
            f"slots={len(copied.get('slots', []))}, "
            f"pokemon={len(copied.get('pokemon', []))}, "
            f"type_combos={len(copied.get('type_combos', []))}, "
            f"single_type_icons={len(copied.get('single_type_icons', []))}"
        )

    report = {
        "inputDir": str(input_dir),
        "importRoot": str(IMPORT_ROOT),
        "outputs": {
            "slots": str(UNSORTED_SLOT_DIR),
            "pokemon": str(UNSORTED_POKEMON_DIR),
            "typeCombos": str(UNSORTED_TYPE_COMBO_DIR),
            "singleTypeIcons": str(UNSORTED_SINGLE_TYPE_DIR),
        },
        "summary": {
            "imageCount": len(results),
            "successCount": sum(1 for item in results if item.get("status") == "success"),
            "slotCropCount": sum(len(item["copied"]["slots"]) for item in results),
            "pokemonCropCount": sum(len(item["copied"]["pokemon"]) for item in results),
            "typeComboCropCount": sum(len(item["copied"]["type_combos"]) for item in results),
            "singleTypeIconCropCount": sum(len(item["copied"]["single_type_icons"]) for item in results),
        },
        "results": results,
    }

    report_path = REVIEW_DIR / "prepare_uploads_for_training_report.json"
    report_path.write_text(
        json.dumps(make_json_safe_cv_value(report), indent=2),
        encoding="utf-8",
    )

    print()
    print("Done.")
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()

