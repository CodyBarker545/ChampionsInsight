"""Run opponent detection and collect review crops into unsorted folders.

This is the crop-harvesting step before manual QA and clustering:

    python scripts/harvest_opponent_review_crops.py --input-dir data/uploads

Outputs:
    data/training_dataset/review/unsorted_pokemon/
    data/training_dataset/review/unsorted_type_combos/

Use --include-slots to also collect full red slot crops.
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
UNSORTED_SLOT_DIR = REVIEW_DIR / "unsorted_slots"
UNSORTED_POKEMON_DIR = REVIEW_DIR / "unsorted_pokemon"
UNSORTED_TYPE_COMBO_DIR = REVIEW_DIR / "unsorted_type_combos"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def resolve_backend_path(path: Path) -> Path:
    if path.is_absolute():
        return path

    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path

    return BACKEND_DIR / path


def list_images(input_dir: Path, pattern: str) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def file_hash(path: Path, length: int = 10) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()[:length]


def build_destination_name(source_path: Path, source_image: Path, crop_kind: str, position: int | str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    digest = file_hash(source_path)
    safe_source = source_image.stem.replace(" ", "_")
    return (
        f"{crop_kind}__{safe_source}__slot-{position}__"
        f"{timestamp}__{digest}{source_path.suffix.lower()}"
    )


def copy_crop(source_path: str | Path, destination_dir: Path, source_image: Path, crop_kind: str, position: int | str) -> Path | None:
    if not source_path:
        return None

    source = Path(source_path)
    if not source.exists() or source.suffix.lower() not in IMAGE_EXTENSIONS:
        return None

    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / build_destination_name(source, source_image, crop_kind, position)

    counter = 2
    while destination.exists():
        destination = destination_dir / f"{destination.stem}__{counter}{destination.suffix}"
        counter += 1

    shutil.copy2(source, destination)
    return destination


def collect_slot_crops(result: dict, source_image: Path, include_slots: bool) -> dict:
    copied = {
        "pokemon": [],
        "type_combos": [],
        "slots": [],
    }

    for slot in result.get("detectedTeam", []) or []:
        position = slot.get("position", "unknown")

        pokemon_destination = copy_crop(
            slot.get("debugPokemonCropPath", ""),
            UNSORTED_POKEMON_DIR,
            source_image,
            "pokemon",
            position,
        )
        if pokemon_destination:
            copied["pokemon"].append(str(pokemon_destination))

        for crop_info in slot.get("debugTypeIconCropPaths", []) or []:
            crop_path = crop_info.get("path", "") if isinstance(crop_info, dict) else ""
            crop_source = crop_info.get("cropSource", "") if isinstance(crop_info, dict) else ""
            if crop_source != "type_icon_cluster_debug":
                continue

            type_destination = copy_crop(
                crop_path,
                UNSORTED_TYPE_COMBO_DIR,
                source_image,
                "type_combo",
                position,
            )
            if type_destination:
                copied["type_combos"].append(str(type_destination))

        if include_slots:
            slot_destination = copy_crop(
                slot.get("debugCropPath", ""),
                UNSORTED_SLOT_DIR,
                source_image,
                "slot",
                position,
            )
            if slot_destination:
                copied["slots"].append(str(slot_destination))

    return copied


def harvest_image(image_path: Path, include_slots: bool) -> dict:
    try:
        result = detect_opponent_team(image_path, save_debug=True)
        copied = collect_slot_crops(result, image_path, include_slots)
        return {
            "image": str(image_path),
            "status": "success",
            "slotCount": len(result.get("detectedTeam", []) or []),
            "copied": copied,
        }
    except ComputerVisionError as error:
        return {
            "image": str(image_path),
            "status": "cv_error",
            "error": str(error),
            "copied": {"pokemon": [], "type_combos": [], "slots": []},
        }
    except Exception as error:
        return {
            "image": str(image_path),
            "status": "error",
            "error": repr(error),
            "copied": {"pokemon": [], "type_combos": [], "slots": []},
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Harvest opponent crop images for manual review.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=UPLOAD_DIR,
        help="Folder of opponent screenshots/photos to process.",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="Recursive glob pattern for input images.",
    )
    parser.add_argument(
        "--include-slots",
        action="store_true",
        help="Also copy full red slot crops into review/unsorted_slots.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = resolve_backend_path(args.input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    for folder in (UNSORTED_POKEMON_DIR, UNSORTED_TYPE_COMBO_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    if args.include_slots:
        UNSORTED_SLOT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(input_dir, args.pattern)
    if not image_paths:
        print(f"No images found in: {input_dir}")
        return

    print(f"Found {len(image_paths)} images.")
    print(f"Input: {input_dir}")
    print(f"Pokemon crops: {UNSORTED_POKEMON_DIR}")
    print(f"Type combo crops: {UNSORTED_TYPE_COMBO_DIR}")
    print()

    results = []
    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] {image_path.name}")
        result = harvest_image(image_path, include_slots=args.include_slots)
        results.append(result)
        copied = result.get("copied", {})
        print(
            "  -> "
            f"{result.get('status')} | "
            f"pokemon={len(copied.get('pokemon', []))}, "
            f"type_combos={len(copied.get('type_combos', []))}, "
            f"slots={len(copied.get('slots', []))}"
        )

    report = {
        "inputDir": str(input_dir),
        "pokemonOutputDir": str(UNSORTED_POKEMON_DIR),
        "typeComboOutputDir": str(UNSORTED_TYPE_COMBO_DIR),
        "results": results,
        "summary": {
            "imageCount": len(results),
            "successCount": sum(1 for item in results if item.get("status") == "success"),
            "pokemonCropCount": sum(len(item.get("copied", {}).get("pokemon", [])) for item in results),
            "typeComboCropCount": sum(len(item.get("copied", {}).get("type_combos", [])) for item in results),
            "slotCropCount": sum(len(item.get("copied", {}).get("slots", [])) for item in results),
        },
    }

    report_path = REVIEW_DIR / "harvest_opponent_review_crops_report.json"
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

