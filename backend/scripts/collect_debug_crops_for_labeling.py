"""Collect debug crops into unsorted training folders for manual labeling.

This script scans:
    backend/data/cv/debug/crops/

It copies:
    opponent-slot-*.jpg         -> training_dataset/review/unsorted_slots/
    opponent-pokemon-*.jpg      -> training_dataset/review/unsorted_pokemon/
    opponent-type-cluster-*.jpg -> training_dataset/review/unsorted_type_combos/
    opponent-type-icon-*.jpg    -> training_dataset/review/unsorted_single_type_icons/

Run from backend:
    python scripts/collect_debug_crops_for_labeling.py

Optional dry run:
    python scripts/collect_debug_crops_for_labeling.py --dry-run
"""

import argparse
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from paths import OPPONENT_DEBUG_CROP_DIR


TRAINING_ROOT = BACKEND_DIR / "data" / "training_dataset"
REVIEW_DIR = TRAINING_ROOT / "review"

UNSORTED_SLOT_DIR = REVIEW_DIR / "unsorted_slots"
UNSORTED_POKEMON_DIR = REVIEW_DIR / "unsorted_pokemon"
UNSORTED_TYPE_COMBO_DIR = REVIEW_DIR / "unsorted_type_combos"
UNSORTED_SINGLE_TYPE_DIR = REVIEW_DIR / "unsorted_single_type_icons"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def file_hash(path, length=10):
    digest = hashlib.sha1()

    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()[:length]


def build_unique_name(source_path, crop_kind):
    image_folder = source_path.parent.name
    source_stem = source_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    digest = file_hash(source_path)

    return (
        f"{crop_kind}__{image_folder}__{source_stem}__"
        f"{timestamp}__{digest}{source_path.suffix.lower()}"
    )


def get_destination_for_crop(source_path):
    name = source_path.name.lower()

    if name.startswith("opponent-slot-"):
        return UNSORTED_SLOT_DIR, "slot"

    if name.startswith("opponent-pokemon-"):
        return UNSORTED_POKEMON_DIR, "pokemon"

    if name.startswith("opponent-type-cluster-"):
        return UNSORTED_TYPE_COMBO_DIR, "type_combo"

    if name.startswith("opponent-type-icon-"):
        return UNSORTED_SINGLE_TYPE_DIR, "single_type"

    return None, None


def collect_debug_crops(dry_run=False):
    if not OPPONENT_DEBUG_CROP_DIR.exists():
        raise FileNotFoundError(
            f"Debug crop folder was not found: {OPPONENT_DEBUG_CROP_DIR}"
        )

    for folder in (
        UNSORTED_SLOT_DIR,
        UNSORTED_POKEMON_DIR,
        UNSORTED_TYPE_COMBO_DIR,
        UNSORTED_SINGLE_TYPE_DIR,
    ):
        folder.mkdir(parents=True, exist_ok=True)

    copied_counts = {
        "slot": 0,
        "pokemon": 0,
        "type_combo": 0,
        "single_type": 0,
        "skipped": 0,
    }

    crop_paths = sorted(
        path
        for path in OPPONENT_DEBUG_CROP_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for source_path in crop_paths:
        destination_dir, crop_kind = get_destination_for_crop(source_path)

        if destination_dir is None:
            copied_counts["skipped"] += 1
            continue

        destination_name = build_unique_name(source_path, crop_kind)
        destination_path = destination_dir / destination_name

        counter = 2
        while destination_path.exists():
            destination_path = destination_dir / (
                f"{destination_path.stem}__{counter}{destination_path.suffix}"
            )
            counter += 1

        if dry_run:
            print(f"[DRY RUN] {source_path} -> {destination_path}")
        else:
            shutil.copy2(source_path, destination_path)
            print(f"Copied {source_path.name} -> {destination_path}")

        copied_counts[crop_kind] += 1

    print()
    print("Done.")
    print(f"Debug crop root: {OPPONENT_DEBUG_CROP_DIR}")
    print(f"Training review root: {REVIEW_DIR}")
    print()
    print("Copied counts:")
    print(f"  Full slot crops:         {copied_counts['slot']}")
    print(f"  Pokémon sprite crops:    {copied_counts['pokemon']}")
    print(f"  Type combo crops:        {copied_counts['type_combo']}")
    print(f"  Single type icon crops:  {copied_counts['single_type']}")
    print(f"  Skipped other files:     {copied_counts['skipped']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without copying files.",
    )

    args = parser.parse_args()

    collect_debug_crops(dry_run=args.dry_run)


if __name__ == "__main__":
    main()