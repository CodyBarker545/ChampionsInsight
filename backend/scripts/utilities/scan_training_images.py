"""
Scan ChampionsInsight image folders and write a report.

This version:
- Lists each folder containing real/non-synthetic images.
- Ignores synthetic generated images.
- Saves folder path, image count, and first image in each folder.
- Writes results to image_dataset_scan.csv and image_dataset_scan.txt.

Run from backend:
python scripts/scan_training_images.py
"""

from pathlib import Path
from collections import defaultdict
import csv


BACKEND_DIR = Path(__file__).resolve().parents[2]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

SCAN_ROOTS = [
    BACKEND_DIR / "data",
    BACKEND_DIR / "uploads",
    BACKEND_DIR / "cv",
]

IGNORE_PARTS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
}

SYNTHETIC_NAME_MARKERS = {
    "_synthetic_",
    "synthetic",
}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def should_ignore_path(path: Path) -> bool:
    return any(part in IGNORE_PARTS for part in path.parts)


def is_synthetic_image(path: Path) -> bool:
    name = path.name.lower()
    return any(marker in name for marker in SYNTHETIC_NAME_MARKERS)


def guess_purpose(folder: Path) -> str:
    text = str(folder).lower()

    if "slot_pokemon_synthetic" in text:
        return "Synthetic PokÃ©mon classifier data - ignored in count"
    if "slot_pokemon_combined" in text:
        return "Combined training folder - may contain real + synthetic"
    if "reald" in text or "real" in text:
        return "Real sorted PokÃ©mon classifier data"
    if "champions_sprites" in text:
        return "Clean Champions sprite references"
    if "debug" in text or "crop" in text:
        return "Debug/crop images - may be useful after sorting"
    if "upload" in text:
        return "Raw uploaded/camera images - useful for full slot detector"
    if "type" in text:
        return "Type icon/type combo images"
    if "sprite_detector" in text or "yolo" in text:
        return "Object detection dataset"

    return "Unknown image folder"


def scan_images():
    folder_images = defaultdict(list)
    synthetic_ignored_count = 0

    for root in SCAN_ROOTS:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if should_ignore_path(path):
                continue

            if not path.is_file() or not is_image(path):
                continue

            if is_synthetic_image(path):
                synthetic_ignored_count += 1
                continue

            folder_images[path.parent].append(path)

    return folder_images, synthetic_ignored_count


def main():
    folder_images, synthetic_ignored_count = scan_images()

    csv_path = BACKEND_DIR / "image_dataset_scan.csv"
    txt_path = BACKEND_DIR / "image_dataset_scan.txt"

    rows = []

    for folder, images in folder_images.items():
        images = sorted(images)

        try:
            relative_folder = folder.relative_to(BACKEND_DIR)
        except ValueError:
            relative_folder = folder

        first_image = images[0] if images else None

        if first_image:
            try:
                relative_first_image = first_image.relative_to(BACKEND_DIR)
            except ValueError:
                relative_first_image = first_image
        else:
            relative_first_image = ""

        rows.append({
            "folder": str(relative_folder),
            "non_synthetic_image_count": len(images),
            "first_image": str(relative_first_image),
            "purpose_guess": guess_purpose(folder),
        })

    rows.sort(key=lambda row: row["non_synthetic_image_count"], reverse=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "folder",
                "non_synthetic_image_count",
                "first_image",
                "purpose_guess",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with open(txt_path, "w", encoding="utf-8") as file:
        file.write("IMAGE DATASET SCAN\n")
        file.write("=" * 80 + "\n")
        file.write(f"Backend: {BACKEND_DIR}\n")
        file.write(f"Synthetic images ignored: {synthetic_ignored_count}\n")
        file.write(f"Folders with non-synthetic images: {len(rows)}\n")
        file.write("\n")

        for row in rows:
            file.write("-" * 80 + "\n")
            file.write(f"Folder: {row['folder']}\n")
            file.write(f"Non-synthetic images: {row['non_synthetic_image_count']}\n")
            file.write(f"First image: {row['first_image']}\n")
            file.write(f"Guess: {row['purpose_guess']}\n")

        file.write("\n")
        file.write("=" * 80 + "\n")
        file.write("Done.\n")

    print("Scan complete.")
    print(f"CSV report: {csv_path}")
    print(f"Text report: {txt_path}")
    print(f"Synthetic images ignored: {synthetic_ignored_count}")
    print(f"Folders with non-synthetic images: {len(rows)}")


if __name__ == "__main__":
    main()
