# -*- coding: utf-8 -*-

from pathlib import Path
import sys

# Make backend folder importable when this script runs from backend/scripts
BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))

import cv2

from services.cv_service import (
    prepare_opponent_team_column_image,
    crop_opponent_team_slots,
)

input_dir = BACKEND_DIR / "data" / "cv" / "test_10_uploads"
slot_output = BACKEND_DIR / "data" / "training_dataset" / "review" / "unsorted_slots"
pokemon_output = BACKEND_DIR / "data" / "training_dataset" / "review" / "unsorted_pokemon"

slot_output.mkdir(parents=True, exist_ok=True)
pokemon_output.mkdir(parents=True, exist_ok=True)

image_paths = sorted([
    p for p in input_dir.iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
])

print(f"Found {len(image_paths)} upload images")

for image_index, image_path in enumerate(image_paths, start=1):
    print(f"Processing {image_index}/{len(image_paths)}: {image_path.name}")

    prepared = prepare_opponent_team_column_image(
        image_path,
        save_debug=True,
    )

    working_image = Path(prepared.get("imagePath", image_path))

    slots = crop_opponent_team_slots(
        working_image,
        save_debug=True,
    )

    for slot in slots:
        position = slot["position"]
        slot_image = slot["image"]

        slot_path = slot_output / f"{image_path.stem}_slot_{position}.jpg"
        cv2.imwrite(str(slot_path), slot_image)

        h, w = slot_image.shape[:2]

        # Approximate pokemon sprite crop from left side of slot.
        pokemon_crop = slot_image[
            int(h * 0.05):int(h * 0.95),
            int(w * 0.00):int(w * 0.42)
        ]

        pokemon_path = pokemon_output / f"{image_path.stem}_pokemon_{position}.jpg"
        cv2.imwrite(str(pokemon_path), pokemon_crop)

print("Done.")
print(f"Slot crops: {slot_output}")
print(f"Pokemon crops: {pokemon_output}")

