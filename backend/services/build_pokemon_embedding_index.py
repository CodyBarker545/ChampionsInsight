"""Builds the Pokémon embedding index used by opponent detection."""

import json
import re
import sys
from pathlib import Path

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from paths import POKEMON_EMBEDDING_INDEX_DIR
from services import cv_service
from services.embedding_model_service import embed_image


OUTPUT_DIR = POKEMON_EMBEDDING_INDEX_DIR
EMBEDDINGS_PATH = OUTPUT_DIR / "embeddings.npy"
METADATA_PATH = OUTPUT_DIR / "metadata.json"

ALLOWED_MEGA_NAMES = {
    # Add names here only if they should be selectable.
    # Example:
    # "Aggron Mega",
    # "Sableye Mega",
}


def normalize_path_text(value):
    return str(value or "").replace("\\", "/").lower()


def is_blocked_mega_reference(reference):
    name = str(reference.get("name", "") or "")
    species = str(reference.get("species", "") or "")
    form = str(reference.get("form", "") or "")
    path = normalize_path_text(reference.get("path", ""))

    if name in ALLOWED_MEGA_NAMES:
        return False

    # Blocks display/form names like "Charizard Mega X", but does NOT block "Meganium".
    name_fields = [name, species, form]
    for field in name_fields:
        if re.search(r"\bmega\b", str(field), flags=re.IGNORECASE):
            return True

    # Blocks filenames/folders like Menu_CP_0006-Mega_X.png or sableye-mega-guided-crop.jpg.
    if re.search(r"(^|[-_/])mega($|[-_. /])", path, flags=re.IGNORECASE):
        return True

    if "gigantamax" in path:
        return True

    return False


def build_pokemon_embedding_index():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    references = cv_service.load_reference_images(
        reference_dir=cv_service.REFERENCE_IMAGE_DIR,
        metadata_path=cv_service.REFERENCE_METADATA_PATH,
        extra_reference_dir=cv_service.EXTRA_REFERENCE_IMAGE_DIR,
    )

    embeddings = []
    metadata = []
    skipped_blocked = 0
    skipped_empty = 0
    skipped_errors = 0

    print(f"Loaded {len(references)} Pokémon reference images.")

    for index, reference in enumerate(references, start=1):
        if is_blocked_mega_reference(reference):
            skipped_blocked += 1
            continue

        image = reference.get("image")

        if cv_service.is_empty_image(image):
            skipped_empty += 1
            continue

        try:
            vector = embed_image(image)
        except Exception as error:
            skipped_errors += 1
            print(f"Skipping {reference.get('path')}: {error}")
            continue

        embeddings.append(vector)
        metadata.append({
            "name": reference.get("name", ""),
            "species": reference.get("species", ""),
            "form": reference.get("form", ""),
            "types": reference.get("types", []),
            "path": reference.get("path", ""),
            "isShiny": bool(reference.get("isShiny", False)),
        })

        if index % 50 == 0:
            print(f"Processed {index}/{len(references)} references...")

    if not embeddings:
        raise RuntimeError("No Pokémon embeddings were created.")

    embedding_matrix = np.vstack(embeddings).astype("float32")

    np.save(EMBEDDINGS_PATH, embedding_matrix)
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Pokémon embedding index built.")
    print(f"Embeddings: {EMBEDDINGS_PATH}")
    print(f"Metadata:   {METADATA_PATH}")
    print(f"Count:      {len(metadata)}")
    print(f"Skipped blocked Mega/Gigantamax references: {skipped_blocked}")
    print(f"Skipped empty images: {skipped_empty}")
    print(f"Skipped errors: {skipped_errors}")


if __name__ == "__main__":
    build_pokemon_embedding_index()
