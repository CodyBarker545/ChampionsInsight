"""Builds the type icon embedding index from single-type and combo-type icons."""

import json
import sys
from pathlib import Path

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from paths import TYPE_EMBEDDING_INDEX_DIR
from services import cv_service
from services.embedding_model_service import embed_image
from services.type_embedding_service import (
    preprocess_type_icon_for_embedding,
    preprocess_type_icon_glare_reduced,
)


OUTPUT_DIR = TYPE_EMBEDDING_INDEX_DIR
EMBEDDINGS_PATH = OUTPUT_DIR / "embeddings.npy"
METADATA_PATH = OUTPUT_DIR / "metadata.json"

TYPE_COMBO_ICON_DIR = BACKEND_DIR / "data" / "cv" / "references" / "types" / "type_combo_icons"


def parse_type_combo_from_filename(path: Path):
    """
    Converts filenames like:
      bug.png -> ["bug"]
      bug_fire.png -> ["bug", "fire"]
      dark_ice_camera-1.jpg -> ["dark", "ice"]

    Camera/debug suffixes are removed.
    """
    stem = path.stem.lower().strip()

    # Remove common suffixes from camera/debug references.
    stem = stem.replace("_camera", "")
    stem = stem.replace("-camera", "")

    # Remove trailing numbered suffixes like -1, _1.
    parts = stem.replace("-", "_").split("_")
    parts = [part for part in parts if not part.isdigit()]

    valid_types = {
        "normal", "fire", "water", "electric", "grass", "ice",
        "fighting", "poison", "ground", "flying", "psychic", "bug",
        "rock", "ghost", "dragon", "dark", "steel", "fairy",
    }

    types = [part for part in parts if part in valid_types]

    if not types:
        return []

    # Keep max 2 because Pokémon normally have 1 or 2 types.
    return types[:2]


def load_type_combo_icon_references(folder: Path):
    references = []

    if not folder.exists():
        raise FileNotFoundError(f"Type combo icon folder not found: {folder}")

    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}

    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in image_extensions:
            continue

        image = cv_service.read_cv_image(path)

        if cv_service.is_empty_image(image):
            print(f"Skipping empty image: {path}")
            continue

        types = parse_type_combo_from_filename(path)

        if not types:
            print(f"Skipping file with unknown type name: {path.name}")
            continue

        references.append({
            "type": types[0],
            "types": types,
            "path": str(path),
            "image": image,
        })

    return references


def build_type_embedding_index():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    references = load_type_combo_icon_references(TYPE_COMBO_ICON_DIR)

    embeddings = []
    metadata = []

    print(f"Loaded {len(references)} type combo icon references from:")
    print(TYPE_COMBO_ICON_DIR)

    for reference in references:
        icon_type = reference.get("type", "")
        icon_types = reference.get("types", [])
        image = reference.get("image")
        path = reference.get("path", "")

        if cv_service.is_empty_image(image):
            continue

        variants = [
            ("original", image),
            ("lab_normalized", preprocess_type_icon_for_embedding(image)),
            ("glare_reduced", preprocess_type_icon_glare_reduced(image)),
        ]

        for variant_name, variant_image in variants:
            try:
                vector = embed_image(variant_image)
            except Exception as error:
                print(f"Skipping {path} {variant_name}: {error}")
                continue

            embeddings.append(vector)
            metadata.append({
                "type": icon_type,
                "types": icon_types,
                "path": path,
                "variant": variant_name,
            })

    if not embeddings:
        raise RuntimeError("No type icon embeddings were created.")

    embedding_matrix = np.vstack(embeddings).astype("float32")

    np.save(EMBEDDINGS_PATH, embedding_matrix)

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("Type embedding index built.")
    print(f"Embeddings: {EMBEDDINGS_PATH}")
    print(f"Metadata:   {METADATA_PATH}")
    print(f"References: {len(references)}")
    print(f"Count:      {len(metadata)} embeddings")

    expected_count = len(references) * 3
    print(f"Expected:   {expected_count} embeddings because each image creates 3 variants")


if __name__ == "__main__":
    build_type_embedding_index()