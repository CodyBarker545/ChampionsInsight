"""Builds the Pokémon embedding index used by opponent detection."""

import json
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


def build_pokemon_embedding_index():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    references = cv_service.load_reference_images(
        reference_dir=cv_service.REFERENCE_IMAGE_DIR,
        metadata_path=cv_service.REFERENCE_METADATA_PATH,
        extra_reference_dir=cv_service.EXTRA_REFERENCE_IMAGE_DIR,
    )

    embeddings = []
    metadata = []

    print(f"Loaded {len(references)} Pokémon reference images.")

    for index, reference in enumerate(references, start=1):
        image = reference.get("image")

        if cv_service.is_empty_image(image):
            continue

        try:
            vector = embed_image(image)
        except Exception as error:
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
            print(f"Embedded {index}/{len(references)} references...")

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


if __name__ == "__main__":
    build_pokemon_embedding_index()
