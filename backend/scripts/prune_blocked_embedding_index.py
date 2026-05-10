"""Removes blocked Pokémon forms from the existing embedding index."""

"""Removes blocked Pokémon forms from the existing embedding index."""

import json
import re
import sys
from pathlib import Path

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from paths import BACKEND_DIR as PATHS_BACKEND_DIR, POKEMON_EMBEDDING_INDEX_DIR

# Run this script from the backend folder:
# python scripts\prune_blocked_embedding_index.py

INDEX_DIR = POKEMON_EMBEDDING_INDEX_DIR
EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
METADATA_PATH = INDEX_DIR / "metadata.json"


ALLOWED_MEGA_NAMES = {
    # Add exceptions only if these are valid in your game/app.
    # "Aggron Mega",
    # "Sableye Mega",
}


def normalize_path(value):
    return str(value or "").replace("\\", "/").lower()


def is_blocked(record):
    name = str(record.get("name", "") or "")
    species = str(record.get("species", "") or "")
    form = str(record.get("form", "") or "")
    path = normalize_path(record.get("path", ""))

    if name in ALLOWED_MEGA_NAMES:
        return False

    # Blocks "Charizard Mega X", but not "Meganium".
    for value in [name, species, form]:
        if re.search(r"\bmega\b", value, flags=re.IGNORECASE):
            return True

    # Blocks filenames like Menu_CP_0006-Mega_X.png or sableye-mega-guided-crop.jpg.
    if re.search(r"(^|[-_/])mega($|[-_. /])", path, flags=re.IGNORECASE):
        return True

    if "gigantamax" in path:
        return True

    return False


def main():
    print(f"Backend dir:     {BACKEND_DIR}")
    print(f"Embeddings path: {EMBEDDINGS_PATH}")
    print(f"Metadata path:   {METADATA_PATH}")

    if not EMBEDDINGS_PATH.exists():
        raise FileNotFoundError(f"Missing embeddings file: {EMBEDDINGS_PATH}")

    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Missing metadata file: {METADATA_PATH}")

    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    if len(embeddings) != len(metadata):
        raise RuntimeError(
            f"embeddings.npy and metadata.json lengths do not match: "
            f"{len(embeddings)} embeddings vs {len(metadata)} metadata records"
        )

    kept_embeddings = []
    kept_metadata = []
    removed = []

    for vector, record in zip(embeddings, metadata):
        if is_blocked(record):
            removed.append(record)
            continue

        kept_embeddings.append(vector)
        kept_metadata.append(record)

    if not kept_embeddings:
        raise RuntimeError("Pruning removed every embedding. Check filter logic.")

    np.save(EMBEDDINGS_PATH, np.vstack(kept_embeddings).astype("float32"))
    METADATA_PATH.write_text(
        json.dumps(kept_metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(f"Original count: {len(metadata)}")
    print(f"Removed count:  {len(removed)}")
    print(f"New count:      {len(kept_metadata)}")

    print()
    print("First removed:")
    for record in removed[:20]:
        print(" -", record.get("name"), record.get("path"))


if __name__ == "__main__":
    main()
