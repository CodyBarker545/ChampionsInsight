"""Build the type icon embedding index from real and synthetic single-type crops."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[2]
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

DEFAULT_INPUT_DIRS = [
    BACKEND_DIR / "data" / "training_dataset" / "types",
    BACKEND_DIR / "data" / "training_dataset" / "type_synthetic",
]
VALID_TYPES = {
    "normal", "fire", "water", "electric", "grass", "ice",
    "fighting", "poison", "ground", "flying", "psychic", "bug",
    "rock", "ghost", "dragon", "dark", "steel", "fairy",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


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

    types = [part for part in parts if part in VALID_TYPES]

    if not types:
        return []

    # Keep max 2 because PokÃ©mon normally have 1 or 2 types.
    return types[:2]


def load_type_icon_references(folders: list[Path]):
    references = []

    for folder in folders:
        if not folder.exists():
            raise FileNotFoundError(f"Type icon folder not found: {folder}")

        for type_dir in sorted(path for path in folder.iterdir() if path.is_dir()):
            icon_type = type_dir.name.lower().strip()
            if icon_type not in VALID_TYPES:
                continue

            for path in sorted(type_dir.iterdir()):
                if path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                image = cv_service.read_cv_image(path)

                if cv_service.is_empty_image(image):
                    print(f"Skipping empty image: {path}")
                    continue

                references.append({
                    "type": icon_type,
                    "types": [icon_type],
                    "path": str(path),
                    "image": image,
                    "source": folder.name,
                })

    return references


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build type embedding index.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        action="append",
        default=None,
        help="Folder containing one subfolder per single type. Repeat to add more sources.",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def resolve_backend_path(path: Path) -> Path:
    return path if path.is_absolute() else BACKEND_DIR / path


def build_type_embedding_index(args: argparse.Namespace):
    output_dir = resolve_backend_path(args.output_dir)
    embeddings_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "metadata.json"
    input_dirs = [resolve_backend_path(path) for path in (args.input_dir or DEFAULT_INPUT_DIRS)]

    output_dir.mkdir(parents=True, exist_ok=True)

    references = load_type_icon_references(input_dirs)

    embeddings = []
    metadata = []

    print(f"Loaded {len(references)} single-type icon references from:")
    for input_dir in input_dirs:
        print(input_dir)

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
                "source": reference.get("source", ""),
            })

    if not embeddings:
        raise RuntimeError("No type icon embeddings were created.")

    embedding_matrix = np.vstack(embeddings).astype("float32")

    np.save(embeddings_path, embedding_matrix)

    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("Type embedding index built.")
    print(f"Embeddings: {embeddings_path}")
    print(f"Metadata:   {metadata_path}")
    print(f"References: {len(references)}")
    print(f"Count:      {len(metadata)} embeddings")

    expected_count = len(references) * 3
    print(f"Expected:   {expected_count} embeddings because each image creates 3 variants")


if __name__ == "__main__":
    build_type_embedding_index(parse_args())

