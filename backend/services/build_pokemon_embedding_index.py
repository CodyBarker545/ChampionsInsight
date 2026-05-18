"""Build the Pokemon embedding index from curated synthetic and real slot crops."""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from paths import POKEMON_EMBEDDING_INDEX_DIR, SPRITE_METADATA_PATH
from services import cv_service
from services.embedding_model_service import (
    center_pad_square,
    convert_cv_image_to_pil,
    load_embedding_model,
)


OUTPUT_DIR = POKEMON_EMBEDDING_INDEX_DIR
EMBEDDINGS_PATH = OUTPUT_DIR / "embeddings.npy"
METADATA_PATH = OUTPUT_DIR / "metadata.json"

DEFAULT_SYNTHETIC_DIR = BACKEND_DIR / "data" / "training_dataset" / "slot_pokemon_synthetic"
DEFAULT_REAL_DIR = BACKEND_DIR / "data" / "training_dataset" / "slot_pokemon_real"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
BATCH_SIZE = 64


def normalize_label(value: str) -> str:
    value = str(value or "").strip().lower()
    value = value.replace("-", "_").replace(" ", "_")
    value = re.sub(r"[^a-z0-9_]+", "_", value)
    return re.sub(r"_+", "_", value).strip("_")


def pretty_label(value: str) -> str:
    return " ".join(part.capitalize() for part in normalize_label(value).split("_") if part)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def is_shiny_image(path: Path) -> bool:
    return "_shiny_" in path.name.lower()


def load_class_metadata_lookup() -> dict[str, dict]:
    sprite_records = json.loads(SPRITE_METADATA_PATH.read_text(encoding="utf-8"))
    lookup: dict[str, dict] = {}

    def store(key: str, record: dict) -> None:
        key = normalize_label(key)
        if not key or key in lookup:
            return
        lookup[key] = {
            "name": record.get("display_name") or record.get("species_display_name") or pretty_label(key),
            "species": record.get("species_display_name") or record.get("display_name") or pretty_label(key),
            "form": record.get("form_display_name") or record.get("display_name") or pretty_label(key),
            "types": record.get("types", []),
        }

    for record in sprite_records:
        store(record.get("form_api_name", ""), record)
        store(record.get("display_name", ""), record)
        store(record.get("form_display_name", ""), record)

        form_api_name = normalize_label(record.get("form_api_name", ""))
        species_name = normalize_label(record.get("species_display_name", ""))
        sprite_form = str(record.get("sprite_form", "") or "").strip()
        is_default_species_form = not sprite_form or form_api_name == species_name
        if is_default_species_form:
            store(record.get("species_display_name", ""), record)

    return lookup


def resolve_class_metadata(class_name: str, metadata_lookup: dict[str, dict]) -> dict:
    normalized = normalize_label(class_name)
    match = metadata_lookup.get(normalized)
    if match:
        return match
    return {
        "name": pretty_label(class_name),
        "species": pretty_label(class_name),
        "form": pretty_label(class_name),
        "types": [],
    }


def collect_variant_samples(class_dir: Path, sample_size: int, rng: random.Random) -> list[Path]:
    images = [path for path in class_dir.iterdir() if is_image_file(path)]
    normal_images = sorted(path for path in images if not is_shiny_image(path))
    shiny_images = sorted(path for path in images if is_shiny_image(path))

    selected: list[Path] = []
    for variant_images in (normal_images, shiny_images):
        if len(variant_images) <= sample_size:
            selected.extend(variant_images)
        else:
            selected.extend(sorted(rng.sample(variant_images, sample_size), key=lambda path: path.name.lower()))

    return selected


def collect_real_images(class_dir: Path) -> list[Path]:
    return sorted(path for path in class_dir.iterdir() if is_image_file(path))


def collect_embedding_sources(
    synthetic_dir: Path,
    real_dir: Path,
    *,
    synthetic_samples_per_variant: int,
    seed: int,
) -> tuple[list[dict], list[str]]:
    metadata_lookup = load_class_metadata_lookup()
    all_class_names = sorted(
        {
            path.name
            for root in (synthetic_dir, real_dir)
            if root.exists()
            for path in root.iterdir()
            if path.is_dir()
        }
    )

    rng = random.Random(seed)
    selected_sources: list[dict] = []
    missing_type_metadata: list[str] = []

    for class_name in all_class_names:
        class_metadata = resolve_class_metadata(class_name, metadata_lookup)
        if not class_metadata.get("types"):
            missing_type_metadata.append(class_name)

        synthetic_class_dir = synthetic_dir / class_name
        if synthetic_class_dir.exists():
            for image_path in collect_variant_samples(
                synthetic_class_dir,
                synthetic_samples_per_variant,
                rng,
            ):
                selected_sources.append(
                    {
                        "name": class_metadata["name"],
                        "species": class_metadata["species"],
                        "form": class_metadata["form"],
                        "types": class_metadata.get("types", []),
                        "path": str(image_path),
                        "isShiny": is_shiny_image(image_path),
                        "datasetSource": "slot_pokemon_synthetic",
                        "className": class_name,
                    }
                )

        real_class_dir = real_dir / class_name
        if real_class_dir.exists():
            for image_path in collect_real_images(real_class_dir):
                selected_sources.append(
                    {
                        "name": class_metadata["name"],
                        "species": class_metadata["species"],
                        "form": class_metadata["form"],
                        "types": class_metadata.get("types", []),
                        "path": str(image_path),
                        "isShiny": False,
                        "datasetSource": "slot_pokemon_real",
                        "className": class_name,
                    }
                )

    return selected_sources, sorted(set(missing_type_metadata))


def build_embedding_batch(sources: list[dict]) -> tuple[list[np.ndarray], list[dict], int, int]:
    model, transform, torch, device = load_embedding_model()
    batch_tensors = []
    batch_sources = []
    skipped_empty = 0
    skipped_errors = 0

    for source in sources:
        try:
            image = cv_service.read_cv_image(source["path"])
            if cv_service.is_empty_image(image):
                skipped_empty += 1
                continue

            pil_image = convert_cv_image_to_pil(image)
            pil_image = center_pad_square(pil_image)
            batch_tensors.append(transform(pil_image))
            batch_sources.append(source)
        except Exception as error:
            skipped_errors += 1
            print(f"Skipping {source['path']}: {error}")

    if not batch_tensors:
        return [], [], skipped_empty, skipped_errors

    tensor_batch = torch.stack(batch_tensors, dim=0).to(device)
    with torch.no_grad():
        vectors = model(tensor_batch).detach().cpu().numpy().astype("float32")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms <= 0, 1.0, norms)
    normalized_vectors = vectors / norms

    return [vector for vector in normalized_vectors], batch_sources, skipped_empty, skipped_errors


def build_pokemon_embedding_index(
    *,
    synthetic_dir: Path = DEFAULT_SYNTHETIC_DIR,
    real_dir: Path = DEFAULT_REAL_DIR,
    synthetic_samples_per_variant: int = 250,
    seed: int = 10,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not synthetic_dir.exists():
        raise FileNotFoundError(f"Synthetic dataset folder does not exist: {synthetic_dir}")
    if not real_dir.exists():
        raise FileNotFoundError(f"Real dataset folder does not exist: {real_dir}")

    sources, missing_type_metadata = collect_embedding_sources(
        synthetic_dir=synthetic_dir,
        real_dir=real_dir,
        synthetic_samples_per_variant=synthetic_samples_per_variant,
        seed=seed,
    )

    embeddings = []
    metadata = []
    skipped_empty = 0
    skipped_errors = 0

    synthetic_count = sum(1 for item in sources if item["datasetSource"] == "slot_pokemon_synthetic")
    real_count = sum(1 for item in sources if item["datasetSource"] == "slot_pokemon_real")

    print(f"Synthetic source dir: {synthetic_dir}")
    print(f"Real source dir:      {real_dir}")
    print(f"Synthetic sample cap: {synthetic_samples_per_variant} per variant")
    print(f"Selected synthetic images: {synthetic_count}")
    print(f"Selected real images:      {real_count}")
    print(f"Total selected images:     {len(sources)}")

    for batch_start in range(0, len(sources), BATCH_SIZE):
        batch_sources = sources[batch_start:batch_start + BATCH_SIZE]
        vectors, kept_sources, batch_skipped_empty, batch_skipped_errors = build_embedding_batch(
            batch_sources
        )
        skipped_empty += batch_skipped_empty
        skipped_errors += batch_skipped_errors

        embeddings.extend(vectors)
        metadata.extend(kept_sources)

        processed = min(batch_start + len(batch_sources), len(sources))
        if processed % 512 == 0 or processed == len(sources):
            print(f"Embedded {processed}/{len(sources)} images...")

    if not embeddings:
        raise RuntimeError("No Pokemon embeddings were created.")

    embedding_matrix = np.vstack(embeddings).astype("float32")
    np.save(EMBEDDINGS_PATH, embedding_matrix)
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Pokemon embedding index built.")
    print(f"Embeddings: {EMBEDDINGS_PATH}")
    print(f"Metadata:   {METADATA_PATH}")
    print(f"Count:      {len(metadata)}")
    print(f"Skipped empty images: {skipped_empty}")
    print(f"Skipped errors: {skipped_errors}")
    if missing_type_metadata:
        print("Classes missing type metadata:")
        for class_name in missing_type_metadata:
            print(f"  - {class_name}")


if __name__ == "__main__":
    build_pokemon_embedding_index()
