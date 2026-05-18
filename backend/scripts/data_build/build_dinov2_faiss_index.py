"""Build a DINOv2 + FAISS index from one or more image-folder datasets."""

from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from paths import CV_INDEX_DIR
from services.dinov2_faiss_service import (
    DEFAULT_DINOV2_MODEL,
    IMAGE_EXTENSIONS,
    embed_images,
    load_pil_image,
    write_faiss_index,
    write_metadata,
)


DEFAULT_OUTPUT_ROOT = CV_INDEX_DIR / "dinov2_faiss"


def stable_source_id(path: Path) -> str:
    text = str(path).replace("\\", "/").lower()
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def collect_image_folder_sources(
    input_dirs: list[Path],
    max_per_class: int | None,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    by_label: dict[str, list[Path]] = defaultdict(list)

    for input_dir in input_dirs:
        input_dir = Path(input_dir)
        if not input_dir.exists():
            raise FileNotFoundError(f"Input folder not found: {input_dir}")

        for class_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
            label = class_dir.name
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    by_label[label].append(path)

    records = []
    for label, paths in sorted(by_label.items()):
        paths = sorted(set(paths))
        if max_per_class and len(paths) > max_per_class:
            paths = rng.sample(paths, max_per_class)
            paths.sort()

        for path in paths:
            records.append({
                "label": label,
                "path": str(path),
                "sourceId": stable_source_id(path),
            })

    return records


def build_index(
    name: str,
    input_dirs: list[Path],
    output_root: Path,
    model_name: str,
    batch_size: int,
    max_per_class: int | None,
    seed: int,
) -> None:
    output_dir = output_root / name
    index_path = output_dir / "index.faiss"
    embeddings_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "metadata.json"

    records = collect_image_folder_sources(
        input_dirs=input_dirs,
        max_per_class=max_per_class,
        seed=seed,
    )
    if not records:
        raise RuntimeError("No images found for index build.")

    labels = sorted({record["label"] for record in records})
    print(f"Index: {name}")
    print(f"Model: {model_name}")
    print(f"Input dirs: {len(input_dirs)}")
    for input_dir in input_dirs:
        print(f"  - {input_dir}")
    print(f"Classes: {len(labels)}")
    print(f"Images: {len(records)}")
    print(f"Batch size: {batch_size}")
    print(f"Max per class: {max_per_class or 'all'}")
    print()

    embeddings = []
    metadata = []
    skipped = []

    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        images = []
        kept_records = []

        for record in batch:
            try:
                images.append(load_pil_image(Path(record["path"])))
                kept_records.append(record)
            except Exception as exc:
                skipped.append((record["path"], str(exc)))

        if images:
            batch_embeddings = embed_images(images, model_name=model_name)
            embeddings.append(batch_embeddings)
            metadata.extend(kept_records)

        done = min(start + batch_size, len(records))
        print(f"Embedded {done}/{len(records)}")

    if not embeddings:
        raise RuntimeError("No embeddings were created.")

    matrix = np.vstack(embeddings).astype("float32")

    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, matrix)
    write_faiss_index(matrix, index_path)
    write_metadata(metadata, metadata_path)

    print()
    print("DINOv2 + FAISS index built.")
    print(f"Output:     {output_dir}")
    print(f"Index:      {index_path}")
    print(f"Embeddings: {embeddings_path}")
    print(f"Metadata:   {metadata_path}")
    print(f"Vectors:    {matrix.shape[0]}")
    print(f"Dim:        {matrix.shape[1]}")
    print(f"Skipped:    {len(skipped)}")
    if skipped:
        print("First skipped:")
        for path, reason in skipped[:20]:
            print(f"  - {path}: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="Index name, e.g. pokemon or types.")
    parser.add_argument("--input-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_DINOV2_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    build_index(
        name=args.name,
        input_dirs=args.input_dir,
        output_root=args.output_root,
        model_name=args.model,
        batch_size=args.batch_size,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

