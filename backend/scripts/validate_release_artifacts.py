"""Validate prebuilt runtime artifacts shipped in the test-machine release."""

from __future__ import annotations

from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]


REQUIRED_ARTIFACTS = [
    BACKEND_DIR / "vector_store" / "pokemon_rag_embeddings.npy",
    BACKEND_DIR / "vector_store" / "chunks.json",
    BACKEND_DIR
    / "data"
    / "cv"
    / "models"
    / "pokemon_mobilenetv3"
    / "pokemon_mobilenetv3.pt",
    BACKEND_DIR / "data" / "cv" / "models" / "pokemon_mobilenetv3" / "idx_to_class.json",
    BACKEND_DIR / "data" / "cv" / "models" / "pokemon_mobilenetv3" / "class_to_idx.json",
    BACKEND_DIR
    / "data"
    / "cv"
    / "models"
    / "slot_object_detector"
    / "yolov8n_slot_objects_padded_clean_types"
    / "weights"
    / "best.pt",
]


REQUIRED_DIRECTORIES = [
    BACKEND_DIR / "RAGDocs",
    BACKEND_DIR / "data" / "pokemon" / "champions_sprites" / "normal",
]


def describe_size(path: Path) -> str:
    if path.is_file():
        return f"{path.stat().st_size / (1024 * 1024):.2f} MB"
    return "directory"


def main() -> None:
    missing = [
        path
        for path in [*REQUIRED_DIRECTORIES, *REQUIRED_ARTIFACTS]
        if not path.exists()
    ]

    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "This release is missing required prebuilt AI/runtime artifacts.\n"
            "Do not train on the test machine; rebuild the release zip from the source machine.\n"
            f"{missing_list}"
        )

    print("Required prebuilt AI/runtime artifacts are present:")
    for path in REQUIRED_ARTIFACTS:
        print(f"  - {path.relative_to(BACKEND_DIR)} ({describe_size(path)})")


if __name__ == "__main__":
    main()
