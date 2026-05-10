"""Loads the sentence embedding model used by RAG retrieval."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TF", "0")


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE_DIR = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--sentence-transformers--all-MiniLM-L6-v2"
    / "snapshots"
)


def get_local_model_path() -> Path | None:
    """Return the newest downloaded embedding model path, if it exists."""

    if not MODEL_CACHE_DIR.exists():
        return None

    snapshots = sorted(
        [path for path in MODEL_CACHE_DIR.iterdir() if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return snapshots[0] if snapshots else None


def load_embedding_model() -> SentenceTransformer:
    """Load the embedding model locally when possible, otherwise download it."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "RAG dependencies are missing. Install backend requirements first."
        ) from error

    local_model_path = get_local_model_path()

    if local_model_path is not None:
        return SentenceTransformer(str(local_model_path), local_files_only=True)

    return SentenceTransformer(MODEL_NAME)
