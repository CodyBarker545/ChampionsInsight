"""Builds the RAG vector index from knowledge documents."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from rag.embedding_model import MODEL_NAME, load_embedding_model
from rag.preprocess import chunk_documents, load_documents


PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = PROJECT_ROOT / "vector_store"
INDEX_PATH = INDEX_DIR / "pokemon_rag_embeddings.npy"
CHUNKS_PATH = INDEX_DIR / "chunks.json"


@dataclass
class NumpyVectorIndex:
    """Small cosine-similarity index with a FAISS-like test surface."""

    embeddings: np.ndarray

    @property
    def ntotal(self) -> int:
        return int(self.embeddings.shape[0])

    @property
    def d(self) -> int:
        return int(self.embeddings.shape[1])

    def search(self, query_embedding: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        scores = query_embedding @ self.embeddings.T
        top_k = min(max(0, top_k), self.ntotal)

        if top_k == 0:
            empty_scores = np.empty((query_embedding.shape[0], 0), dtype="float32")
            empty_ids = np.empty((query_embedding.shape[0], 0), dtype="int64")
            return empty_scores, empty_ids

        sorted_ids = np.argsort(-scores, axis=1)[:, :top_k]
        sorted_scores = np.take_along_axis(scores, sorted_ids, axis=1)
        return sorted_scores.astype("float32"), sorted_ids.astype("int64")


def normalize_l2(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms <= 0] = 1.0
    return embeddings / norms


def embed_chunks(chunks: list[str], model_name: str = MODEL_NAME) -> np.ndarray:
    """Convert text chunks into normalized embedding vectors."""

    model = load_embedding_model()
    embeddings = model.encode(
        chunks,
        batch_size=16,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    embeddings = embeddings.astype("float32")
    return normalize_l2(embeddings).astype("float32")


def build_index(embeddings: np.ndarray) -> NumpyVectorIndex:
    """Create an in-memory cosine index for normalized embeddings."""

    return NumpyVectorIndex(embeddings.astype("float32"))


def save_chunk_records(chunks) -> None:
    """Save chunk text and metadata so search results can be displayed later."""

    records = [
        {
            "id": position,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }
        for position, chunk in enumerate(chunks)
    ]

    CHUNKS_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")


def main() -> None:
    """Build and save the RAG vector store from the dataset."""

    documents = load_documents()
    chunks = chunk_documents(documents)
    chunk_texts = [chunk.text for chunk in chunks]

    if not chunks:
        raise ValueError("No chunks were created. Check that RAGDocs contains .txt files.")

    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(chunks)} chunks")
    print(f"Embedding model: {MODEL_NAME}")

    embeddings = embed_chunks(chunk_texts)
    index = build_index(embeddings)

    INDEX_DIR.mkdir(exist_ok=True)
    np.save(INDEX_PATH, index.embeddings)
    save_chunk_records(chunks)

    print()
    print("RAG vector index built successfully")
    print(f"Index vectors: {index.ntotal}")
    print(f"Embedding dimension: {index.d}")
    print(f"Saved embeddings: {INDEX_PATH}")
    print(f"Saved chunk metadata/text: {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
