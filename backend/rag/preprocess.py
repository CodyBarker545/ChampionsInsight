"""Loads and chunks RAG knowledge documents for indexing."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "RAGDocs"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
SAMPLE_CHUNKS_TO_PRINT = 5


@dataclass
class Document:
    """A full source document and the metadata that identifies where it came from."""

    text: str
    metadata: dict[str, str]


@dataclass
class Chunk:
    """A smaller piece of a document that can be embedded and retrieved."""

    text: str
    metadata: dict[str, str | int]


def clean_text(text: str) -> str:
    """Clean copied text so it is easier to chunk and search."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r":contentReference\[[^\]]+\]\{[^}]+\}", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    """Load every text file from the dataset folder and attach source metadata."""

    documents: list[Document] = []

    for file_path in sorted(data_dir.glob("*.txt")):
        text = clean_text(file_path.read_text(encoding="utf-8"))
        documents.append(
            Document(
                text=text,
                metadata={
                    "source": str(file_path),
                    "filename": file_path.name,
                    "stem": file_path.stem,
                },
            )
        )

    return documents


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split one document into overlapping chunks for retrieval."""

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Prefer ending chunks at a natural boundary when one is nearby.
        if end < len(text):
            boundary = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
            if boundary > start + int(chunk_size * 0.6):
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = max(0, end - chunk_overlap)
        if next_start > 0 and not text[next_start].isspace():
            next_space = text.find(" ", next_start, min(next_start + 40, len(text)))
            if next_space != -1:
                next_start = next_space + 1

        start = next_start

    return chunks


def chunk_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Chunk all loaded documents and keep metadata with each chunk."""

    chunks: list[Chunk] = []

    for document in documents:
        text_chunks = split_text(document.text, chunk_size, chunk_overlap)
        for index, text in enumerate(text_chunks, start=1):
            chunks.append(
                Chunk(
                    text=text,
                    metadata={
                        **document.metadata,
                        "chunk_index": index,
                        "chunk_size": len(text),
                    },
                )
            )

    return chunks


def print_summary(documents: list[Document], chunks: list[Chunk]) -> None:
    """Print a quick overview of how many documents and chunks were created."""

    print("RAG preprocessing summary")
    print("=" * 28)
    print(f"Documents loaded: {len(documents)}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Chunk overlap: {CHUNK_OVERLAP} characters")
    print()

    print("Documents")
    print("-" * 28)
    for document in documents:
        print(
            f"- {document.metadata['filename']}: "
            f"{len(document.text)} cleaned characters"
        )
    print()


def print_sample_chunks(chunks: list[Chunk], sample_count: int = SAMPLE_CHUNKS_TO_PRINT) -> None:
    """Print a few chunks so the preprocessing output can be checked manually."""

    print(f"Sample chunks first {min(sample_count, len(chunks))}")
    print("-" * 28)

    for chunk in chunks[:sample_count]:
        print(
            f"Source: {chunk.metadata['filename']} | "
            f"Chunk: {chunk.metadata['chunk_index']} | "
            f"Length: {chunk.metadata['chunk_size']}"
        )
        print(chunk.text[:700])
        if len(chunk.text) > 700:
            print("...")
        print()


def main() -> None:
    """Run preprocessing from the command line."""

    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Could not find dataset folder: {DATA_DIR}")

    documents = load_documents()
    chunks = chunk_documents(documents)

    print_summary(documents, chunks)
    print_sample_chunks(chunks)


if __name__ == "__main__":
    main()
