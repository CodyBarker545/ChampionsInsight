"""Searches the RAG vector index for chunks related to a user question."""

from __future__ import annotations

import argparse
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rag.build_faiss_index import CHUNKS_PATH, INDEX_PATH, MODEL_NAME, build_index, normalize_l2
from rag.embedding_model import load_embedding_model


STOPWORDS = {
    "a", "an", "and", "are", "be", "do", "does", "for", "how", "in", "is",
    "it", "of", "on", "or", "the", "to", "what", "when", "why",
}

TOPIC_KEYWORDS = {
    "status_conditions": {"status", "burn", "paralysis", "poison", "sleep", "freeze", "confusion", "toxic", "frostbite"},
    "held_items": {"item", "items", "held", "amulet", "boots", "vest", "sash", "scarf", "band", "specs", "policy", "berry"},
    "berry": {"berry", "berries", "rawst", "pecha", "aspear", "chesto", "cheri", "lum", "sitrus"},
    "weather": {"weather", "rain", "sun", "sand", "snow", "hail"},
    "terrain": {"terrain", "electric", "grassy", "misty", "psychic"},
    "battle_mechanics": {"mechanic", "mechanics", "damage", "priority", "speed", "critical", "accuracy", "evasion", "stab"},
    "pokemon_types": {"type", "types", "matchup", "matchups", "immune", "immunity", "resist", "weakness", "super"},
    "pokemon_abilites": {"ability", "abilities", "intimidate", "levitate", "prankster", "mold", "natural", "cure"},
}


@lru_cache(maxsize=4)
def load_chunk_records(path: Path = CHUNKS_PATH) -> list[dict]:
    """Load the saved chunk text and metadata for search results."""

    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=2)
def load_search_index(index_path: Path = INDEX_PATH):
    """Load and build the in-memory index once per process."""

    return build_index(np.load(index_path).astype("float32"))


def tokenize_query(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens for lexical reranking."""

    return [
        token
        for token in re.findall(r"[a-z0-9']+", str(text or "").lower())
        if token and token not in STOPWORDS
    ]


def infer_query_topics(query: str) -> set[str]:
    """Infer likely document topics from the user's wording."""

    tokens = set(tokenize_query(query))
    lowered_query = str(query or "").lower()
    topics: set[str] = set()

    for topic, keywords in TOPIC_KEYWORDS.items():
        if tokens & keywords:
            topics.add(topic)

    if any(phrase in lowered_query for phrase in ("what is burn", "what does burn", "burn do", "burn effect")):
        topics.add("status_conditions")
    if any(phrase in lowered_query for phrase in ("what does tailwind", "what is tailwind", "trick room", "priority", "damage calculation")):
        topics.add("battle_mechanics")
    if "item" in tokens or "holding" in tokens:
        topics.add("held_items")
    if "berry" in tokens or "berries" in tokens:
        topics.add("berry")

    return topics


def infer_record_topics(record: dict) -> set[str]:
    """Read topic tags from record metadata, with filename fallback for old indexes."""

    metadata = record.get("metadata", {})
    topic_tags = metadata.get("topic_tags")
    if isinstance(topic_tags, list) and topic_tags:
        return {str(topic).lower() for topic in topic_tags}

    filename = str(metadata.get("filename", "")).lower()
    inferred = {
        topic
        for topic in TOPIC_KEYWORDS
        if topic in filename
    }
    return inferred or {str(metadata.get("stem", "unknown")).lower()}


def lexical_overlap_score(query: str, text: str) -> float:
    """Score direct keyword overlap so exact mechanic terms can beat broad semantic matches."""

    query_tokens = tokenize_query(query)
    if not query_tokens:
        return 0.0

    lowered_text = str(text or "").lower()
    token_matches = sum(1 for token in query_tokens if token in lowered_text)
    overlap = token_matches / len(query_tokens)

    exact_phrase_bonus = 0.0
    if len(query_tokens) <= 4:
        exact_phrase = " ".join(query_tokens)
        if exact_phrase and exact_phrase in lowered_text:
            exact_phrase_bonus = 0.25

    line_anchor_bonus = 0.0
    for token in query_tokens[:3]:
        if re.search(rf"(^|\n)\s*{re.escape(token)}\b", lowered_text):
            line_anchor_bonus += 0.08

    return overlap + exact_phrase_bonus + line_anchor_bonus


def topic_alignment_score(query: str, record: dict) -> float:
    """Reward topic matches and penalize clearly mismatched sources."""

    query_topics = infer_query_topics(query)
    if not query_topics:
        return 0.0

    record_topics = infer_record_topics(record)
    topic_match_count = len(query_topics & record_topics)
    score = topic_match_count * 0.35

    lowered_query = str(query or "").lower()
    if "burn" in lowered_query and "status_conditions" in query_topics and {"held_items", "berry"} & record_topics:
        score -= 0.25
    if "item" in lowered_query and "status_conditions" in record_topics and "held_items" not in record_topics:
        score -= 0.12

    return score


def rerank_results(query: str, results: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
    """Blend vector similarity with lexical and topic-aware signals."""

    reranked: list[tuple[float, dict]] = []
    for base_score, record in results:
        lexical_score = lexical_overlap_score(query, record.get("text", ""))
        alignment_score = topic_alignment_score(query, record)
        final_score = float(base_score) + (lexical_score * 0.45) + alignment_score
        reranked.append((final_score, record))

    reranked.sort(
        key=lambda item: (
            item[0],
            lexical_overlap_score(query, item[1].get("text", "")),
            item[1].get("metadata", {}).get("filename", ""),
        ),
        reverse=True,
    )
    return reranked


def embed_query(query: str, model_name: str = MODEL_NAME) -> np.ndarray:
    """Convert a user question into the same embedding format as the chunks."""

    model = load_embedding_model()
    embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    return normalize_l2(embedding).astype("float32")


def search(query: str, top_k: int = 3) -> list[tuple[float, dict]]:
    """Search the vector index and return the most similar chunks with their scores."""

    if not INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        raise FileNotFoundError("Run python -m rag.build_faiss_index before searching.")

    index = load_search_index(INDEX_PATH)
    records = load_chunk_records()
    query_embedding = embed_query(query)

    candidate_count = max(top_k, min(len(records), max(top_k * 4, top_k + 8)))
    scores, ids = index.search(query_embedding, candidate_count)
    results: list[tuple[float, dict]] = []

    for score, record_id in zip(scores[0], ids[0]):
        if record_id == -1:
            continue
        results.append((float(score), records[int(record_id)]))

    return rerank_results(query, results)[:top_k]


def main() -> None:
    """Run a retrieval search from the command line."""

    parser = argparse.ArgumentParser(description="Search the Pokemon RAG vector index.")
    parser.add_argument("query", help="Question or search query to retrieve chunks for.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve.")
    args = parser.parse_args()

    results = search(args.query, args.top_k)

    print(f"Query: {args.query}")
    print(f"Top {len(results)} retrieved chunks")
    print("=" * 40)

    for rank, (score, record) in enumerate(results, start=1):
        metadata = record["metadata"]
        print(
            f"{rank}. Score: {score:.4f} | "
            f"Source: {metadata['filename']} | "
            f"Chunk: {metadata['chunk_index']}"
        )
        print(record["text"][:900])
        if len(record["text"]) > 900:
            print("...")
        print()


if __name__ == "__main__":
    main()
