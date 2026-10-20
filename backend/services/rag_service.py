"""Connects RAG questions to the vector pipeline or local document fallback."""

from __future__ import annotations

import re
from typing import Any

from paths import RAG_DOCS_DIR
from rag.search_faiss import lexical_overlap_score, load_chunk_records, topic_alignment_score, tokenize_query


MECHANIC_ALIASES = {
    "para": "paralysis", "paralyzed": "paralysis", "paralysed": "paralysis",
    "par": "paralysis", "brn": "burn", "slp": "sleep", "psn": "poison",
    "frz": "freeze", "cb": "choice band", "specs": "choice specs",
    "scarf": "choice scarf", "av": "assault vest", "tr": "trick room",
}


def normalize_battle_query(question: str) -> str:
    """Expand common battle shorthand at word boundaries before retrieval."""
    text = str(question or "").lower().replace("-", " ")
    return re.sub(
        r"\b(" + "|".join(map(re.escape, MECHANIC_ALIASES)) + r")\b",
        lambda match: (
            match.group() if match.group() in {"specs", "scarf"}
            and text[:match.start()].rstrip().endswith("choice")
            else MECHANIC_ALIASES[match.group()]
        ), text,
    )


def mechanic_sections(question: str) -> list[tuple[float, dict[str, Any]]]:
    """Find named mechanics in heading-and-bullet sections, without a model load.

    Only headings followed by actual effect bullets qualify. General prose and
    mentions in other items' descriptions cannot become an entity definition.
    """
    query = normalize_battle_query(question)
    matches = []
    for path in sorted(RAG_DOCS_DIR.glob("*.txt")):
        if path.stem == "evolution_items":
            continue
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            heading = line.strip().rstrip(":")
            name = normalize_battle_query(heading)
            names = [name, normalize_battle_query(re.sub(r"\s*\([^)]*\)", "", heading))]
            names.extend(normalize_battle_query(term) for term in re.findall(r"\(([^)]+)\)", heading)
                         if term.lower() not in {"volatile", "non-volatile"})
            if not name or len(name) > 65 or not any(
                candidate and re.search(rf"(?<!\w){re.escape(candidate)}(?!\w)", query)
                for candidate in names
            ):
                continue
            bullets = []
            for following in lines[index + 1:]:
                stripped = following.strip()
                if not stripped or re.fullmatch(r"-{5,}", stripped):
                    if bullets:
                        break
                    continue
                if not stripped.startswith("- "):
                    break
                bullets.append(stripped)
            if bullets:
                matches.append((float(len(name.split())), {
                    "text": heading + ":\n" + "\n".join(bullets),
                    "metadata": {"filename": path.name, "section": heading, "chunk_index": index},
                }))
    # Longer names beat overlapping terms such as Poison / Badly Poisoned.
    return sorted(matches, key=lambda item: item[0], reverse=True)


# Answers a question using the RAG pipeline when available.
def answer_rag_question(question: str) -> dict:
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("Question is required.")

    context = retrieve_rag_context(cleaned_question)
    answer = answer_from_retrieved_context(cleaned_question, context)
    return {
        "question": cleaned_question,
        "answer": answer,
        "source": context.get("source", "Local RAG Knowledge Base"),
        "retrievalMode": context.get("retrievalMode", "local_documents"),
    }


def retrieve_rag_context(question: str, top_k: int = 3, max_context_chunks: int = 4) -> dict[str, Any]:
    """Fetch RAG evidence from the shared vector index, with one local-doc fallback."""

    cleaned_question = str(question or "").strip()
    if not cleaned_question:
        raise ValueError("Question is required.")

    sections = mechanic_sections(cleaned_question)
    if sections:
        return {
            "question": cleaned_question, "chunks": sections[:max_context_chunks],
            "source": format_sources(sections[:max_context_chunks]),
            "retrievalMode": "mechanic_sections", "notes": [],
        }

    cleaned_question = normalize_battle_query(cleaned_question)
    try:
        from rag.search_faiss import search

        chunks = expand_with_neighbor_chunks(search(cleaned_question, top_k=top_k))
        chunks = chunks[:max_context_chunks]
        if chunks:
            return {
                "question": cleaned_question,
                "chunks": chunks,
                "source": format_sources(chunks),
                "retrievalMode": "vector_store",
                "notes": [],
            }
    except Exception:
        pass

    local_chunks = local_document_chunks(cleaned_question, limit=max_context_chunks)
    return {
        "question": cleaned_question,
        "chunks": local_chunks,
        "source": format_sources(local_chunks),
        "retrievalMode": "local_documents",
        "notes": [] if local_chunks else ["No matching local RAG documents were found."],
    }


# Formats retrieved chunk sources for the frontend.
def format_sources(chunks) -> str:
    filenames = []
    for _score, chunk in chunks:
        filename = chunk.get("metadata", {}).get("filename")
        if filename and filename not in filenames:
            filenames.append(filename)

    return ", ".join(filenames) if filenames else "Local RAG Knowledge Base"


# Answers from local knowledge files when vector search is unavailable.
def answer_from_local_documents(question: str) -> dict:
    local_chunks = local_document_chunks(question)
    context = {
        "question": question,
        "chunks": local_chunks,
        "source": format_sources(local_chunks),
        "retrievalMode": "local_documents",
    }
    answer = answer_from_retrieved_context(question, context)
    if answer == "Insufficient information available.":
        return {
            "question": question,
            "answer": answer,
            "source": "Local RAG Knowledge Base",
        }

    return {
        "question": question,
        "answer": answer,
        "source": context.get("source", "Local RAG Knowledge Base"),
    }


def answer_from_retrieved_context(question: str, context: dict[str, Any]) -> str:
    """Build a concise answer from already-retrieved chunks."""

    question = normalize_battle_query(question)
    chunks = context.get("chunks", [])
    if not chunks:
        return "Insufficient information available."

    direct_answer = answer_evolution_question_from_context(question, chunks)
    if direct_answer:
        return direct_answer

    best_score, best_record = max(chunks, key=lambda item: item[0])
    best_text = best_record.get("text", "")
    if not best_text or lexical_overlap_score(question, best_text) <= 0:
        return "Insufficient information available."

    return build_extract_answer(question, best_text)


def expand_with_neighbor_chunks(results: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
    """Add adjacent chunks from the same source without importing generation code."""

    records = load_chunk_records()
    records_by_source_and_index = {
        (record["metadata"]["filename"], record["metadata"]["chunk_index"]): record
        for record in records
    }

    expanded_results: list[tuple[float, dict]] = []
    seen_ids: set[int] = set()

    for score, record in results:
        candidates = [record]
        metadata = record.get("metadata", {})
        for neighbor_index in (
            metadata.get("chunk_index", 0) - 1,
            metadata.get("chunk_index", 0) + 1,
        ):
            neighbor = records_by_source_and_index.get((metadata.get("filename"), neighbor_index))
            if neighbor is not None:
                candidates.append(neighbor)

        for candidate in candidates:
            candidate_id = int(candidate.get("id", len(seen_ids)))
            if candidate_id in seen_ids:
                continue
            candidate_score = score if candidate_id == record.get("id") else score * 0.95
            expanded_results.append((candidate_score, candidate))
            seen_ids.add(candidate_id)

    return expanded_results


def parse_evolution_line(line: str) -> tuple[str, str, str] | None:
    cleaned_line = line.strip().lstrip("-").strip()
    cleaned_line = cleaned_line.replace("Ã¢â€ â€™", "->").replace("â†’", "->")

    match = re.match(r"(.+?)\s+\+\s+(.+?)\s+->\s+(.+)", cleaned_line)
    if match is None:
        return None

    pokemon, item, evolution = [part.strip() for part in match.groups()]
    return pokemon, item, evolution


def infer_evolution_condition(context_before_line: str, item: str) -> str:
    lower_context = context_before_line.lower()
    lower_item = item.lower()

    if "trade evolutions with items" in lower_context:
        return f"by being traded while holding {item}"
    if "level-up while holding item" in lower_context or "night" in lower_item or "day" in lower_item:
        return f"by leveling up while holding {item}"
    return f"using or meeting the condition for {item}"


def answer_evolution_question_from_context(
    question: str,
    results: list[tuple[float, dict]],
) -> str | None:
    question_lower = question.lower()
    if "evolve" not in question_lower and "evolution" not in question_lower:
        return None

    for _score, record in results:
        context_so_far = ""
        for line in record.get("text", "").splitlines():
            parsed_line = parse_evolution_line(line)
            if parsed_line is None:
                context_so_far += f"\n{line}"
                continue

            pokemon, item, evolution = parsed_line
            if pokemon.lower() not in question_lower:
                context_so_far += f"\n{line}"
                continue

            condition = infer_evolution_condition(context_so_far, item)
            return f"{pokemon} evolves into {evolution} {condition}."

    return None


# Scores local text documents by keyword overlap with the question.
def score_documents(question: str) -> list[tuple[str, str]]:
    query_tokens = tokenize_query(question)
    scored_documents: list[tuple[float, str, str]] = []
    for file_path in sorted(RAG_DOCS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        lexical_score = lexical_overlap_score(question, text)
        pseudo_record = {
            "text": text,
            "metadata": {
                "filename": file_path.name,
                "stem": file_path.stem,
            },
        }
        score = lexical_score + topic_alignment_score(question, pseudo_record)
        if len(query_tokens) <= 3 and query_tokens:
            score += 0.05 * sum(1 for token in query_tokens if re.search(rf"(^|\n)\s*{re.escape(token)}\b", text.lower()))
        if score > 0:
            scored_documents.append((score, file_path.name, text))

    scored_documents.sort(key=lambda item: item[0], reverse=True)
    return [(filename, text) for _score, filename, text in scored_documents]


def local_document_chunks(question: str, limit: int = 4) -> list[tuple[float, dict[str, Any]]]:
    """Return local text documents in the same tuple shape as vector retrieval."""

    sections = mechanic_sections(question)
    if sections:
        return sections[:limit]
    question = normalize_battle_query(question)
    query_tokens = tokenize_query(question)
    chunks: list[tuple[float, dict[str, Any]]] = []
    for file_path in sorted(RAG_DOCS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        lexical_score = lexical_overlap_score(question, text)
        pseudo_record = {
            "id": len(chunks),
            "text": text,
            "metadata": {
                "source": str(file_path),
                "filename": file_path.name,
                "stem": file_path.stem,
                "chunk_index": 1,
            },
        }
        score = lexical_score + topic_alignment_score(question, pseudo_record)
        if len(query_tokens) <= 3 and query_tokens:
            score += 0.05 * sum(1 for token in query_tokens if re.search(rf"(^|\n)\s*{re.escape(token)}\b", text.lower()))
        if score > 0:
            chunks.append((score, pseudo_record))

    chunks.sort(key=lambda item: item[0], reverse=True)
    return chunks[:limit]


# Builds a short grounded answer from the best local document excerpt.
def build_extract_answer(question: str, text: str) -> str:
    keywords = tokenize_query(question)
    targeted_answer = build_targeted_extract_answer(question, text, keywords)
    if targeted_answer:
        return targeted_answer

    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]

    best_paragraph = paragraphs[0] if paragraphs else text.strip()
    best_score = -1
    for paragraph in paragraphs:
        score = lexical_overlap_score(question, paragraph)
        if keywords and len(keywords) <= 3:
            paragraph_lower = paragraph.lower()
            if all(keyword in paragraph_lower for keyword in keywords):
                score += 0.2
        if score > best_score:
            best_score = score
            best_paragraph = paragraph

    compact_answer = re.sub(r"\s+", " ", best_paragraph).strip()
    if len(compact_answer) > 420:
        compact_answer = f"{compact_answer[:417].rstrip()}..."

    return compact_answer or "Insufficient information available."


def build_targeted_extract_answer(question: str, text: str, keywords: list[str] | None = None) -> str | None:
    """Extract a focused section when the document uses heading-plus-bullets formatting."""

    keywords = keywords if keywords is not None else tokenize_query(question)
    if not keywords:
        return None

    normalized_lines = [line.strip() for line in text.splitlines()]

    for index, line in enumerate(normalized_lines):
        lowered_line = line.lower()
        if not lowered_line:
            continue

        is_anchor = False
        if all(keyword in lowered_line for keyword in keywords):
            is_anchor = True
        elif len(keywords) == 1 and re.match(rf"^{re.escape(keywords[0])}\s*:", lowered_line):
            is_anchor = True
        elif any(re.match(rf"^{re.escape(keyword)}\s*:", lowered_line) for keyword in keywords):
            is_anchor = True

        if not is_anchor:
            continue

        collected_lines = [line]
        for following_line in normalized_lines[index + 1:]:
            stripped = following_line.strip()
            lowered_following = stripped.lower()
            if not stripped:
                if len(collected_lines) > 1:
                    break
                continue
            if re.match(r"^-{5,}$", stripped):
                continue
            if re.match(r"^\d+\.", stripped):
                break
            if stripped.endswith(":") and not lowered_following.startswith("-"):
                break
            if ":" in stripped and not stripped.startswith("-") and collected_lines and len(collected_lines) > 1:
                break
            collected_lines.append(stripped)
            if len(collected_lines) >= 5:
                break

        compact_answer = re.sub(r"\s+", " ", " ".join(collected_lines)).strip()
        compact_answer = compact_answer.replace(" :", ":")
        if len(compact_answer) > 420:
            compact_answer = f"{compact_answer[:417].rstrip()}..."
        if compact_answer:
            return compact_answer

    return None
