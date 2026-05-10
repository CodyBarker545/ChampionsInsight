"""Connects RAG questions to the vector pipeline or local document fallback."""

from __future__ import annotations

import re

from paths import RAG_DOCS_DIR


# Answers a question using the RAG pipeline when available.
def answer_rag_question(question: str) -> dict:
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("Question is required.")

    try:
        from rag.generate_answer import generate_answer

        answer, chunks = generate_answer(cleaned_question)
        return {
            "question": cleaned_question,
            "answer": answer or "Insufficient information available.",
            "source": format_sources(chunks),
        }
    except Exception:
        return answer_from_local_documents(cleaned_question)


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
    scored_documents = score_documents(question)
    if not scored_documents:
        return {
            "question": question,
            "answer": "Insufficient information available.",
            "source": "Local RAG Knowledge Base",
        }

    filename, text = scored_documents[0]
    answer = build_extract_answer(question, text)
    return {
        "question": question,
        "answer": answer,
        "source": filename,
    }


# Scores local text documents by keyword overlap with the question.
def score_documents(question: str) -> list[tuple[str, str]]:
    keywords = {
        word
        for word in re.findall(r"[a-z0-9]+", question.lower())
        if len(word) > 2 and word not in {"what", "does", "this", "that", "the", "and", "for"}
    }

    scored_documents: list[tuple[int, str, str]] = []
    for file_path in sorted(RAG_DOCS_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        lowered_text = text.lower()
        score = sum(lowered_text.count(keyword) for keyword in keywords)
        if score > 0:
            scored_documents.append((score, file_path.name, text))

    scored_documents.sort(key=lambda item: item[0], reverse=True)
    return [(filename, text) for _score, filename, text in scored_documents]


# Builds a short grounded answer from the best local document excerpt.
def build_extract_answer(question: str, text: str) -> str:
    keywords = [
        word
        for word in re.findall(r"[a-z0-9]+", question.lower())
        if len(word) > 2
    ]
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]

    best_paragraph = paragraphs[0] if paragraphs else text.strip()
    best_score = -1
    for paragraph in paragraphs:
        paragraph_lower = paragraph.lower()
        score = sum(paragraph_lower.count(keyword) for keyword in keywords)
        if score > best_score:
            best_score = score
            best_paragraph = paragraph

    compact_answer = re.sub(r"\s+", " ", best_paragraph).strip()
    if len(compact_answer) > 420:
        compact_answer = f"{compact_answer[:417].rstrip()}..."

    return compact_answer or "Insufficient information available."
