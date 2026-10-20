"""Tests the query-aware local RAG fallback scoring."""

from __future__ import annotations

from services import rag_service
from backend.rag.tests.unit.helpers import workspace_temp_dir


def test_named_effect_lookup_precedes_vector_search(monkeypatch):
    from rag import search_faiss

    def unexpected(*_args, **_kwargs):
        raise AssertionError("Exact mechanics lookup should not load embeddings")

    monkeypatch.setattr(search_faiss, "search", unexpected)
    context = rag_service.retrieve_rag_context("what does para do")
    assert context["retrievalMode"] == "mechanic_sections"
    assert context["source"] == "status_conditions.txt"
    answer = rag_service.answer_from_retrieved_context("what does para do", context)
    assert "Paralysis" in answer
    assert "25%" in answer
    assert "held" not in answer.lower()


def test_unrelated_vector_evidence_is_not_an_answer():
    context = {"chunks": [(0.9, {"text": "Some items are consumed after activation."})]}
    assert rag_service.answer_from_retrieved_context("paralysis", context) == "Insufficient information available."


def test_alias_expansion_is_stable_and_uses_word_boundaries():
    query = "choice specs and scarf on Parasect"
    normalized = rag_service.normalize_battle_query(query)
    assert normalized == "choice specs and choice scarf on parasect"
    assert rag_service.normalize_battle_query(normalized) == normalized


def test_score_documents_prefers_status_doc_for_burn_question(monkeypatch):
    with workspace_temp_dir() as temp_dir:
        (temp_dir / "status_conditions.txt").write_text(
            "Burn deals damage each turn and halves physical Attack.",
            encoding="utf-8",
        )
        (temp_dir / "berry.txt").write_text(
            "Rawst Berry cures burn when held.",
            encoding="utf-8",
        )

        monkeypatch.setattr(rag_service, "RAG_DOCS_DIR", temp_dir)

        ranked = rag_service.score_documents("What does burn do?")

        assert ranked
        assert ranked[0][0] == "status_conditions.txt"


def test_answer_from_local_documents_extracts_burn_section(monkeypatch):
    with workspace_temp_dir() as temp_dir:
        (temp_dir / "status_conditions.txt").write_text(
            "\n".join(
                [
                    "Status Conditions Guide",
                    "",
                    "Burn:",
                    "- Deals damage each turn (1/16 of max HP)",
                    "- Reduces physical Attack stat by 50%",
                    "",
                    "Paralysis:",
                    "- Reduces Speed",
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(rag_service, "RAG_DOCS_DIR", temp_dir)

        result = rag_service.answer_from_local_documents(
            "what does the burn status do"
        )

        assert "Burn:" in result["answer"]
        assert "1/16 of max HP" in result["answer"]
        assert "Attack stat by 50%" in result["answer"]


def test_answer_rag_question_uses_shared_retrieved_context(monkeypatch):
    monkeypatch.setattr(
        rag_service,
        "retrieve_rag_context",
        lambda question: {
            "question": question,
            "chunks": [
                (
                    1.0,
                    {
                        "text": "Tailwind doubles Speed for four turns.",
                        "metadata": {"filename": "battle_mechanics.txt", "chunk_index": 1},
                    },
                )
            ],
            "source": "battle_mechanics.txt",
            "retrievalMode": "vector_store",
            "notes": [],
        },
    )

    result = rag_service.answer_rag_question("what does tailwind do")

    assert result["answer"] == "Tailwind doubles Speed for four turns."
    assert result["source"] == "battle_mechanics.txt"
    assert result["retrievalMode"] == "vector_store"
