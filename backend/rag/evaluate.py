"""Runs evaluation checks for the RAG retrieval and answer pipeline."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rag.generate_answer import generate_answer


@dataclass
class EvaluationQuestion:
    """One test question and the keywords used for simple rule-based grading."""

    question: str
    expected_answer: str
    required_keywords: list[str]


EVALUATION_QUESTIONS = [
    EvaluationQuestion(
        question="What happens when a Pokemon is burned?",
        expected_answer="Burn deals damage each turn equal to 1/16 of max HP and reduces physical Attack by 50%.",
        required_keywords=["1/16", "attack", "50"],
    ),
    EvaluationQuestion(
        question="What does Stealth Rock do?",
        expected_answer="Stealth Rock damages Pokemon when they switch in, with damage based on weakness or resistance to Rock.",
        required_keywords=["switch", "rock", "damage"],
    ),
    EvaluationQuestion(
        question="What is STAB in Pokemon battles?",
        expected_answer="STAB increases damage when a Pokemon uses a move that matches its own type, normally by 1.5x.",
        required_keywords=["type", "damage", "1.5"],
    ),
    EvaluationQuestion(
        question="What does Heavy-Duty Boots do?",
        expected_answer="Heavy-Duty Boots prevents damage and effects from entry hazards.",
        required_keywords=["entry hazards", "prevents"],
    ),
    EvaluationQuestion(
        question="What happens during rain weather?",
        expected_answer="Rain boosts Water-type moves, weakens Fire-type moves, and supports rain-based abilities.",
        required_keywords=["water", "fire", "rain"],
    ),
]


def grade_answer(answer: str, required_keywords: list[str]) -> str:
    """Grade an answer as Correct, Partial, or Incorrect using required keywords."""

    answer_lower = answer.lower()
    matched_keywords = [
        keyword for keyword in required_keywords if keyword.lower() in answer_lower
    ]

    if len(matched_keywords) == len(required_keywords):
        return "Correct"
    if matched_keywords:
        return "Partial"
    return "Incorrect"


def print_retrieved_summary(results: list[tuple[float, dict]]) -> None:
    """Print a short list of retrieved chunks for the evaluation report."""

    for rank, (score, record) in enumerate(results, start=1):
        metadata = record["metadata"]
        print(
            f"  {rank}. {metadata['filename']} | "
            f"Chunk {metadata['chunk_index']} | Score {score:.4f}"
        )


def main() -> None:
    """Run the evaluation questions and print the grading summary."""

    print("RAG Evaluation")
    print("=" * 60)
    print(f"Questions tested: {len(EVALUATION_QUESTIONS)}")
    print()

    accuracy_counts = {"Correct": 0, "Partial": 0, "Incorrect": 0}

    for index, eval_question in enumerate(EVALUATION_QUESTIONS, start=1):
        answer, retrieved_chunks = generate_answer(
            eval_question.question,
            top_k=3,
            max_context_chunks=4,
            max_new_tokens=140,
        )
        accuracy = grade_answer(answer, eval_question.required_keywords)
        accuracy_counts[accuracy] += 1

        print(f"Question {index}: {eval_question.question}")
        print("Retrieved chunks:")
        print_retrieved_summary(retrieved_chunks)
        print(f"Expected answer: {eval_question.expected_answer}")
        print(f"Final answer: {answer}")
        print(f"Accuracy: {accuracy}")
        print("-" * 60)

    print("Summary")
    print("=" * 60)
    for label, count in accuracy_counts.items():
        print(f"{label}: {count}")


if __name__ == "__main__":
    main()
