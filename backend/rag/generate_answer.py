"""Retrieves RAG context and generates grounded answers."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from rag.generation_model import GENERATION_MODEL_NAME, load_generation_model
from rag.search_faiss import load_chunk_records, search


def parse_evolution_line(line: str) -> tuple[str, str, str] | None:
    """Read an evolution line like 'Magmar + Magmarizer -> Magmortar'."""

    cleaned_line = line.strip().lstrip("-").strip()
    cleaned_line = cleaned_line.replace("â†’", "->").replace("→", "->")

    match = re.match(r"(.+?)\s+\+\s+(.+?)\s+->\s+(.+)", cleaned_line)
    if match is None:
        return None

    pokemon, item, evolution = [part.strip() for part in match.groups()]
    return pokemon, item, evolution


def infer_evolution_condition(context_before_line: str, item: str) -> str:
    """Use nearby section text to explain whether evolution needs trade or level up."""

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
    """Answer evolution questions directly when an exact evolution line is retrieved."""

    question_lower = question.lower()
    if "evolve" not in question_lower and "evolution" not in question_lower:
        return None

    for _score, record in results:
        context_so_far = ""
        for line in record["text"].splitlines():
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


def expand_with_neighbor_chunks(results: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
    """Add nearby chunks from the same file so split context is not lost."""

    records = load_chunk_records()
    records_by_source_and_index = {
        (record["metadata"]["filename"], record["metadata"]["chunk_index"]): record
        for record in records
    }

    expanded_results: list[tuple[float, dict]] = []
    seen_ids: set[int] = set()

    for score, record in results:
        candidates = [record]
        metadata = record["metadata"]

        # Add neighbors to avoid losing details split across chunk boundaries.
        for neighbor_index in (
            metadata["chunk_index"] - 1,
            metadata["chunk_index"] + 1,
        ):
            neighbor = records_by_source_and_index.get((metadata["filename"], neighbor_index))
            if neighbor is not None:
                candidates.append(neighbor)

        for candidate in candidates:
            if candidate["id"] in seen_ids:
                continue
            candidate_score = score if candidate["id"] == record["id"] else score * 0.95
            expanded_results.append((candidate_score, candidate))
            seen_ids.add(candidate["id"])

    return expanded_results


def format_context(results: list[tuple[float, dict]]) -> str:
    """Turn retrieved chunks into one context string for the LLM prompt."""

    context_blocks: list[str] = []

    for rank, (_score, record) in enumerate(results, start=1):
        metadata = record["metadata"]
        context_blocks.append(
            f"[Chunk {rank} | Source: {metadata['filename']} | "
            f"Chunk index: {metadata['chunk_index']}]\n{record['text']}"
        )

    return "\n\n".join(context_blocks)


def build_prompt(question: str, context: str) -> list[dict[str, str]]:
    """Build the instruction prompt that tells the LLM to use only retrieved context."""

    system_prompt = (
        "You are a Pokemon RAG assistant. Answer using only the provided context. "
        "If the context does not contain the answer, say: "
        "'I do not know based on the provided context.' "
        "Do not add outside knowledge. Prefer exact details from chunks that mention "
        "the same concept as the question."
    )
    user_prompt = (
        "Context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer:"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def generate_answer(
    question: str,
    top_k: int = 3,
    max_new_tokens: int = 180,
    max_context_chunks: int = 4,
) -> tuple[str, list[tuple[float, dict]]]:
    """Run retrieval and generation for one user question."""

    retrieved_chunks = search(question, top_k=top_k)
    retrieved_chunks = expand_with_neighbor_chunks(retrieved_chunks)
    retrieved_chunks = retrieved_chunks[:max_context_chunks]

    direct_answer = answer_evolution_question_from_context(question, retrieved_chunks)
    if direct_answer is not None:
        return direct_answer, retrieved_chunks

    context = format_context(retrieved_chunks)
    messages = build_prompt(question, context)

    tokenizer, model = load_generation_model()
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return answer, retrieved_chunks


def print_retrieved_chunks(results: list[tuple[float, dict]]) -> None:
    """Print retrieved chunks so the RAG evidence can be reviewed."""

    print("Retrieved chunks")
    print("=" * 40)

    for rank, (score, record) in enumerate(results, start=1):
        metadata = record["metadata"]
        print(
            f"{rank}. Score: {score:.4f} | "
            f"Source: {metadata['filename']} | "
            f"Chunk: {metadata['chunk_index']}"
        )
        print(record["text"][:700])
        if len(record["text"]) > 700:
            print("...")
        print()


def main() -> None:
    """Run the full RAG question-answering pipeline from the command line."""

    parser = argparse.ArgumentParser(description="Ask a question using the Pokemon RAG pipeline.")
    parser.add_argument("question", help="Question to answer using retrieved RAG context.")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--max-context-chunks",
        type=int,
        default=4,
        help="Maximum retrieved/neighbor chunks to pass into the LLM.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=180, help="Maximum answer length.")
    args = parser.parse_args()

    print(f"Question: {args.question}")
    print(f"Generation model: {GENERATION_MODEL_NAME}")
    print()

    answer, retrieved_chunks = generate_answer(
        args.question,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        max_context_chunks=args.max_context_chunks,
    )

    print_retrieved_chunks(retrieved_chunks)
    print("Final answer")
    print("=" * 40)
    print(answer)


if __name__ == "__main__":
    main()
