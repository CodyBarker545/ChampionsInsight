from __future__ import annotations

import unittest
from unittest.mock import patch

from rag.generate_answer import build_prompt, expand_with_neighbor_chunks, format_context


def make_record(record_id: int, filename: str, chunk_index: int, text: str) -> dict:
    return {
        "id": record_id,
        "text": text,
        "metadata": {
            "source": f"RAGDocs\\{filename}",
            "filename": filename,
            "stem": filename.removesuffix(".txt"),
            "chunk_index": chunk_index,
            "chunk_size": len(text),
        },
    }


class TestGenerationHelpers(unittest.TestCase):
    def test_format_context_includes_source_and_chunk_text(self) -> None:
        record = make_record(0, "weather.txt", 2, "Rain boosts Water-type moves.")

        context = format_context([(0.75, record)])

        self.assertIn("Source: weather.txt", context)
        self.assertIn("Chunk index: 2", context)
        self.assertIn("Rain boosts Water-type moves.", context)

    def test_build_prompt_requires_context_only_answers(self) -> None:
        messages = build_prompt("What does burn do?", "Burn context")

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("Answer using only the provided context", messages[0]["content"])
        self.assertIn("What does burn do?", messages[1]["content"])
        self.assertIn("Burn context", messages[1]["content"])

    def test_expand_with_neighbor_chunks_adds_adjacent_chunks_once(self) -> None:
        records = [
            make_record(0, "status_conditions.txt", 1, "General status context"),
            make_record(1, "status_conditions.txt", 2, "Burn details"),
            make_record(2, "status_conditions.txt", 3, "Paralysis details"),
        ]

        with patch("rag.generate_answer.load_chunk_records", return_value=records):
            expanded = expand_with_neighbor_chunks([(0.9, records[1])])

        expanded_ids = [record["id"] for _score, record in expanded]
        self.assertEqual(expanded_ids, [1, 0, 2])


if __name__ == "__main__":
    unittest.main()
