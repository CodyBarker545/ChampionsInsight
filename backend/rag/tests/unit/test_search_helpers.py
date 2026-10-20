from __future__ import annotations

import json
import unittest

from rag.search_faiss import load_chunk_records, rerank_results, tokenize_query
from backend.rag.tests.unit.helpers import workspace_temp_dir


class TestSearchHelpers(unittest.TestCase):
    def test_load_chunk_records_reads_json_records(self) -> None:
        expected_records = [
            {
                "id": 0,
                "text": "Sample chunk",
                "metadata": {"filename": "sample.txt", "chunk_index": 1},
            }
        ]

        with workspace_temp_dir() as temp_dir:
            records_path = temp_dir / "chunks.json"
            records_path.write_text(json.dumps(expected_records), encoding="utf-8")

            records = load_chunk_records(records_path)

        self.assertEqual(records, expected_records)

    def test_tokenize_query_drops_stopwords(self) -> None:
        self.assertEqual(tokenize_query("What does burn do?"), ["burn"])

    def test_rerank_results_prefers_status_document_for_burn_question(self) -> None:
        status_record = {
            "id": 0,
            "text": "Burn: Deals damage each turn and cuts Attack.",
            "metadata": {
                "filename": "status_conditions.txt",
                "stem": "status_conditions",
                "topic_tags": ["status_conditions"],
            },
        }
        berry_record = {
            "id": 1,
            "text": "Rawst Berry cures burn as a held item.",
            "metadata": {
                "filename": "berry.txt",
                "stem": "berry",
                "topic_tags": ["berry", "held_items"],
            },
        }

        reranked = rerank_results(
            "What does burn do?",
            [(0.65, berry_record), (0.61, status_record)],
        )

        self.assertEqual(reranked[0][1]["metadata"]["filename"], "status_conditions.txt")


if __name__ == "__main__":
    unittest.main()
