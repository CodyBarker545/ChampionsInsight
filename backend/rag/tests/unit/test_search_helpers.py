from __future__ import annotations

import json
import unittest

from rag.search_faiss import load_chunk_records
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


if __name__ == "__main__":
    unittest.main()
