from __future__ import annotations

import unittest

from rag.preprocess import clean_text, chunk_documents, load_documents, split_text
from backend.rag.tests.unit.helpers import workspace_temp_dir


class TestPreprocess(unittest.TestCase):
    def test_clean_text_removes_citation_artifacts_and_extra_whitespace(self) -> None:
        raw_text = "Hello   world\r\n\r\n\r\nText :contentReference[oaicite:1]{index=1}"

        cleaned_text = clean_text(raw_text)

        self.assertEqual(cleaned_text, "Hello world\n\nText")

    def test_load_documents_attaches_source_metadata(self) -> None:
        with workspace_temp_dir() as data_dir:
            (data_dir / "pokemon_types.txt").write_text("Type guide text", encoding="utf-8")

            documents = load_documents(data_dir)

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].metadata["filename"], "pokemon_types.txt")
        self.assertEqual(documents[0].metadata["stem"], "pokemon_types")
        self.assertTrue(documents[0].metadata["source"].endswith("pokemon_types.txt"))

    def test_split_text_uses_overlap(self) -> None:
        text = "A" * 60 + " " + "B" * 60 + " " + "C" * 60

        chunks = split_text(text, chunk_size=80, chunk_overlap=20)

        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(max(len(chunk) for chunk in chunks), 80)

    def test_chunk_documents_adds_chunk_metadata(self) -> None:
        with workspace_temp_dir() as data_dir:
            (data_dir / "status_conditions.txt").write_text("Burn details. " * 80, encoding="utf-8")
            documents = load_documents(data_dir)

        chunks = chunk_documents(documents, chunk_size=120, chunk_overlap=20)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["filename"], "status_conditions.txt")
        self.assertEqual(chunks[0].metadata["chunk_index"], 1)
        self.assertEqual(chunks[0].metadata["chunk_size"], len(chunks[0].text))


if __name__ == "__main__":
    unittest.main()
