from __future__ import annotations

import unittest

import numpy as np

from rag.build_faiss_index import build_index


class TestFaissIndex(unittest.TestCase):
    def test_build_index_adds_all_embeddings(self) -> None:
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype="float32",
        )

        index = build_index(embeddings)

        self.assertEqual(index.ntotal, 3)
        self.assertEqual(index.d, 3)


if __name__ == "__main__":
    unittest.main()
