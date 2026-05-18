"""CLI entrypoint for building the Pokemon embedding index from slot datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.build_pokemon_embedding_index import (
    DEFAULT_REAL_DIR,
    DEFAULT_SYNTHETIC_DIR,
    build_pokemon_embedding_index,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synthetic-dir",
        default=str(DEFAULT_SYNTHETIC_DIR),
        help="Synthetic slot dataset root.",
    )
    parser.add_argument(
        "--real-dir",
        default=str(DEFAULT_REAL_DIR),
        help="Real slot dataset root.",
    )
    parser.add_argument(
        "--synthetic-samples-per-variant",
        type=int,
        default=250,
        help="Maximum number of normal and shiny synthetic images to sample per class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Seed for deterministic synthetic sampling.",
    )
    args = parser.parse_args()

    build_pokemon_embedding_index(
        synthetic_dir=Path(args.synthetic_dir),
        real_dir=Path(args.real_dir),
        synthetic_samples_per_variant=int(args.synthetic_samples_per_variant),
        seed=int(args.seed),
    )


if __name__ == "__main__":
    main()
