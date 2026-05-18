"""Print explainable tournament-based move/item/ability/nature probabilities."""

from __future__ import annotations

import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
for path in (PROJECT_ROOT, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from services.set_probability_service import calculate_basic_probabilities


POKEMON_TO_TEST = [
    "Sneasler",
    "Garchomp",
    "Tyranitar",
    "Articuno",
    "DefinitelyNotAPokemon",
]


def main() -> None:
    for pokemon_name in POKEMON_TO_TEST:
        result = calculate_basic_probabilities(pokemon_name, limit=10)
        print(f"== {pokemon_name} ==")
        print(json.dumps(result, indent=2))
        print()


if __name__ == "__main__":
    main()
