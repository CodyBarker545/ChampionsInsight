"""Print conditional set probabilities and likely-set predictions."""

from __future__ import annotations

import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
for path in (PROJECT_ROOT, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from services.set_probability_service import (
    calculate_conditional_probabilities,
    predict_likely_set,
)


TEST_CASES = [
    {"pokemon": "Sneasler"},
    {"pokemon": "Sneasler", "teammates": ["Indeedee"]},
    {"pokemon": "Tyranitar"},
    {"pokemon": "Tyranitar", "teammates": ["Garchomp"]},
    {"pokemon": "Sneasler", "teammates": ["DefinitelyNotAPokemon"]},
]


def main() -> None:
    for case in TEST_CASES:
        print(f"== {case['pokemon']} | teammates={case.get('teammates')} | archetype={case.get('archetype')} ==")
        probabilities = calculate_conditional_probabilities(
            case["pokemon"],
            teammates=case.get("teammates"),
            archetype=case.get("archetype"),
            limit=10,
        )
        likely_set = predict_likely_set(
            case["pokemon"],
            teammates=case.get("teammates"),
            archetype=case.get("archetype"),
        )
        print("-- Conditional probabilities --")
        print(json.dumps(probabilities, indent=2))
        print("-- Likely set --")
        print(json.dumps(likely_set, indent=2))
        print()


if __name__ == "__main__":
    main()
