"""Smoke-test Pokemon meta intelligence output for a few representative names."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.meta_trend_service import get_pokemon_meta_intelligence
from services.pokedex_service import get_pokedex_entry_by_name
from services.set_probability_service import load_pokemon_set_records


def find_low_sample_pokemon() -> str | None:
    counts = Counter(record.get("pokemon") for record in load_pokemon_set_records() if record.get("pokemon"))
    for pokemon_name, count in counts.most_common()[::-1]:
        if 1 <= count <= 4:
            entry = get_pokedex_entry_by_name(pokemon_name)
            if entry is not None:
                return entry.get("name") or pokemon_name
    return None


def print_result(name: str) -> None:
    print(f"\n=== {name} ===")
    try:
        result = get_pokemon_meta_intelligence(name)
    except Exception as error:  # noqa: BLE001 - smoke-test script
        print(f"ERROR: {error}")
        return

    preview = {
        "pokemon": result.get("pokemon"),
        "usageTrend": result.get("usageTrend"),
        "moveTrends": result.get("moveTrends", [])[:3],
        "itemTrends": result.get("itemTrends", [])[:3],
        "archetypeTrends": result.get("archetypeTrends", [])[:3],
        "winRateTrend": result.get("winRateTrend", {}),
        "commonCores": result.get("commonCores", [])[:3],
        "newCores": result.get("newCores", [])[:3],
        "dataQualityNotes": result.get("dataQualityNotes", []),
    }
    print(json.dumps(preview, indent=2))


def main() -> None:
    sample_names = [
        "Garchomp",
        "Tyranitar",
        "Sneasler",
    ]
    low_sample_name = find_low_sample_pokemon()
    if low_sample_name:
        sample_names.append(low_sample_name)
    sample_names.append("NotARealPokemonName123")

    print("Testing Pokedex meta intelligence with:")
    for name in sample_names:
        print(f"- {name}")

    for name in sample_names:
        print_result(name)


if __name__ == "__main__":
    main()
