"""Smoke test for rule-based archetype scoring."""

from __future__ import annotations

import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.archetype_rule_service import classify_team_archetype


SAMPLE_TEAMS = [
    {
        "pokemon": ["Pelipper", "Basculegion", "Archaludon", "Amoonguss", "Tornadus", "Urshifu"],
    },
    {
        "pokemon": ["Torkoal", "Venusaur", "Farigiraf", "Kingambit", "Flutter Mane", "Urshifu"],
    },
    {
        "pokemon": ["Cresselia", "Torkoal", "Rhyperior", "Farigiraf", "Amoonguss", "Iron Hands"],
    },
    {
        "pokemon": ["Whimsicott", "Garchomp", "Dragonite", "Chi-Yu", "Urshifu", "Gholdengo"],
    },
]


def main() -> None:
    for index, team in enumerate(SAMPLE_TEAMS, start=1):
        result = classify_team_archetype(team)
        print(f"== Team {index} ==")
        print("team pokemon:", team["pokemon"])
        print("primary archetype:", result["primaryArchetype"])
        print("secondary archetype:", result["secondaryArchetype"])
        print("all scores:")
        print(json.dumps(result["scores"], indent=2))
        print("reasons:")
        print(json.dumps(result["reasons"], indent=2))
        print("data quality notes:")
        print(json.dumps(result["dataQualityNotes"], indent=2))
        print()


if __name__ == "__main__":
    main()
