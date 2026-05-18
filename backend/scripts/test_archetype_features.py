"""Smoke test for archetype feature extraction."""

from __future__ import annotations

import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.archetype_feature_service import build_feature_records, load_tournament_teams


def main() -> None:
    teams = load_tournament_teams()
    feature_records = build_feature_records()

    teams_with_full_data = 0
    missing_moves = 0
    missing_items = 0
    missing_abilities = 0

    for team in teams:
        notes = set(team.get("data_quality_notes", []))
        if not notes:
            teams_with_full_data += 1
        if "missing moves" in notes:
            missing_moves += 1
        if "missing items" in notes:
            missing_items += 1
        if "missing abilities" in notes:
            missing_abilities += 1

    print(f"number of teams loaded: {len(teams)}")
    print(f"number of teams with full data: {teams_with_full_data}")
    print(f"number of teams missing moves/items/abilities: {missing_moves}/{missing_items}/{missing_abilities}")
    print()
    print("sample raw team:")
    if teams:
        print(json.dumps(teams[0], indent=2, default=str))
    else:
        print("No teams found.")
    print()
    print("sample extracted feature record:")
    if feature_records:
        print(json.dumps(feature_records[0], indent=2, default=str))
    else:
        print("No feature records found.")


if __name__ == "__main__":
    main()
