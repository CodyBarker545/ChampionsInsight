"""Smoke test for team archetype feature preparation."""

from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.team_archetype_service import load_team_archetype_features


def main() -> None:
    team_records, feature_matrix = load_team_archetype_features()
    print(f"Teams loaded: {len(team_records)}")
    print(f"Feature columns: {feature_matrix.shape[1]}")
    print(f"Sample feature names: {feature_matrix.columns[:20].tolist()}")
    if team_records:
        print(
            "Sample weighting fields:",
            {
                "players_in_tournament": team_records[0].get("players_in_tournament"),
                "rounds_played": team_records[0].get("rounds_played"),
                "player_scale_weight": team_records[0].get("player_scale_weight"),
                "round_scale_weight": team_records[0].get("round_scale_weight"),
                "outcome_weight": team_records[0].get("outcome_weight"),
            },
        )
    print("Sample team record:")
    if team_records:
        print(team_records[0])
    else:
        print("No team records found.")


if __name__ == "__main__":
    main()
