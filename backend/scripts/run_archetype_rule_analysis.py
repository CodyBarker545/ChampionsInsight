"""Run rule-based archetype classification across all tournament teams."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from paths import ARCHETYPE_RULE_SUMMARY_PATH, ARCHETYPE_TEAM_CLASSIFICATIONS_PATH
from services.archetype_feature_service import load_tournament_teams, extract_team_features
from services.archetype_rule_service import classify_team_archetype


def canonical_display(value: str) -> str:
    return " ".join(part.capitalize() for part in str(value or "").replace("-", " ").split())


def safe_average(values: list[float]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    return round(sum(usable) / len(usable), 3)


def top_counter_values(counter: Counter[str], limit: int = 10) -> list[dict[str, Any]]:
    return [
        {"name": canonical_display(name), "count": count}
        for name, count in counter.most_common(limit)
    ]


def summarize_archetype_group(
    archetype_name: str,
    grouped_entries: list[dict[str, Any]],
    total_teams: int,
) -> dict[str, Any]:
    pokemon_counter: Counter[str] = Counter()
    move_counter: Counter[str] = Counter()
    item_counter: Counter[str] = Counter()
    ability_counter: Counter[str] = Counter()
    note_counter: Counter[str] = Counter()

    average_speed_values: list[float] = []
    average_attack_values: list[float] = []
    average_sp_attack_values: list[float] = []
    average_bulk_values: list[float] = []
    average_win_rate_values: list[float] = []

    sample_teams: list[dict[str, Any]] = []

    for entry in grouped_entries:
        team = entry["team"]
        features = entry["features"]

        pokemon_counter.update(team.get("pokemon_names", []))
        move_counter.update(team.get("moves", []))
        item_counter.update(team.get("items", []))
        ability_counter.update(team.get("abilities", []))
        note_counter.update(features.get("data_quality_notes", []))

        if features.get("average_speed") is not None:
            average_speed_values.append(features["average_speed"])
        if features.get("average_attack") is not None:
            average_attack_values.append(features["average_attack"])
        if features.get("average_sp_attack") is not None:
            average_sp_attack_values.append(features["average_sp_attack"])
        if features.get("average_bulk") is not None:
            average_bulk_values.append(features["average_bulk"])
        if features.get("team_win_rate") is not None:
            average_win_rate_values.append(features["team_win_rate"])

        if len(sample_teams) < 5:
            sample_teams.append(
                {
                    "teamId": team.get("team_id"),
                    "playerName": team.get("player_name"),
                    "tournamentName": team.get("tournament_name"),
                    "pokemon": [canonical_display(name) for name in team.get("pokemon_names", [])],
                    "wins": team.get("wins"),
                    "losses": team.get("losses"),
                    "placement": team.get("placement"),
                    "confidence": entry["classification"].get("confidence"),
                    "secondaryArchetype": entry["classification"].get("secondaryArchetype"),
                }
            )

    team_count = len(grouped_entries)
    percentage_of_teams = round((team_count / total_teams) * 100, 3) if total_teams else 0.0

    return {
        "archetype": archetype_name,
        "teamCount": team_count,
        "percentageOfTeams": percentage_of_teams,
        "mostCommonPokemon": top_counter_values(pokemon_counter),
        "mostCommonMoves": top_counter_values(move_counter),
        "mostCommonItems": top_counter_values(item_counter),
        "mostCommonAbilities": top_counter_values(ability_counter),
        "averageSpeed": safe_average(average_speed_values),
        "averageAttack": safe_average(average_attack_values),
        "averageSpecialAttack": safe_average(average_sp_attack_values),
        "averageBulk": safe_average(average_bulk_values),
        "averageWinRate": safe_average(average_win_rate_values),
        "sampleTeams": sample_teams,
        "commonDataQualityNotes": [
            {"note": note, "count": count}
            for note, count in note_counter.most_common(10)
        ],
    }


def main() -> None:
    teams = load_tournament_teams()
    classifications_output: list[dict[str, Any]] = []
    grouped_by_archetype: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_data_counter: Counter[str] = Counter()

    for team in teams:
        features = extract_team_features(team)
        classification = classify_team_archetype(features)

        grouped_by_archetype[classification["primaryArchetype"]].append(
            {
                "team": team,
                "features": features,
                "classification": classification,
            }
        )

        missing_data_counter.update(classification.get("dataQualityNotes", []))

        classifications_output.append(
            {
                "teamId": team.get("team_id"),
                "tournamentId": team.get("tournament_id"),
                "tournamentName": team.get("tournament_name"),
                "playerId": team.get("player_id"),
                "playerName": team.get("player_name"),
                "pokemon": [canonical_display(name) for name in team.get("pokemon_names", [])],
                "wins": team.get("wins"),
                "losses": team.get("losses"),
                "placement": team.get("placement"),
                "teamWinRate": team.get("team_win_rate"),
                "primaryArchetype": classification.get("primaryArchetype"),
                "secondaryArchetype": classification.get("secondaryArchetype"),
                "confidence": classification.get("confidence"),
                "scores": classification.get("scores"),
                "reasons": classification.get("reasons"),
                "dataQualityNotes": classification.get("dataQualityNotes", []),
            }
        )

    total_teams = len(teams)
    archetype_summaries = [
        summarize_archetype_group(archetype_name, grouped_entries, total_teams)
        for archetype_name, grouped_entries in sorted(
            grouped_by_archetype.items(),
            key=lambda item: len(item[1]),
            reverse=True,
        )
    ]

    summary_output = {
        "totalTeamsAnalyzed": total_teams,
        "archetypeCount": len(archetype_summaries),
        "archetypes": archetype_summaries,
        "missingDataWarnings": [
            {"note": note, "count": count}
            for note, count in missing_data_counter.most_common()
        ],
    }

    ARCHETYPE_RULE_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARCHETYPE_RULE_SUMMARY_PATH.write_text(
        json.dumps(summary_output, indent=2),
        encoding="utf-8",
    )
    ARCHETYPE_TEAM_CLASSIFICATIONS_PATH.write_text(
        json.dumps(
            {
                "totalTeamsAnalyzed": total_teams,
                "teams": classifications_output,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    archetype_counts = [
        (summary["archetype"], summary["teamCount"])
        for summary in archetype_summaries
    ]
    top_three = archetype_counts[:3]

    print(f"total teams analyzed: {total_teams}")
    print("archetype counts:")
    for archetype_name, count in archetype_counts:
        print(f"  - {archetype_name}: {count}")

    print("top 3 archetypes:")
    for archetype_name, count in top_three:
        print(f"  - {archetype_name}: {count}")

    print("missing data warnings:")
    if missing_data_counter:
        for note, count in missing_data_counter.most_common():
            print(f"  - {note}: {count}")
    else:
        print("  - none")


if __name__ == "__main__":
    main()
