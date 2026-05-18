"""Build saved move/item set probability summaries from tournament data."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
for path in (PROJECT_ROOT, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from services.set_probability_service import (  # noqa: E402
    calculate_basic_probabilities,
    calculate_conditional_probabilities,
    load_pokemon_set_records,
    normalize_name,
)


OUTPUT_DIR = BACKEND_DIR / "data" / "competitive" / "analysis_json"
SET_SUMMARY_PATH = OUTPUT_DIR / "set_probability_summary.json"
CONDITIONAL_SUMMARY_PATH = OUTPUT_DIR / "conditional_set_probability_summary.json"

TOP_LIMIT = 10
PAIR_MIN_SAMPLE = 10
LOW_SAMPLE_THRESHOLD = 10
MAX_TEAMMATES_PER_POKEMON = 12
MAX_CONDITIONAL_SHIFTS = 5


def display_name(value: str) -> str:
    normalized = normalize_name(value)
    if not normalized:
        return ""
    return " ".join(part.capitalize() for part in normalized.split())


def summarize_common_values(counter: Counter[str], total: int, limit: int) -> list[dict[str, float | int | str]]:
    if total <= 0:
        return []

    summary = []
    for name, count in counter.most_common(limit):
        probability = count / total
        summary.append(
            {
                "name": display_name(name),
                "count": count,
                "probability": round(probability, 3),
                "percentage": round(probability * 100, 1),
            }
        )
    return summary


def summarize_pokemon(records_by_pokemon: dict[str, list[dict[str, object]]]) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}

    for pokemon_name in sorted(records_by_pokemon):
        records = records_by_pokemon[pokemon_name]
        probability_summary = calculate_basic_probabilities(pokemon_name, limit=TOP_LIMIT)

        teammate_counter: Counter[str] = Counter()
        archetype_counter: Counter[str] = Counter()
        for record in records:
            teammate_counter.update(record.get("teammates", []))
            archetype = normalize_name(record.get("archetype"))
            if archetype:
                archetype_counter[archetype] += 1

        summaries[display_name(pokemon_name)] = {
            "pokemon": display_name(pokemon_name),
            "sampleSize": probability_summary["sampleSize"],
            "moves": probability_summary["moves"][:TOP_LIMIT],
            "items": probability_summary["items"][:TOP_LIMIT],
            "abilities": probability_summary["abilities"][:TOP_LIMIT],
            "natures": probability_summary["natures"][:TOP_LIMIT],
            "commonTeammates": summarize_common_values(
                teammate_counter,
                len(records),
                TOP_LIMIT,
            ),
            "commonArchetypes": summarize_common_values(
                archetype_counter,
                len(records),
                TOP_LIMIT,
            ),
            "dataQualityNotes": probability_summary["dataQualityNotes"],
        }

    return summaries


def biggest_probability_increases(conditional_result: dict[str, object]) -> list[dict[str, object]]:
    shifts: list[dict[str, object]] = []
    for category in ("items", "moves"):
        for entry in conditional_result.get(category, []):
            change = float(entry.get("change", 0.0))
            if change <= 0:
                continue
            shifts.append(
                {
                    "category": category[:-1],
                    "name": entry.get("name", ""),
                    "change": round(change, 1),
                    "conditionalPercentage": entry.get("conditionalPercentage", 0.0),
                    "basePercentage": entry.get("basePercentage", 0.0),
                }
            )

    shifts.sort(
        key=lambda entry: (
            float(entry["change"]),
            float(entry["conditionalPercentage"]),
        ),
        reverse=True,
    )
    return shifts[:MAX_CONDITIONAL_SHIFTS]


def build_conditional_pair_summaries(records_by_pokemon: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    pair_summaries: list[dict[str, object]] = []
    top_conditional_item_shifts: list[dict[str, object]] = []

    for pokemon_name in sorted(records_by_pokemon):
        pokemon_records = records_by_pokemon[pokemon_name]
        teammate_counter: Counter[str] = Counter()
        for record in pokemon_records:
            teammate_counter.update(record.get("teammates", []))

        candidate_teammates = [
            teammate
            for teammate, count in teammate_counter.most_common(MAX_TEAMMATES_PER_POKEMON)
            if count >= PAIR_MIN_SAMPLE
        ]

        for teammate in candidate_teammates:
            conditional_result = calculate_conditional_probabilities(
                pokemon_name,
                teammates=[teammate],
                limit=TOP_LIMIT,
            )
            matched_sample_size = int(conditional_result.get("matchedSampleSize", 0))
            if matched_sample_size < PAIR_MIN_SAMPLE:
                continue

            pair_summary = {
                "pokemon": display_name(pokemon_name),
                "teammate": display_name(teammate),
                "baseSampleSize": int(conditional_result.get("baseSampleSize", 0)),
                "matchedSampleSize": matched_sample_size,
                "baseMoves": [
                    {
                        "name": entry.get("name", ""),
                        "percentage": entry.get("basePercentage", 0.0),
                    }
                    for entry in conditional_result.get("moves", [])[:TOP_LIMIT]
                ],
                "conditionalMoves": conditional_result.get("moves", [])[:TOP_LIMIT],
                "baseItems": [
                    {
                        "name": entry.get("name", ""),
                        "percentage": entry.get("basePercentage", 0.0),
                    }
                    for entry in conditional_result.get("items", [])[:TOP_LIMIT]
                ],
                "conditionalItems": conditional_result.get("items", [])[:TOP_LIMIT],
                "biggestProbabilityIncreases": biggest_probability_increases(conditional_result),
                "dataQualityNotes": conditional_result.get("dataQualityNotes", []),
            }
            pair_summaries.append(pair_summary)

            for entry in pair_summary["conditionalItems"]:
                change = float(entry.get("change", 0.0))
                if change <= 0:
                    continue
                top_conditional_item_shifts.append(
                    {
                        "pokemon": display_name(pokemon_name),
                        "teammate": display_name(teammate),
                        "item": entry.get("name", ""),
                        "change": round(change, 1),
                        "conditionalPercentage": entry.get("conditionalPercentage", 0.0),
                        "matchedSampleSize": matched_sample_size,
                    }
                )

    pair_summaries.sort(
        key=lambda entry: (
            int(entry["matchedSampleSize"]),
            int(entry["baseSampleSize"]),
        ),
        reverse=True,
    )
    top_conditional_item_shifts.sort(
        key=lambda entry: (
            float(entry["change"]),
            int(entry["matchedSampleSize"]),
        ),
        reverse=True,
    )

    return {
        "minimumMatchedSampleSize": PAIR_MIN_SAMPLE,
        "pairCount": len(pair_summaries),
        "pairs": pair_summaries,
        "topConditionalItemShifts": top_conditional_item_shifts[:10],
    }


def build_missing_data_warnings(pokemon_summaries: dict[str, dict[str, object]]) -> list[str]:
    warnings: list[str] = []
    no_ability = [
        summary["pokemon"]
        for summary in pokemon_summaries.values()
        if "No ability data was available for this Pokemon." in summary.get("dataQualityNotes", [])
    ]
    no_nature = [
        summary["pokemon"]
        for summary in pokemon_summaries.values()
        if "No nature data was available for this Pokemon." in summary.get("dataQualityNotes", [])
    ]
    if no_ability:
        warnings.append(f"No ability data for {len(no_ability)} Pokemon.")
    if no_nature:
        warnings.append(f"No nature data for {len(no_nature)} Pokemon.")
    return warnings


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    records = load_pokemon_set_records()
    records_by_pokemon: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        pokemon_name = normalize_name(record.get("pokemon"))
        if pokemon_name:
            records_by_pokemon[pokemon_name].append(record)

    pokemon_summaries = summarize_pokemon(records_by_pokemon)
    conditional_pair_output = build_conditional_pair_summaries(records_by_pokemon)

    low_sample_pokemon = sorted(
        [
            summary["pokemon"]
            for summary in pokemon_summaries.values()
            if int(summary.get("sampleSize", 0)) < LOW_SAMPLE_THRESHOLD
        ]
    )
    missing_data_warnings = build_missing_data_warnings(pokemon_summaries)

    set_summary_output = {
        "generatedAt": datetime.now().isoformat(),
        "pokemonCount": len(pokemon_summaries),
        "totalAppearances": len(records),
        "topLimit": TOP_LIMIT,
        "lowSampleThreshold": LOW_SAMPLE_THRESHOLD,
        "lowSamplePokemon": low_sample_pokemon,
        "missingDataWarnings": missing_data_warnings,
        "pokemon": pokemon_summaries,
    }

    SET_SUMMARY_PATH.write_text(
        json.dumps(set_summary_output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    CONDITIONAL_SUMMARY_PATH.write_text(
        json.dumps(conditional_pair_output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Total Pokémon analyzed: {len(pokemon_summaries)}")
    print(f"Total Pokémon appearances: {len(records)}")
    print(f"Pokémon with low sample size (< {LOW_SAMPLE_THRESHOLD}): {len(low_sample_pokemon)}")
    if low_sample_pokemon:
        print(", ".join(low_sample_pokemon[:25]))
    print("Top conditional item shifts:")
    for entry in conditional_pair_output["topConditionalItemShifts"][:10]:
        print(
            f"- {entry['pokemon']} + {entry['teammate']}: {entry['item']} "
            f"(+{entry['change']:.1f} points, matched sample {entry['matchedSampleSize']})"
        )
    print("Missing data warnings:")
    if missing_data_warnings:
        for warning in missing_data_warnings:
            print(f"- {warning}")
    else:
        print("- None")


if __name__ == "__main__":
    main()
