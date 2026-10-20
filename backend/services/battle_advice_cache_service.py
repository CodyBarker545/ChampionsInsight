"""Fast JSON-backed cache helpers for battle advice tournament context."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from paths import CONDITIONAL_SET_PROBABILITY_SUMMARY_PATH, SET_PROBABILITY_SUMMARY_PATH
from services.set_probability_service import normalize_name, predict_team_sets


def _display_name(value: Any) -> str:
    normalized = normalize_name(value)
    if not normalized:
        return ""
    return " ".join(part.capitalize() for part in normalized.split())


def _normalize_team_member_name(member: Any) -> str:
    if isinstance(member, str):
        return member.strip()
    if isinstance(member, dict):
        candidate = member.get("name") or member.get("pokemon") or member.get("pokemonName")
        return str(candidate or "").strip()
    return str(member or "").strip()


def _normalize_team_input(opponent_team: list[Any] | None) -> tuple[list[str], list[str]]:
    if not opponent_team:
        return [], []

    normalized_names: list[str] = []
    data_quality_notes: list[str] = []
    seen_names: set[str] = set()

    for member in opponent_team:
        raw_name = _normalize_team_member_name(member)
        normalized_name = normalize_name(raw_name)
        if not normalized_name:
            data_quality_notes.append("Ignored a team slot with no recognizable Pokemon name.")
            continue
        if normalized_name in seen_names:
            continue
        seen_names.add(normalized_name)
        normalized_names.append(_display_name(normalized_name))

    return normalized_names, data_quality_notes


@lru_cache(maxsize=1)
def load_set_probability_summary_cache() -> dict[str, Any]:
    if not SET_PROBABILITY_SUMMARY_PATH.exists():
        return {}

    payload = json.loads(SET_PROBABILITY_SUMMARY_PATH.read_text(encoding="utf-8"))
    pokemon_payload = payload.get("pokemon", {})
    normalized_payload: dict[str, Any] = {}
    if not isinstance(pokemon_payload, dict):
        return {}

    for pokemon_name, entry in pokemon_payload.items():
        key = normalize_name(pokemon_name)
        if key and isinstance(entry, dict):
            normalized_payload[key] = entry
    return normalized_payload


@lru_cache(maxsize=1)
def load_conditional_pair_summary_cache() -> dict[tuple[str, str], dict[str, Any]]:
    if not CONDITIONAL_SET_PROBABILITY_SUMMARY_PATH.exists():
        return {}

    payload = json.loads(CONDITIONAL_SET_PROBABILITY_SUMMARY_PATH.read_text(encoding="utf-8"))
    pair_payload = payload.get("pairs", [])
    if not isinstance(pair_payload, list):
        return {}

    normalized_payload: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in pair_payload:
        if not isinstance(entry, dict):
            continue
        pokemon_name = normalize_name(entry.get("pokemon"))
        teammate_name = normalize_name(entry.get("teammate"))
        if pokemon_name and teammate_name:
            normalized_payload[(pokemon_name, teammate_name)] = entry
    return normalized_payload


def _merge_probability_entries(
    base_entries: list[dict[str, Any]],
    conditional_entries: list[tuple[dict[str, Any], str, int]],
    limit: int,
) -> list[dict[str, Any]]:
    base_map: dict[str, dict[str, Any]] = {}
    weighted_changes: dict[str, float] = {}
    weights_by_name: dict[str, int] = {}
    mention_counts: dict[str, int] = {}

    for entry in base_entries or []:
        name = normalize_name(entry.get("name"))
        if not name:
            continue
        base_map[name] = {
            "name": _display_name(name),
            "count": int(entry.get("count", 0) or 0),
            "basePercentage": float(entry.get("percentage", 0.0) or 0.0),
        }

    for pair_entry, _teammate_name, matched_sample_size in conditional_entries:
        for entry in pair_entry or []:
            name = normalize_name(entry.get("name"))
            if not name:
                continue
            if name not in base_map:
                base_map[name] = {
                    "name": _display_name(name),
                    "count": int(entry.get("count", 0) or 0),
                    "basePercentage": float(entry.get("basePercentage", 0.0) or 0.0),
                }
            change = float(entry.get("change", 0.0) or 0.0)
            weighted_changes[name] = weighted_changes.get(name, 0.0) + (change * matched_sample_size)
            weights_by_name[name] = weights_by_name.get(name, 0) + matched_sample_size
            mention_counts[name] = mention_counts.get(name, 0) + 1

    ranked = []
    for name, base_entry in base_map.items():
        base_percentage = float(base_entry.get("basePercentage", 0.0))
        total_weight = max(weights_by_name.get(name, 0), 0)
        average_change = (
            weighted_changes[name] / total_weight
            if total_weight > 0
            else 0.0
        )
        conditional_percentage = max(0.0, min(100.0, base_percentage + average_change))
        ranked.append(
            {
                "name": base_entry["name"],
                "count": int(base_entry.get("count", 0)),
                "basePercentage": round(base_percentage, 1),
                "conditionalPercentage": round(conditional_percentage, 1),
                "change": round(conditional_percentage - base_percentage, 1),
                "supportingPairs": mention_counts.get(name, 0),
            }
        )

    ranked.sort(
        key=lambda entry: (
            float(entry["conditionalPercentage"]),
            float(entry["change"]),
            int(entry["count"]),
        ),
        reverse=True,
    )
    return ranked[:limit]


def _build_context_explanations(
    pokemon_name: str,
    teammate_names: list[str],
    pair_matches: list[dict[str, Any]],
) -> tuple[list[str], list[str], int]:
    if not teammate_names:
        return (
            ["No teammate context was provided, so this uses the broader tournament baseline."],
            [],
            0,
        )

    if not pair_matches:
        return (
            ["No cached teammate-specific set data was available for this team shell, so predictions leaned on the broader tournament baseline."],
            ["No cached teammate-specific set data was available for this team shell."],
            0,
        )

    matched_sample_size = sum(int(entry.get("matchedSampleSize", 0) or 0) for entry in pair_matches)
    biggest_shifts: list[tuple[float, str]] = []
    matched_teammates: list[str] = []

    for pair_match in pair_matches:
        teammate = str(pair_match.get("teammate") or "").strip()
        if teammate:
            matched_teammates.append(teammate)
        for shift in pair_match.get("biggestProbabilityIncreases", [])[:2]:
            change = float(shift.get("change", 0.0) or 0.0)
            if change <= 0:
                continue
            label = f"{shift.get('name', '')} rises with {teammate}"
            biggest_shifts.append((change, label))

    biggest_shifts.sort(key=lambda item: item[0], reverse=True)
    explanations = [
        f"Prediction is using cached teammate trends for {pokemon_name} with {', '.join(matched_teammates[:3])}."
    ]
    for change, label in biggest_shifts[:2]:
        explanations.append(f"{label} (+{change:.1f} points in the matched sample).")
    if matched_sample_size < 20:
        explanations.append("Matched teammate sample size is still modest, so treat the shifts as directional.")

    notes: list[str] = []
    if matched_sample_size < 10:
        notes.append("Cached teammate-specific set data had a low matched sample size.")
    return explanations, notes, matched_sample_size


def _confidence_payload(
    base_sample_size: int,
    matched_sample_size: int,
    top_item_gap: float,
    top_move_gap: float,
) -> tuple[float, str]:
    confidence_score = min(
        1.0,
        0.45 * min(base_sample_size / 80, 1.0)
        + 0.35 * min(matched_sample_size / 40, 1.0)
        + 0.20 * min(max(top_item_gap, top_move_gap) / 25, 1.0),
    )
    if confidence_score >= 0.7:
        return round(confidence_score, 3), "High"
    if confidence_score >= 0.4:
        return round(confidence_score, 3), "Medium"
    return round(confidence_score, 3), "Low"


def _top_gap(entries: list[dict[str, Any]], key: str) -> float:
    if not entries:
        return 0.0
    if len(entries) == 1:
        return float(entries[0].get(key, 0.0) or 0.0)
    return max(
        0.0,
        float(entries[0].get(key, 0.0) or 0.0) - float(entries[1].get(key, 0.0) or 0.0),
    )


def _build_cached_prediction(
    pokemon_name: str,
    teammate_context: list[str],
    summary_entry: dict[str, Any],
    pair_lookup: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    base_moves = list(summary_entry.get("moves", []))
    base_items = list(summary_entry.get("items", []))
    base_abilities = list(summary_entry.get("abilities", []))
    base_natures = list(summary_entry.get("natures", []))
    pair_matches = [
        pair_lookup[(normalize_name(pokemon_name), normalize_name(teammate))]
        for teammate in teammate_context
        if (normalize_name(pokemon_name), normalize_name(teammate)) in pair_lookup
    ]

    conditional_payloads_moves = [
        (pair_match.get("conditionalMoves", []), str(pair_match.get("teammate") or ""), int(pair_match.get("matchedSampleSize", 0) or 0))
        for pair_match in pair_matches
    ]
    conditional_payloads_items = [
        (pair_match.get("conditionalItems", []), str(pair_match.get("teammate") or ""), int(pair_match.get("matchedSampleSize", 0) or 0))
        for pair_match in pair_matches
    ]

    merged_moves = _merge_probability_entries(base_moves, conditional_payloads_moves, 10)
    merged_items = _merge_probability_entries(base_items, conditional_payloads_items, 10)
    explanations, extra_notes, matched_sample_size = _build_context_explanations(
        pokemon_name,
        teammate_context,
        pair_matches,
    )

    predicted_moves = [entry["name"] for entry in merged_moves[:4]]
    predicted_item = merged_items[0]["name"] if merged_items else ""
    predicted_ability = str((base_abilities[:1] or [{}])[0].get("name", "") or "")
    predicted_nature = str((base_natures[:1] or [{}])[0].get("name", "") or "")
    base_sample_size = int(summary_entry.get("sampleSize", 0) or 0)
    confidence, confidence_label = _confidence_payload(
        base_sample_size,
        matched_sample_size,
        _top_gap(merged_items, "conditionalPercentage"),
        _top_gap(merged_moves, "conditionalPercentage"),
    )

    if not pair_matches:
        confidence = min(confidence, 0.6)
        confidence_label = "Medium" if confidence >= 0.4 else "Low"

    return {
        "pokemon": _display_name(pokemon_name),
        "teammateContext": [_display_name(name) for name in teammate_context],
        "baseProbabilities": {
            "pokemon": _display_name(pokemon_name),
            "sampleSize": base_sample_size,
            "moves": base_moves[:10],
            "items": base_items[:10],
            "abilities": base_abilities[:10],
            "natures": base_natures[:10],
            "commonTeammates": list(summary_entry.get("commonTeammates", []))[:10],
            "dataQualityNotes": list(summary_entry.get("dataQualityNotes", [])),
        },
        "conditionalProbabilities": {
            "pokemon": _display_name(pokemon_name),
            "context": {
                "teammates": [_display_name(name) for name in teammate_context],
                "archetype": "",
            },
            "baseSampleSize": base_sample_size,
            "matchedSampleSize": matched_sample_size,
            "moves": merged_moves,
            "items": merged_items,
            "abilities": base_abilities[:10],
            "natures": base_natures[:10],
            "explanations": explanations,
            "dataQualityNotes": [
                *list(summary_entry.get("dataQualityNotes", [])),
                *extra_notes,
            ],
        },
        "likelyMoves": merged_moves[:4],
        "likelyItems": merged_items[:5],
        "likelyAbilities": base_abilities[:3],
        "likelyNature": base_natures[:3],
        "predictedSet": {
            "item": predicted_item,
            "ability": predicted_ability,
            "nature": predicted_nature,
            "moves": predicted_moves,
        },
        "baseSampleSize": base_sample_size,
        "matchedSampleSize": matched_sample_size,
        "confidence": confidence,
        "confidenceLabel": confidence_label,
        "explanations": explanations,
        "dataQualityNotes": [
            *list(summary_entry.get("dataQualityNotes", [])),
            *extra_notes,
        ],
    }


def _build_team_prediction_explanations(
    prediction_by_name: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    team_explanations: list[str] = []
    data_quality_notes: list[str] = []
    for pokemon_name, prediction in prediction_by_name.items():
        top_item = next((entry for entry in prediction.get("likelyItems", []) if float(entry.get("change", 0.0) or 0.0) > 0), None)
        top_move = next((entry for entry in prediction.get("likelyMoves", []) if float(entry.get("change", 0.0) or 0.0) > 0), None)
        if top_item:
            team_explanations.append(
                f"{pokemon_name}: {top_item['name']} gets a teammate-context bump of {top_item['change']:.1f} points."
            )
        elif top_move:
            team_explanations.append(
                f"{pokemon_name}: {top_move['name']} gets a teammate-context bump of {top_move['change']:.1f} points."
            )

        if prediction.get("confidenceLabel") == "Low":
            data_quality_notes.append(f"Low-confidence set prediction for {pokemon_name}.")

    if not team_explanations:
        team_explanations.append(
            "Cached team set predictions leaned mostly on each Pokemon's overall tournament baseline because teammate-specific shifts were limited."
        )
    return team_explanations[:3], data_quality_notes


def predict_team_sets_from_cache(opponent_team: list[Any] | None) -> dict[str, Any]:
    team_names, data_quality_notes = _normalize_team_input(opponent_team)
    if not team_names:
        return {
            "team": [],
            "predictions": {},
            "teamExplanations": ["No opponent team members were provided."],
            "dataQualityNotes": [
                *data_quality_notes,
                "No opponent team members were provided.",
            ],
        }

    basic_summary = load_set_probability_summary_cache()
    pair_summary = load_conditional_pair_summary_cache()
    if not basic_summary:
        return predict_team_sets(team_names)

    predictions: dict[str, dict[str, Any]] = {}
    missing_summary_names: list[str] = []

    for pokemon_name in team_names:
        summary_entry = basic_summary.get(normalize_name(pokemon_name))
        if not summary_entry:
            missing_summary_names.append(_display_name(pokemon_name))
            continue
        teammate_context = [
            teammate for teammate in team_names
            if normalize_name(teammate) != normalize_name(pokemon_name)
        ]
        prediction = _build_cached_prediction(
            pokemon_name,
            teammate_context,
            summary_entry,
            pair_summary,
        )
        predictions[prediction["pokemon"]] = prediction

    if missing_summary_names:
        data_quality_notes.append(
            f"No cached set summary was available for {', '.join(missing_summary_names)}."
        )

    if missing_summary_names and len(predictions) != len(team_names):
        live_fallback = predict_team_sets(team_names)
        for pokemon_name in missing_summary_names:
            live_prediction = live_fallback.get("predictions", {}).get(pokemon_name)
            if live_prediction:
                predictions[pokemon_name] = live_prediction
        data_quality_notes.extend(live_fallback.get("dataQualityNotes", []))

    team_explanations, extra_notes = _build_team_prediction_explanations(predictions)
    return {
        "team": [_display_name(name) for name in team_names],
        "predictions": predictions,
        "teamExplanations": team_explanations,
        "dataQualityNotes": [
            *data_quality_notes,
            *extra_notes,
        ],
    }


def warm_battle_advice_cache() -> None:
    """Prime cached JSON summaries so the first team question is faster."""
    load_set_probability_summary_cache()
    load_conditional_pair_summary_cache()
