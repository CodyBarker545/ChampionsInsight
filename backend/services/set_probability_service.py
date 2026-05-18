"""Explainable move/item/ability/nature usage probabilities from tournament rows."""

from __future__ import annotations

import json
from collections import Counter
from functools import lru_cache
from typing import Any

import pandas as pd

from services.archetype_rule_service import classify_team_archetype
from services.pokemon_data_service import lookup_keys_for_value
from services.team_archetype_service import load_tournament_team_rows


MOVE_COLUMNS = ["move_1", "move_2", "move_3", "move_4"]


def normalize_name(value: Any) -> str:
    """Normalize a Pokemon, move, item, ability, or nature string."""
    text = str(value or "").strip().lower()
    if not text or text in {"none", "nan", "null", "n/a"}:
        return ""

    replacements = {
        "Ã¢â‚¬â„¢": "'",
        "â€™": "'",
        "_": " ",
        "-": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)

    cleaned = []
    for character in text:
        if character.isalnum() or character in {" ", "'"}:
            cleaned.append(character)
        else:
            cleaned.append(" ")

    return " ".join("".join(cleaned).split())


def _display_name(value: Any) -> str:
    normalized = normalize_name(value)
    if not normalized:
        return ""
    return " ".join(part.capitalize() for part in normalized.split())


def _lookup_aliases(value: Any) -> set[str]:
    aliases = {normalized for normalized in (normalize_name(value),) if normalized}
    for key in lookup_keys_for_value(value):
        normalized_key = normalize_name(key)
        if normalized_key:
            aliases.add(normalized_key)
    return aliases


def _coerce_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _extract_nature_from_raw(raw_value: Any) -> str:
    if raw_value is None or (isinstance(raw_value, float) and pd.isna(raw_value)):
        return ""

    try:
        payload = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""

    if not isinstance(payload, dict):
        return ""

    candidates = [
        payload.get("nature"),
        (payload.get("stats") or {}).get("nature") if isinstance(payload.get("stats"), dict) else None,
        (payload.get("spread") or {}).get("nature") if isinstance(payload.get("spread"), dict) else None,
    ]

    for candidate in candidates:
        normalized = normalize_name(candidate)
        if normalized:
            return normalized
    return ""


def _team_payload_from_group(group: pd.DataFrame) -> dict[str, Any]:
    ordered_group = group.copy()
    if "team_slot" in ordered_group.columns:
        ordered_group["_slot_sort"] = pd.to_numeric(
            ordered_group["team_slot"],
            errors="coerce",
        )
        ordered_group = ordered_group.sort_values(
            by=["_slot_sort", "pokemon"],
            na_position="last",
        )

    pokemon = [str(value).strip() for value in ordered_group.get("pokemon", []).tolist() if str(value).strip()]
    moves = []
    items = []
    abilities = []

    for _, row in ordered_group.iterrows():
        member_moves = []
        for column in MOVE_COLUMNS:
            normalized_move = normalize_name(row.get(column))
            if normalized_move:
                member_moves.append(normalized_move)
        moves.append(member_moves)
        items.append(str(row.get("item") or "").strip())
        abilities.append(str(row.get("ability") or "").strip())

    return {
        "pokemon": pokemon,
        "moves": moves,
        "items": items,
        "abilities": abilities,
    }


@lru_cache(maxsize=1)
def load_pokemon_set_records() -> list[dict[str, Any]]:
    """Return one normalized record per Pokemon appearance in tournament data."""
    team_rows = load_tournament_team_rows()
    if team_rows.empty:
        return []

    grouped = team_rows.groupby(["tournament_id", "player_id"], dropna=True, sort=False)
    records: list[dict[str, Any]] = []
    noted_missing_nature = False

    for team_key, group in grouped:
        team_payload = _team_payload_from_group(group)
        classification = classify_team_archetype(team_payload)
        primary_archetype = classification.get("displayLabel") or classification.get("primaryArchetype") or ""
        normalized_team_names = [normalize_name(name) for name in team_payload["pokemon"] if normalize_name(name)]

        for _, row in group.iterrows():
            pokemon_name = normalize_name(row.get("pokemon"))
            if not pokemon_name:
                continue

            moves = []
            for column in MOVE_COLUMNS:
                normalized_move = normalize_name(row.get(column))
                if normalized_move:
                    moves.append(normalized_move)

            item = normalize_name(row.get("item"))
            ability = normalize_name(row.get("ability"))
            nature = _extract_nature_from_raw(row.get("raw"))
            teammates = [name for name in normalized_team_names if name != pokemon_name]
            tournament_date = row.get("tournament_date")
            tournament_date = None if pd.isna(tournament_date) else str(tournament_date)

            record = {
                "team_id": f"{team_key[0]}::{team_key[1]}",
                "tournament_id": str(team_key[0] or ""),
                "player_id": str(team_key[1] or ""),
                "pokemon": pokemon_name,
                "moves": moves,
                "item": item,
                "ability": ability,
                "nature": nature,
                "teammates": teammates,
                "archetype": primary_archetype,
                "tournament_date": tournament_date,
                "wins": _coerce_int(row.get("wins")),
                "losses": _coerce_int(row.get("losses")),
                "placement": _coerce_int(row.get("placing")),
            }

            if not nature:
                noted_missing_nature = True

            records.append(record)

    if noted_missing_nature:
        for record in records:
            record.setdefault("data_quality_notes", [])
            if "nature data unavailable for many rows" not in record["data_quality_notes"]:
                record["data_quality_notes"].append("nature data unavailable for many rows")

    return records


def _probability_entries(counter: Counter[str], denominator: int, limit: int) -> list[dict[str, Any]]:
    if denominator <= 0:
        return []

    return [
        {
            "name": _display_name(name),
            "count": count,
            "probability": round(count / denominator, 3),
            "percentage": round((count / denominator) * 100, 1),
        }
        for name, count in counter.most_common(limit)
    ]


def _top_gap_score(entries: list[dict[str, Any]], percentage_key: str) -> float:
    if not entries:
        return 0.0
    if len(entries) == 1:
        return 1.0

    top_value = float(entries[0].get(percentage_key, 0.0))
    next_value = float(entries[1].get(percentage_key, 0.0))
    return min(max((top_value - next_value) / 25.0, 0.0), 1.0)


def _collect_matching_records(pokemon_name: str) -> list[dict[str, Any]]:
    normalized_pokemon = _lookup_aliases(pokemon_name)
    return [
        record
        for record in load_pokemon_set_records()
        if record.get("pokemon") in normalized_pokemon
    ]


def _summarize_missing_fields(records: list[dict[str, Any]]) -> tuple[int, int, int]:
    missing_item_count = 0
    missing_ability_count = 0
    missing_nature_count = 0

    for record in records:
        if not record.get("item"):
            missing_item_count += 1
        if not record.get("ability"):
            missing_ability_count += 1
        if not record.get("nature"):
            missing_nature_count += 1

    return missing_item_count, missing_ability_count, missing_nature_count


def _build_quality_notes(records: list[dict[str, Any]], label: str = "this Pokemon") -> list[str]:
    sample_size = len(records)
    data_quality_notes: list[str] = []
    missing_item_count, missing_ability_count, missing_nature_count = _summarize_missing_fields(records)

    if sample_size and sample_size < 5:
        data_quality_notes.append("Low sample size. Treat these probabilities as directional only.")
    if sample_size and missing_item_count == sample_size:
        data_quality_notes.append(f"No item data was available for {label}.")
    elif missing_item_count:
        data_quality_notes.append(f"Missing item data for {missing_item_count} appearances.")
    if sample_size and missing_ability_count == sample_size:
        data_quality_notes.append(f"No ability data was available for {label}.")
    elif missing_ability_count:
        data_quality_notes.append(f"Missing ability data for {missing_ability_count} appearances.")
    if sample_size and missing_nature_count == sample_size:
        data_quality_notes.append(f"No nature data was available for {label}.")
    elif missing_nature_count:
        data_quality_notes.append(f"Missing nature data for {missing_nature_count} appearances.")

    return data_quality_notes


def _basic_probability_result(
    pokemon_name: str,
    matching_records: list[dict[str, Any]],
    limit: int,
) -> dict[str, Any]:
    sample_size = len(matching_records)
    move_counter: Counter[str] = Counter()
    item_counter: Counter[str] = Counter()
    ability_counter: Counter[str] = Counter()
    nature_counter: Counter[str] = Counter()
    teammate_counter: Counter[str] = Counter()

    for record in matching_records:
        move_counter.update(set(record.get("moves", [])))
        teammate_counter.update(set(record.get("teammates", [])))

        if record.get("item"):
            item_counter[record["item"]] += 1
        if record.get("ability"):
            ability_counter[record["ability"]] += 1
        if record.get("nature"):
            nature_counter[record["nature"]] += 1

    return {
        "pokemon": _display_name(pokemon_name),
        "sampleSize": sample_size,
        "moves": _probability_entries(move_counter, sample_size, limit),
        "items": _probability_entries(item_counter, sample_size, limit),
        "abilities": _probability_entries(ability_counter, sample_size, limit),
        "natures": _probability_entries(nature_counter, sample_size, limit),
        "commonTeammates": _probability_entries(teammate_counter, sample_size, limit),
        "dataQualityNotes": _build_quality_notes(matching_records, "this Pokemon"),
    }


def _normalize_context_teammates(teammates: list[str] | None) -> list[str]:
    if not teammates:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for teammate in teammates:
        aliases = _lookup_aliases(teammate)
        canonical = min(aliases, key=len) if aliases else normalize_name(teammate)
        if canonical and canonical not in seen:
            normalized.append(canonical)
            seen.add(canonical)
    return normalized


def _record_teammate_aliases(record: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    for teammate in record.get("teammates", []):
        aliases.update(_lookup_aliases(teammate))
    return aliases


def _weight_record(
    record: dict[str, Any],
    teammates: list[str],
    archetype: str,
) -> tuple[float, int]:
    record_teammates = _record_teammate_aliases(record)
    requested_teammates = set(teammates)
    matched_teammate_count = len(record_teammates & requested_teammates)
    teammate_count = len(requested_teammates)

    archetype_match = bool(archetype and normalize_name(record.get("archetype")) == archetype)

    if teammate_count == 0 and not archetype:
        return 1.0, 0

    if teammate_count > 0 and matched_teammate_count == teammate_count:
        base_weight = 1.0
    elif teammate_count > 0 and matched_teammate_count > 0:
        base_weight = 0.45 + 0.35 * (matched_teammate_count / teammate_count)
    elif archetype_match:
        base_weight = 0.2
    else:
        base_weight = 0.05

    if archetype_match:
        base_weight += 0.15 if matched_teammate_count > 0 else 0.1

    return min(base_weight, 1.0), matched_teammate_count


def _weighted_probability_map(
    records: list[dict[str, Any]],
    weights: list[float],
    extractor,
) -> tuple[dict[str, float], dict[str, int]]:
    weighted_counts: dict[str, float] = {}
    raw_counts: dict[str, int] = {}

    for record, weight in zip(records, weights):
        values = extractor(record)
        for value in values:
            weighted_counts[value] = weighted_counts.get(value, 0.0) + weight
            raw_counts[value] = raw_counts.get(value, 0) + 1

    return weighted_counts, raw_counts


def _rank_conditional_entries(
    base_entries: list[dict[str, Any]],
    records: list[dict[str, Any]],
    weights: list[float],
    denominator: float,
    limit: int,
    extractor,
) -> list[dict[str, Any]]:
    base_map = {normalize_name(entry["name"]): entry for entry in base_entries}
    weighted_counts, raw_counts = _weighted_probability_map(records, weights, extractor)

    for key in weighted_counts:
        if key not in base_map:
            base_map[key] = {
                "name": _display_name(key),
                "count": 0,
                "probability": 0.0,
                "percentage": 0.0,
            }

    ranked = []
    for key, base_entry in base_map.items():
        conditional_probability = (
            weighted_counts.get(key, 0.0) / denominator
            if denominator > 0
            else 0.0
        )
        conditional_percentage = round(conditional_probability * 100, 1)
        base_percentage = round(float(base_entry.get("percentage", 0.0)), 1)
        ranked.append(
            {
                "name": base_entry["name"],
                "basePercentage": base_percentage,
                "conditionalPercentage": conditional_percentage,
                "change": round(conditional_percentage - base_percentage, 1),
                "count": raw_counts.get(key, 0),
                "weightedCount": round(weighted_counts.get(key, 0.0), 3),
            }
        )

    ranked.sort(
        key=lambda entry: (
            entry["conditionalPercentage"],
            entry["change"],
            entry["count"],
        ),
        reverse=True,
    )
    return ranked[:limit]


def _build_conditional_explanations(
    pokemon_name: str,
    teammates: list[str],
    archetype: str,
    items: list[dict[str, Any]],
    moves: list[dict[str, Any]],
    matched_sample_size: int,
    exact_match_count: int,
    partial_match_count: int,
) -> list[str]:
    explanations: list[str] = []
    teammate_display = ", ".join(_display_name(name) for name in teammates)

    if not teammates and not archetype:
        return ["No context filters were provided, so the conditional view matches the base tournament sample."]

    if teammate_display and exact_match_count == 0 and partial_match_count == 0:
        explanations.append(
            f"No direct teammate matches were found for {pokemon_name} with {teammate_display}, so the result falls back to the broader tournament sample."
        )
        if archetype:
            explanations.append(
                f"The fallback still gives a light boost to teams labeled {archetype} when available."
            )
        return explanations

    if teammate_display and exact_match_count == 0 and partial_match_count > 0:
        explanations.append(
            f"No exact teammate sample was available for {pokemon_name} with {teammate_display}, so partial teammate matches were weighted instead."
        )

    boosted_items = [entry for entry in items if entry["change"] >= 8][:2]
    boosted_moves = [entry for entry in moves if entry["change"] >= 5][:2]

    for entry in boosted_items:
        if teammate_display:
            explanations.append(
                f"{entry['name']} is more likely because {pokemon_name} appears with {teammate_display} in the matched tournament sample."
            )
        elif archetype:
            explanations.append(
                f"{entry['name']} is more likely in the matched {archetype} tournament sample."
            )

    for entry in boosted_moves:
        if teammate_display:
            explanations.append(
                f"{entry['name']} gains usage when {pokemon_name} is paired with {teammate_display}."
            )
        elif archetype:
            explanations.append(
                f"{entry['name']} gains usage in the matched {archetype} tournament sample."
            )

    if not explanations:
        if teammate_display and archetype:
            explanations.append(
                f"Conditional probabilities were weighted toward teams containing {teammate_display} and the {archetype} archetype."
            )
        elif teammate_display:
            explanations.append(
                f"Conditional probabilities were weighted toward teams containing {teammate_display}."
            )
        elif archetype:
            explanations.append(
                f"Conditional probabilities were weighted toward teams classified as {archetype}."
            )
        else:
            explanations.append(
                f"Conditional probabilities were estimated from {matched_sample_size} matched tournament appearances."
            )

    return explanations


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
        normalized_names.append(raw_name)

    return normalized_names, data_quality_notes


def _build_team_prediction_explanations(
    prediction_by_name: dict[str, dict[str, Any]],
) -> tuple[list[str], list[str]]:
    team_explanations: list[str] = []
    data_quality_notes: list[str] = []

    boosted_summaries: list[str] = []
    low_confidence_names: list[str] = []
    insufficient_data_names: list[str] = []

    for pokemon_name, prediction in prediction_by_name.items():
        top_item = next(
            (
                entry for entry in prediction.get("likelyItems", [])
                if entry.get("change", 0.0) >= 8
            ),
            None,
        )
        top_move = next(
            (
                entry for entry in prediction.get("likelyMoves", [])
                if entry.get("change", 0.0) >= 5
            ),
            None,
        )
        if top_item:
            boosted_summaries.append(
                f"{pokemon_name}: {top_item['name']} rises by {top_item['change']:.1f} points with the current teammate context."
            )
        elif top_move:
            boosted_summaries.append(
                f"{pokemon_name}: {top_move['name']} rises by {top_move['change']:.1f} points with the current teammate context."
            )

        if prediction.get("confidenceLabel") == "Low":
            low_confidence_names.append(pokemon_name)
        if prediction.get("baseSampleSize", 0) == 0:
            insufficient_data_names.append(pokemon_name)

    if boosted_summaries:
        team_explanations.extend(boosted_summaries[:3])
    if low_confidence_names:
        team_explanations.append(
            f"Lower-confidence predictions: {', '.join(low_confidence_names)}."
        )
        data_quality_notes.append(
            f"Low-confidence set predictions for {', '.join(low_confidence_names)}."
        )
    if insufficient_data_names:
        team_explanations.append(
            f"Insufficient tournament data for: {', '.join(insufficient_data_names)}."
        )
        data_quality_notes.append(
            f"Insufficient tournament data for {', '.join(insufficient_data_names)}."
        )

    if not team_explanations:
        team_explanations.append(
            "Team-level set predictions are driven mostly by each Pokemon's overall tournament sample because teammate-specific shifts were limited."
        )

    return team_explanations, data_quality_notes


def calculate_basic_probabilities(pokemon_name: str, limit: int = 10) -> dict[str, Any]:
    normalized_pokemon = normalize_name(pokemon_name)
    matching_records = _collect_matching_records(normalized_pokemon)

    sample_size = len(matching_records)
    data_quality_notes: list[str] = []

    if not normalized_pokemon:
        data_quality_notes.append("A Pokemon name is required.")
    elif sample_size == 0:
        data_quality_notes.append(f"No tournament records found for {pokemon_name}.")

    result = _basic_probability_result(normalized_pokemon or pokemon_name, matching_records, limit)
    result["dataQualityNotes"] = [
        *data_quality_notes,
        *result["dataQualityNotes"],
    ]
    return result


def calculate_conditional_probabilities(
    pokemon_name: str,
    teammates: list[str] | None = None,
    archetype: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    normalized_pokemon = normalize_name(pokemon_name)
    normalized_teammates = _normalize_context_teammates(teammates)
    normalized_archetype = normalize_name(archetype)
    matching_records = _collect_matching_records(normalized_pokemon)
    base_result = _basic_probability_result(normalized_pokemon or pokemon_name, matching_records, limit)
    data_quality_notes = list(base_result["dataQualityNotes"])

    if not normalized_pokemon:
        return {
            "pokemon": _display_name(pokemon_name),
            "context": {
                "teammates": [_display_name(name) for name in normalized_teammates],
                "archetype": archetype or "",
            },
            "baseSampleSize": 0,
            "matchedSampleSize": 0,
            "moves": [],
            "items": [],
            "abilities": [],
            "natures": [],
            "explanations": ["A Pokemon name is required."],
            "dataQualityNotes": ["A Pokemon name is required."],
        }

    if not matching_records:
        return {
            "pokemon": _display_name(normalized_pokemon),
            "context": {
                "teammates": [_display_name(name) for name in normalized_teammates],
                "archetype": archetype or "",
            },
            "baseSampleSize": 0,
            "matchedSampleSize": 0,
            "moves": [],
            "items": [],
            "abilities": [],
            "natures": [],
            "explanations": [f"No tournament records found for {pokemon_name}."],
            "dataQualityNotes": [f"No tournament records found for {pokemon_name}."],
        }

    weights: list[float] = []
    matched_records: list[dict[str, Any]] = []
    exact_match_count = 0
    partial_match_count = 0
    archetype_only_count = 0

    for record in matching_records:
        weight, matched_teammate_count = _weight_record(
            record,
            normalized_teammates,
            normalized_archetype,
        )
        if weight <= 0:
            continue

        weights.append(weight)
        matched_records.append(record)

        requested_teammate_count = len(normalized_teammates)
        record_archetype_match = bool(
            normalized_archetype
            and normalize_name(record.get("archetype")) == normalized_archetype
        )
        if requested_teammate_count and matched_teammate_count == requested_teammate_count:
            exact_match_count += 1
        elif matched_teammate_count > 0:
            partial_match_count += 1
        elif record_archetype_match:
            archetype_only_count += 1

    weighted_denominator = sum(weights)
    if not normalized_teammates and not normalized_archetype:
        matched_sample_size = base_result["sampleSize"]
    elif exact_match_count > 0:
        matched_sample_size = exact_match_count
    else:
        matched_sample_size = partial_match_count + archetype_only_count

    if normalized_teammates and exact_match_count == 0 and partial_match_count > 0:
        data_quality_notes.append("No exact teammate matches were found. Falling back to partial teammate matching.")
    if normalized_teammates and exact_match_count == 0 and partial_match_count == 0:
        data_quality_notes.append("No direct teammate matches were found. Conditional probabilities fall back toward the overall sample.")
    if weighted_denominator and 0 < matched_sample_size < 8 and normalized_teammates:
        data_quality_notes.append("Conditional teammate sample size was low, so weighted partial matches were used.")
    if weighted_denominator and matched_sample_size < 5 and normalized_archetype and not normalized_teammates:
        data_quality_notes.append("Matched archetype sample size was low, so all appearances were lightly weighted as fallback.")

    move_entries = _rank_conditional_entries(
        base_result["moves"],
        matched_records,
        weights,
        weighted_denominator,
        limit,
        lambda record: set(record.get("moves", [])),
    )
    item_entries = _rank_conditional_entries(
        base_result["items"],
        matched_records,
        weights,
        weighted_denominator,
        limit,
        lambda record: [record["item"]] if record.get("item") else [],
    )
    ability_entries = _rank_conditional_entries(
        base_result["abilities"],
        matched_records,
        weights,
        weighted_denominator,
        limit,
        lambda record: [record["ability"]] if record.get("ability") else [],
    )
    nature_entries = _rank_conditional_entries(
        base_result["natures"],
        matched_records,
        weights,
        weighted_denominator,
        limit,
        lambda record: [record["nature"]] if record.get("nature") else [],
    )

    explanations = _build_conditional_explanations(
        _display_name(normalized_pokemon),
        normalized_teammates,
        archetype or "",
        item_entries,
        move_entries,
        matched_sample_size,
        exact_match_count,
        partial_match_count,
    )

    return {
        "pokemon": _display_name(normalized_pokemon),
        "context": {
            "teammates": [_display_name(name) for name in normalized_teammates],
            "archetype": archetype or "",
        },
        "baseSampleSize": base_result["sampleSize"],
        "matchedSampleSize": matched_sample_size,
        "moves": move_entries,
        "items": item_entries,
        "abilities": ability_entries,
        "natures": nature_entries,
        "explanations": explanations,
        "dataQualityNotes": data_quality_notes,
    }


def predict_likely_set(
    pokemon_name: str,
    teammates: list[str] | None = None,
    archetype: str | None = None,
) -> dict[str, Any]:
    conditional_result = calculate_conditional_probabilities(
        pokemon_name,
        teammates=teammates,
        archetype=archetype,
        limit=10,
    )

    def top_name(entries: list[dict[str, Any]]) -> str:
        return entries[0]["name"] if entries else ""

    top_moves = [entry["name"] for entry in conditional_result["moves"][:4]]
    top_item = top_name(conditional_result["items"])
    top_ability = top_name(conditional_result["abilities"])
    top_nature = top_name(conditional_result["natures"])

    matched_sample_size = int(conditional_result.get("matchedSampleSize", 0))
    base_sample_size = int(conditional_result.get("baseSampleSize", 0))
    has_context = bool(conditional_result["context"]["teammates"] or conditional_result["context"]["archetype"])
    top_move_strength = (
        sum(entry["conditionalPercentage"] for entry in conditional_result["moves"][:4]) / max(1, len(conditional_result["moves"][:4]))
        if conditional_result["moves"]
        else 0.0
    )
    item_gap_score = _top_gap_score(conditional_result["items"], "conditionalPercentage")
    move_gap_score = _top_gap_score(conditional_result["moves"], "conditionalPercentage")
    certainty_score = max(item_gap_score, move_gap_score)

    context_sample_size = matched_sample_size if has_context else base_sample_size
    confidence_score = min(
        1.0,
        0.35 * min(context_sample_size / 20, 1.0)
        + 0.20 * min(base_sample_size / 100, 1.0)
        + 0.20 * min(top_move_strength / 100, 1.0)
        + 0.25 * certainty_score,
    )
    if has_context and matched_sample_size == 0:
        confidence_score = min(confidence_score, 0.35)

    if confidence_score >= 0.7:
        confidence_label = "High"
    elif confidence_score >= 0.4:
        confidence_label = "Medium"
    else:
        confidence_label = "Low"

    explanation_parts = []
    if conditional_result["context"]["teammates"] and matched_sample_size > 0:
        explanation_parts.append(
            f"Prediction is weighted toward teams containing {', '.join(conditional_result['context']['teammates'])}."
        )
    elif conditional_result["context"]["teammates"]:
        explanation_parts.append(
            f"No direct teammate sample was found for {', '.join(conditional_result['context']['teammates'])}, so this prediction leans on the broader tournament baseline."
        )
    if conditional_result["context"]["archetype"]:
        explanation_parts.append(
            f"Prediction also favors the {conditional_result['context']['archetype']} archetype."
        )
    if conditional_result["explanations"]:
        explanation_parts.append(conditional_result["explanations"][0])
    if has_context:
        explanation_parts.append(
            f"Confidence is {confidence_label.lower()} from {matched_sample_size} context-matched appearances out of {base_sample_size} total samples."
        )
    else:
        explanation_parts.append(
            f"Confidence is {confidence_label.lower()} from {base_sample_size} total tournament appearances."
        )

    return {
        "pokemon": conditional_result["pokemon"],
        "context": conditional_result["context"],
        "baseSampleSize": base_sample_size,
        "matchedSampleSize": matched_sample_size,
        "predictedItem": top_item,
        "predictedAbility": top_ability,
        "predictedNature": top_nature,
        "predictedMoves": top_moves,
        "confidence": round(confidence_score, 3),
        "confidenceLabel": confidence_label,
        "explanation": " ".join(part for part in explanation_parts if part),
        "dataQualityNotes": conditional_result["dataQualityNotes"],
    }


def predict_team_sets(opponent_team: list[Any] | None) -> dict[str, Any]:
    team_names, data_quality_notes = _normalize_team_input(opponent_team)
    display_team = [_display_name(name) for name in team_names]

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

    predictions: dict[str, dict[str, Any]] = {}

    for pokemon_name in team_names:
        teammate_context = [name for name in team_names if normalize_name(name) != normalize_name(pokemon_name)]
        base_probabilities = calculate_basic_probabilities(pokemon_name, limit=10)
        conditional_probabilities = calculate_conditional_probabilities(
            pokemon_name,
            teammates=teammate_context,
            limit=10,
        )
        predicted_set = predict_likely_set(
            pokemon_name,
            teammates=teammate_context,
        )

        predictions[predicted_set["pokemon"]] = {
            "pokemon": predicted_set["pokemon"],
            "teammateContext": [_display_name(name) for name in teammate_context],
            "baseProbabilities": base_probabilities,
            "conditionalProbabilities": conditional_probabilities,
            "likelyMoves": conditional_probabilities["moves"][:4],
            "likelyItems": conditional_probabilities["items"][:5],
            "likelyAbilities": conditional_probabilities["abilities"][:3],
            "likelyNature": conditional_probabilities["natures"][:3],
            "predictedSet": {
                "item": predicted_set["predictedItem"],
                "ability": predicted_set["predictedAbility"],
                "nature": predicted_set["predictedNature"],
                "moves": predicted_set["predictedMoves"],
            },
            "baseSampleSize": predicted_set["baseSampleSize"],
            "matchedSampleSize": predicted_set["matchedSampleSize"],
            "confidence": predicted_set["confidence"],
            "confidenceLabel": predicted_set["confidenceLabel"],
            "explanations": conditional_probabilities["explanations"][:],
            "dataQualityNotes": [
                *predicted_set["dataQualityNotes"],
            ],
        }

    team_explanations, extra_notes = _build_team_prediction_explanations(predictions)

    return {
        "team": display_team,
        "predictions": predictions,
        "teamExplanations": team_explanations,
        "dataQualityNotes": [
            *data_quality_notes,
            *extra_notes,
        ],
    }
