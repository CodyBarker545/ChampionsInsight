"""Tournament-backed meta intelligence for Pokemon detail views."""

from __future__ import annotations

import json
from collections import Counter
from datetime import timedelta
from functools import lru_cache
from itertools import combinations
from typing import Any

import pandas as pd

from paths import POKEDEX_META_INTELLIGENCE_PATH
from services.pokedex_service import get_pokedex_grid_entries
from services.pokedex_service import get_pokedex_entry_by_name
from services.pokemon_data_service import alternate_lookup_names
from services.set_probability_service import load_pokemon_set_records, normalize_name
from services.team_archetype_service import create_team_records, load_tournament_team_rows


EMPTY_META_INTELLIGENCE = {
    "pokemon": "",
    "usageTrend": {
        "currentUsageRate": 0.0,
        "previousUsageRate": 0.0,
        "absoluteChange": 0.0,
        "percentageChange": 0.0,
        "trendDirection": "stable",
        "confidence": 0.0,
        "confidenceLabel": "Low",
        "sampleSize": 0,
        "explanation": "",
    },
    "moveTrends": [],
    "itemTrends": [],
    "archetypeTrends": [],
    "winRateTrend": {},
    "commonCores": [],
    "newCores": [],
    "dataQualityNotes": [],
}


def _empty_result(pokemon_name: str) -> dict[str, Any]:
    return {
        **EMPTY_META_INTELLIGENCE,
        "pokemon": pokemon_name,
        "usageTrend": {
            **EMPTY_META_INTELLIGENCE["usageTrend"],
        },
        "moveTrends": [],
        "itemTrends": [],
        "archetypeTrends": [],
        "winRateTrend": {},
        "commonCores": [],
        "newCores": [],
        "dataQualityNotes": [],
    }


def _cache_key(value: str) -> str:
    return normalize_name(value)


@lru_cache(maxsize=1)
def load_meta_intelligence_cache() -> dict[str, dict[str, Any]]:
    if not POKEDEX_META_INTELLIGENCE_PATH.exists():
        return {}

    with open(POKEDEX_META_INTELLIGENCE_PATH, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        return {}

    normalized_payload: dict[str, dict[str, Any]] = {}
    for pokemon_name, entry in payload.items():
        key = _cache_key(pokemon_name)
        if key and isinstance(entry, dict):
            normalized_payload[key] = entry

    return normalized_payload


@lru_cache(maxsize=1)
def load_meta_team_records() -> list[dict[str, Any]]:
    return create_team_records(load_tournament_team_rows())


def _build_aliases(pokemon_name: str) -> set[str]:
    aliases = {
        normalize_name(candidate)
        for candidate in alternate_lookup_names(pokemon_name)
        if normalize_name(candidate)
    }
    aliases.add(normalize_name(pokemon_name))
    return {alias for alias in aliases if alias}


def _match_target_records(pokemon_name: str) -> list[dict[str, Any]]:
    aliases = _build_aliases(pokemon_name)
    return [
        record
        for record in load_pokemon_set_records()
        if normalize_name(record.get("pokemon")) in aliases
    ]


def _match_target_teams(pokemon_name: str) -> list[dict[str, Any]]:
    aliases = _build_aliases(pokemon_name)
    matching_teams = []
    for team in load_meta_team_records():
        normalized_members = {normalize_name(member) for member in team.get("pokemon", [])}
        if normalized_members & aliases:
            matching_teams.append(team)
    return matching_teams


def _parse_timestamp(record: dict[str, Any]) -> pd.Timestamp | None:
    value = record.get("date") or record.get("tournament_date")
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed


def _ordered_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dated = []
    undated = []
    for index, record in enumerate(records):
        timestamp = _parse_timestamp(record)
        payload = (timestamp, index, record)
        if timestamp is None:
            undated.append(payload)
        else:
            dated.append(payload)

    dated.sort(key=lambda item: (item[0], item[1]))
    undated.sort(key=lambda item: item[1])
    return [record for _timestamp, _index, record in [*dated, *undated]]


def _split_periods(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    if len(records) < 2:
        return records[:], [], "insufficient"

    ordered = _ordered_records(records)
    timestamps = [_parse_timestamp(record) for record in ordered]
    valid_timestamps = [timestamp for timestamp in timestamps if timestamp is not None]

    if len(valid_timestamps) >= 2:
        latest = max(valid_timestamps)
        earliest = min(valid_timestamps)
        span_days = max((latest - earliest).days, 0)
        if span_days >= 56:
            window_days = 28
            mode = "monthly"
        elif span_days >= 14:
            window_days = 7
            mode = "weekly"
        else:
            window_days = 0
            mode = "ordered"

        if window_days:
            current_cutoff = latest - timedelta(days=window_days - 1)
            previous_cutoff = current_cutoff - timedelta(days=window_days)
            current = [
                record for record in ordered
                if (timestamp := _parse_timestamp(record)) is not None and timestamp >= current_cutoff
            ]
            previous = [
                record for record in ordered
                if (timestamp := _parse_timestamp(record)) is not None
                and previous_cutoff <= timestamp < current_cutoff
            ]
            if current and previous:
                return current, previous, mode

    chunk_size = max(len(ordered) // 3, 1)
    current = ordered[-chunk_size:]
    previous = ordered[-(chunk_size * 2):-chunk_size]
    if not previous:
        midpoint = len(ordered) // 2
        previous = ordered[:midpoint]
        current = ordered[midpoint:]
    return current, previous, "ordered"


def _safe_percentage_change(current_value: float, previous_value: float) -> float:
    if previous_value <= 0:
        return 100.0 if current_value > 0 else 0.0
    return ((current_value - previous_value) / previous_value) * 100.0


def _trend_direction(absolute_change: float, stable_threshold: float) -> str:
    if abs(absolute_change) < stable_threshold:
        return "stable"
    return "rising" if absolute_change > 0 else "falling"


def _confidence_payload(sample_size: int, absolute_change: float, scale: float = 0.1) -> tuple[float, str]:
    confidence = min(
        1.0,
        0.65 * min(sample_size / 60, 1.0)
        + 0.35 * min(abs(absolute_change) / max(scale, 0.001), 1.0),
    )
    if confidence >= 0.7:
        return round(confidence, 3), "High"
    if confidence >= 0.4:
        return round(confidence, 3), "Medium"
    return round(confidence, 3), "Low"


def _count_entities(records: list[dict[str, Any]], extractor) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        values = extractor(record)
        counter.update({value for value in values if value})
    return counter


def _build_named_trends(
    current_records: list[dict[str, Any]],
    previous_records: list[dict[str, Any]],
    extractor,
    stable_threshold: float,
    limit: int = 6,
) -> list[dict[str, Any]]:
    if not current_records or not previous_records:
        return []

    current_counter = _count_entities(current_records, extractor)
    previous_counter = _count_entities(previous_records, extractor)
    current_sample_size = max(len(current_records), 1)
    previous_sample_size = max(len(previous_records), 1)

    results = []
    for name in sorted(set(current_counter) | set(previous_counter)):
        current_percentage = (current_counter.get(name, 0) / current_sample_size) * 100.0
        previous_percentage = (previous_counter.get(name, 0) / previous_sample_size) * 100.0
        absolute_change = current_percentage - previous_percentage
        percentage_change = _safe_percentage_change(current_percentage, previous_percentage)
        results.append({
            "name": name,
            "currentPercentage": round(current_percentage, 1),
            "previousPercentage": round(previous_percentage, 1),
            "absoluteChange": round(absolute_change, 1),
            "percentageChange": round(percentage_change, 1),
            "trendDirection": _trend_direction(absolute_change, stable_threshold),
            "currentCount": int(current_counter.get(name, 0)),
            "previousCount": int(previous_counter.get(name, 0)),
        })

    results.sort(
        key=lambda entry: (
            abs(float(entry["absoluteChange"])),
            float(entry["currentPercentage"]),
            entry["currentCount"],
        ),
        reverse=True,
    )
    return results[:limit]


def _build_usage_trend(
    pokemon_name: str,
    current_teams: list[dict[str, Any]],
    previous_teams: list[dict[str, Any]],
    current_pokemon_teams: list[dict[str, Any]],
    previous_pokemon_teams: list[dict[str, Any]],
) -> dict[str, Any]:
    current_rate = len(current_pokemon_teams) / max(len(current_teams), 1)
    previous_rate = len(previous_pokemon_teams) / max(len(previous_teams), 1)
    absolute_change = current_rate - previous_rate
    percentage_change = _safe_percentage_change(current_rate, previous_rate)
    trend_direction = _trend_direction(absolute_change, 0.01)
    confidence, confidence_label = _confidence_payload(
        len(current_pokemon_teams) + len(previous_pokemon_teams),
        absolute_change,
        scale=0.08,
    )
    if trend_direction == "rising":
        explanation = f"{pokemon_name} usage increased compared with the previous tournament period."
    elif trend_direction == "falling":
        explanation = f"{pokemon_name} usage decreased compared with the previous tournament period."
    else:
        explanation = f"{pokemon_name} usage stayed relatively stable between tournament periods."

    return {
        "currentUsageRate": round(current_rate, 3),
        "previousUsageRate": round(previous_rate, 3),
        "absoluteChange": round(absolute_change, 3),
        "percentageChange": round(percentage_change, 1),
        "trendDirection": trend_direction,
        "confidence": confidence,
        "confidenceLabel": confidence_label,
        "sampleSize": len(current_pokemon_teams) + len(previous_pokemon_teams),
        "explanation": explanation,
    }


def _build_win_rate_trend(
    current_records: list[dict[str, Any]],
    previous_records: list[dict[str, Any]],
    data_quality_notes: list[str],
) -> dict[str, Any]:
    def aggregate(records: list[dict[str, Any]]) -> tuple[int, int]:
        wins = 0
        losses = 0
        for record in records:
            record_wins = record.get("wins")
            record_losses = record.get("losses")
            if isinstance(record_wins, int) and isinstance(record_losses, int):
                wins += record_wins
                losses += record_losses
        return wins, losses

    current_wins, current_losses = aggregate(current_records)
    previous_wins, previous_losses = aggregate(previous_records)
    current_games = current_wins + current_losses
    previous_games = previous_wins + previous_losses

    if current_games == 0 or previous_games == 0:
        data_quality_notes.append(
            "Win-rate trend was unavailable because complete win/loss data was missing for one or both comparison periods."
        )
        return {}

    current_rate = current_wins / current_games
    previous_rate = previous_wins / previous_games
    absolute_change = current_rate - previous_rate
    confidence, confidence_label = _confidence_payload(current_games + previous_games, absolute_change, scale=0.06)
    return {
        "currentWinRate": round(current_rate, 3),
        "previousWinRate": round(previous_rate, 3),
        "absoluteChange": round(absolute_change, 3),
        "percentageChange": round(_safe_percentage_change(current_rate, previous_rate), 1),
        "trendDirection": _trend_direction(absolute_change, 0.01),
        "confidence": confidence,
        "confidenceLabel": confidence_label,
        "sampleSize": current_games + previous_games,
        "explanation": "Win rate compares tournament match results between the recent and previous periods.",
    }


def _core_key(core_members: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(sorted(core_members))


def _core_entries(teams: list[dict[str, Any]], pokemon_name: str) -> Counter[tuple[str, ...]]:
    counter: Counter[tuple[str, ...]] = Counter()
    target_aliases = _build_aliases(pokemon_name)

    for team in teams:
        normalized_team = [normalize_name(member) for member in team.get("pokemon", []) if normalize_name(member)]
        if not (set(normalized_team) & target_aliases):
            continue

        display_lookup = {
            normalize_name(member): member
            for member in team.get("pokemon", [])
            if normalize_name(member)
        }
        teammates = [
            display_lookup[name]
            for name in normalized_team
            if name not in target_aliases
        ]
        for pair in combinations(sorted(set(teammates)), 2):
            counter[_core_key((pokemon_name, *pair))] += 1

    return counter


def _format_core_entry(core: tuple[str, ...], count: int, denominator: int, explanation: str) -> dict[str, Any]:
    percentage = (count / max(denominator, 1)) * 100.0
    return {
        "corePokemon": list(core),
        "count": int(count),
        "percentage": round(percentage, 1),
        "explanation": explanation,
    }


def _build_core_sections(
    pokemon_name: str,
    current_teams: list[dict[str, Any]],
    previous_teams: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    current_counter = _core_entries(current_teams, pokemon_name)
    previous_counter = _core_entries(previous_teams, pokemon_name)
    current_denominator = max(len(current_teams), 1)
    previous_denominator = max(len(previous_teams), 1)

    common_cores = [
        _format_core_entry(
            core,
            count,
            current_denominator,
            f"This core appeared in {count} recent tournament teams featuring {pokemon_name}.",
        )
        for core, count in current_counter.most_common(5)
    ]

    new_core_candidates = []
    for core, count in current_counter.items():
        current_percentage = (count / current_denominator) * 100.0
        previous_percentage = (previous_counter.get(core, 0) / previous_denominator) * 100.0
        absolute_change = current_percentage - previous_percentage
        if count <= 0 or absolute_change < 4.0:
            continue
        new_core_candidates.append({
            **_format_core_entry(
                core,
                count,
                current_denominator,
                f"This core is gaining traction around {pokemon_name} in the recent tournament period.",
            ),
            "previousPercentage": round(previous_percentage, 1),
            "absoluteChange": round(absolute_change, 1),
        })

    new_core_candidates.sort(
        key=lambda entry: (entry["absoluteChange"], entry["count"]),
        reverse=True,
    )
    return common_cores, new_core_candidates[:5]


def _compute_pokemon_meta_intelligence(pokemon_name: str) -> dict[str, Any]:
    pokemon = get_pokedex_entry_by_name(pokemon_name)
    if pokemon is None:
        raise ValueError(f"Pokemon not found: {pokemon_name}")

    canonical_name = pokemon.get("speciesName") or pokemon.get("name") or pokemon_name
    result = _empty_result(canonical_name)

    team_records = load_meta_team_records()
    pokemon_team_records = _match_target_teams(canonical_name)
    pokemon_set_records = _match_target_records(canonical_name)

    if not pokemon_team_records or not pokemon_set_records:
        result["dataQualityNotes"].append(
            f"No tournament data was available for {canonical_name}."
        )
        result["usageTrend"]["explanation"] = (
            f"No tournament data was available to evaluate {canonical_name}."
        )
        return result

    current_teams, previous_teams, period_mode = _split_periods(team_records)
    current_team_ids = {team.get("team_id") for team in current_teams}
    previous_team_ids = {team.get("team_id") for team in previous_teams}

    current_pokemon_teams = [
        team for team in pokemon_team_records if team.get("team_id") in current_team_ids
    ]
    previous_pokemon_teams = [
        team for team in pokemon_team_records if team.get("team_id") in previous_team_ids
    ]
    current_records = [
        record for record in pokemon_set_records if record.get("team_id") in current_team_ids
    ]
    previous_records = [
        record for record in pokemon_set_records if record.get("team_id") in previous_team_ids
    ]

    if period_mode == "ordered":
        result["dataQualityNotes"].append(
            "Tournament date coverage was limited, so trends used tournament order as a fallback."
        )

    if len(current_records) + len(previous_records) < 12:
        result["dataQualityNotes"].append(
            "Low sample size. Treat these trend shifts as directional rather than definitive."
        )

    result["usageTrend"] = _build_usage_trend(
        canonical_name,
        current_teams,
        previous_teams,
        current_pokemon_teams,
        previous_pokemon_teams,
    )

    result["moveTrends"] = _build_named_trends(
        current_records,
        previous_records,
        lambda record: record.get("moves", []),
        stable_threshold=4.0,
    )
    result["itemTrends"] = _build_named_trends(
        current_records,
        previous_records,
        lambda record: [record.get("item")] if record.get("item") else [],
        stable_threshold=3.0,
    )
    result["archetypeTrends"] = _build_named_trends(
        current_records,
        previous_records,
        lambda record: [record.get("archetype")] if record.get("archetype") else [],
        stable_threshold=4.0,
    )
    result["winRateTrend"] = _build_win_rate_trend(
        current_records,
        previous_records,
        result["dataQualityNotes"],
    )
    common_cores, new_cores = _build_core_sections(
        canonical_name,
        current_pokemon_teams or pokemon_team_records,
        previous_pokemon_teams,
    )
    result["commonCores"] = common_cores
    result["newCores"] = new_cores

    if not result["moveTrends"]:
        result["dataQualityNotes"].append(
            "Move trend data was limited for the current comparison window."
        )
    if not result["itemTrends"]:
        result["dataQualityNotes"].append(
            "Item trend data was limited for the current comparison window."
        )
    if not result["archetypeTrends"]:
        result["dataQualityNotes"].append(
            "Archetype trend data was limited for the current comparison window."
        )
    if not result["commonCores"]:
        result["dataQualityNotes"].append(
            "Core trend data was limited for the current comparison window."
        )

    return result


def build_pokedex_meta_intelligence_cache() -> dict[str, dict[str, Any]]:
    meta_by_pokemon: dict[str, dict[str, Any]] = {}
    seen_species: set[str] = set()

    for pokemon in get_pokedex_grid_entries():
        species_name = pokemon.get("speciesName") or pokemon.get("name") or ""
        normalized_species = _cache_key(species_name)
        if not normalized_species or normalized_species in seen_species:
            continue

        seen_species.add(normalized_species)
        meta_by_pokemon[species_name] = _compute_pokemon_meta_intelligence(species_name)

    return dict(sorted(meta_by_pokemon.items(), key=lambda item: _cache_key(item[0])))


def write_pokedex_meta_intelligence_cache() -> dict[str, dict[str, Any]]:
    POKEDEX_META_INTELLIGENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = build_pokedex_meta_intelligence_cache()
    with open(POKEDEX_META_INTELLIGENCE_PATH, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)

    load_meta_intelligence_cache.cache_clear()
    return payload


@lru_cache(maxsize=256)
def get_pokemon_meta_intelligence(pokemon_name: str) -> dict[str, Any]:
    cache_key = _cache_key(pokemon_name)
    cached_payload = load_meta_intelligence_cache().get(cache_key)
    if cached_payload is not None:
        return cached_payload

    return _compute_pokemon_meta_intelligence(pokemon_name)
