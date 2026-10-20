"""Cached tournament-backed recommended builds for the Pokedex Teams tab."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from paths import POKEMON_RECOMMENDED_BUILDS_PATH
from services.battle_advice_cache_service import load_set_probability_summary_cache
from services.tournament_usage_service import get_pokemon_usage_summary, load_usage_summary_data


def normalize_name(value: Any) -> str:
    return str(value or "").strip().lower()


def display_name(value: Any) -> str:
    normalized = normalize_name(value)
    if not normalized:
        return ""
    return " ".join(part.capitalize() for part in normalized.replace("-", " ").split())


def _empty_payload(pokemon_name: str) -> dict[str, Any]:
    return {
        "pokemon": pokemon_name,
        "recommendedTeams": [],
        "dataQualityNotes": [],
    }


def _team_signature(team_members: list[str]) -> tuple[str, ...]:
    return tuple(sorted(normalize_name(member) for member in team_members if normalize_name(member)))


def _safe_win_rate(wins: Any, losses: Any) -> float:
    wins_value = int(wins or 0)
    losses_value = int(losses or 0)
    games = wins_value + losses_value
    if games <= 0:
        return 0.0
    return round(wins_value / games, 3)


def _top_set_details(pokemon_name: str) -> dict[str, Any]:
    summary = load_set_probability_summary_cache().get(normalize_name(pokemon_name), {})
    moves = [entry.get("name", "") for entry in summary.get("moves", [])[:4] if entry.get("name")]
    item = next((entry.get("name", "") for entry in summary.get("items", []) if entry.get("name")), "")
    ability = next((entry.get("name", "") for entry in summary.get("abilities", []) if entry.get("name")), "")
    nature = next((entry.get("name", "") for entry in summary.get("natures", []) if entry.get("name")), "")
    teammates = {
        normalize_name(entry.get("name")): entry
        for entry in summary.get("commonTeammates", [])
        if entry.get("name")
    }
    return {
        "item": item,
        "ability": ability,
        "nature": nature,
        "moves": moves,
        "commonTeammates": teammates,
        "sampleSize": int(summary.get("sampleSize", 0) or 0),
    }


def _build_signal_entries(anchor_name: str, team_members: list[str], set_details: dict[str, Any]) -> list[dict[str, Any]]:
    signals = []
    teammate_lookup = set_details.get("commonTeammates", {})
    for teammate in team_members:
        if normalize_name(teammate) == normalize_name(anchor_name):
            continue
        teammate_entry = teammate_lookup.get(normalize_name(teammate), {})
        signals.append(
            {
                "name": display_name(teammate),
                "count": int(teammate_entry.get("count", 0) or 0),
                "percentage": float(teammate_entry.get("percentage", 0.0) or 0.0),
            }
        )
    signals.sort(
        key=lambda entry: (float(entry["percentage"]), int(entry["count"]), entry["name"]),
        reverse=True,
    )
    return signals


def _summarize_team_build(
    pokemon_name: str,
    team: dict[str, Any],
    index: int,
    set_details: dict[str, Any],
) -> dict[str, Any]:
    team_members = [display_name(member) for member in team.get("pokemon", []) if member]
    source_wins = int(team.get("wins", 0) or 0)
    source_losses = int(team.get("losses", 0) or 0)
    win_rate = _safe_win_rate(source_wins, source_losses)
    signals = _build_signal_entries(pokemon_name, team_members, set_details)

    summary_parts = []
    if signals:
        top_names = ", ".join(signal["name"] for signal in signals[:3])
        summary_parts.append(f"Built around common partners like {top_names}.")
    if set_details.get("item"):
        summary_parts.append(f"Most common tournament item is {set_details['item']}.")
    if set_details.get("moves"):
        summary_parts.append(
            f"Typical move package includes {', '.join(set_details['moves'][:3])}."
        )

    return {
        "label": [
            "Most Proven Tournament Shell",
            "High-Finish Variant",
            "Alternate Tournament Variant",
        ][min(index, 2)],
        "pokemon": team_members[:6],
        "sampleSize": max(1, int(team.get("sourceCount", 1) or 1)),
        "winRate": win_rate,
        "wins": source_wins,
        "losses": source_losses,
        "tournament": team.get("tournament", "") or "Tournament results",
        "player": team.get("player", "") or "Unknown Player",
        "signals": signals[:5],
        "recommendedSet": {
            "item": set_details.get("item", ""),
            "ability": set_details.get("ability", ""),
            "nature": set_details.get("nature", ""),
            "moves": list(set_details.get("moves", []))[:4],
            "sampleSize": int(set_details.get("sampleSize", 0) or 0),
        },
        "summary": " ".join(summary_parts).strip(),
    }


def _dedupe_top_teams(top_teams: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, Any]] = {}
    for team in top_teams:
        signature = _team_signature(team.get("pokemon", []))
        if not signature:
            continue
        existing = grouped.get(signature)
        if existing is None:
            grouped[signature] = {
                **team,
                "sourceCount": 1,
            }
            continue

        existing["wins"] = int(existing.get("wins", 0) or 0) + int(team.get("wins", 0) or 0)
        existing["losses"] = int(existing.get("losses", 0) or 0) + int(team.get("losses", 0) or 0)
        existing["sourceCount"] = int(existing.get("sourceCount", 1) or 1) + 1

    ranked = sorted(
        grouped.values(),
        key=lambda entry: (
            _safe_win_rate(entry.get("wins", 0), entry.get("losses", 0)),
            int(entry.get("wins", 0) or 0),
            int(entry.get("sourceCount", 1) or 1),
        ),
        reverse=True,
    )
    return ranked[:3]


def _compute_recommended_builds_for_pokemon(pokemon_name: str) -> dict[str, Any]:
    usage_summary = get_pokemon_usage_summary(pokemon_name)
    top_teams = list(usage_summary.get("topTeams", []))
    if not top_teams:
        return {
            **_empty_payload(display_name(pokemon_name)),
            "dataQualityNotes": ["No tournament team data was available for this Pokemon."],
        }

    set_details = _top_set_details(pokemon_name)
    ranked_teams = _dedupe_top_teams(top_teams)
    recommended = [
        _summarize_team_build(display_name(pokemon_name), team, index, set_details)
        for index, team in enumerate(ranked_teams)
    ]

    notes: list[str] = []
    if not set_details.get("sampleSize"):
        notes.append("Set recommendation details were limited, so builds rely mostly on team results.")
    if len(top_teams) < 3:
        notes.append("Only a small number of tournament lineups were available for this Pokemon.")

    return {
        "pokemon": display_name(pokemon_name),
        "recommendedTeams": recommended,
        "dataQualityNotes": notes,
    }


def build_recommended_builds_cache() -> dict[str, dict[str, Any]]:
    usage_summary = load_usage_summary_data()
    payload: dict[str, dict[str, Any]] = {}
    for pokemon_name in usage_summary:
        payload[pokemon_name] = _compute_recommended_builds_for_pokemon(pokemon_name)
    return payload


def write_recommended_builds_cache() -> dict[str, dict[str, Any]]:
    payload = build_recommended_builds_cache()
    POKEMON_RECOMMENDED_BUILDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    POKEMON_RECOMMENDED_BUILDS_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    load_recommended_builds_cache.cache_clear()
    return payload


@lru_cache(maxsize=1)
def load_recommended_builds_cache() -> dict[str, dict[str, Any]]:
    if not POKEMON_RECOMMENDED_BUILDS_PATH.exists():
        return {}

    payload = json.loads(POKEMON_RECOMMENDED_BUILDS_PATH.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}

    normalized_payload: dict[str, dict[str, Any]] = {}
    for pokemon_name, entry in payload.items():
        normalized_key = normalize_name(pokemon_name)
        if normalized_key and isinstance(entry, dict):
            normalized_payload[normalized_key] = entry
    return normalized_payload


def get_pokemon_recommended_builds_by_name(pokemon_name: str) -> dict[str, Any]:
    if not pokemon_name:
        return _empty_payload("")

    cached = load_recommended_builds_cache().get(normalize_name(pokemon_name))
    if cached is not None:
        return cached
    return _compute_recommended_builds_for_pokemon(pokemon_name)
