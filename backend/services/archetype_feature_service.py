"""Feature extraction layer for hybrid team archetype detection.

This module builds one normalized record per tournament team, then extracts
explainable team-level features from Pokemon stats plus available moves,
items, abilities, and tournament metadata.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Any

import pandas as pd

from services.pokemon_data_service import load_pokemon_records, lookup_keys_for_value
from services.team_archetype_service import load_tournament_team_rows


SETUP_MOVES = {
    "swords dance",
    "nasty plot",
    "dragon dance",
    "quiver dance",
    "calm mind",
    "bulk up",
    "curse",
    "belly drum",
    "shell smash",
    "iron defense",
    "amnesia",
    "trailblaze",
    "howl",
    "shift gear",
    "agility",
}

RECOVERY_MOVES = {
    "recover",
    "roost",
    "slack off",
    "moonlight",
    "morning sun",
    "soft boiled",
    "synthesis",
    "wish",
    "shore up",
    "milk drink",
    "rest",
    "strength sap",
    "leech seed",
    "giga drain",
    "drain punch",
}

STATUS_MOVES = {
    "will o wisp",
    "thunder wave",
    "spore",
    "sleep powder",
    "stun spore",
    "toxic",
    "taunt",
    "encore",
    "disable",
    "yawn",
    "parting shot",
    "snarl",
    "fake tears",
    "charm",
    "tickle",
    "haze",
    "follow me",
    "rage powder",
}

PRIORITY_MOVES = {
    "fake out",
    "extreme speed",
    "quick attack",
    "bullet punch",
    "mach punch",
    "vacuum wave",
    "ice shard",
    "grassy glide",
    "shadow sneak",
    "sucker punch",
    "aqua jet",
    "jet punch",
    "first impression",
}

SPEED_CONTROL_MOVES = {
    "tailwind",
    "trick room",
    "icy wind",
    "electroweb",
    "thunder wave",
    "bulldoze",
    "rock tomb",
    "scary face",
    "string shot",
}

TRICK_ROOM_SETTER_POKEMON = {
    "cresselia",
    "porygon2",
    "hatterene",
    "indeedee",
    "indeedee f",
    "farigiraf",
    "dusclops",
    "bronzong",
    "oranguru",
    "mimikyu",
    "diancie",
}

TAILWIND_SETTER_POKEMON = {
    "whimsicott",
    "tornadus",
    "talonflame",
    "murkrow",
    "aerodactyl",
    "pelipper",
    "zapdos",
    "crobat",
    "mandibuzz",
    "kilowattrel",
}

WEATHER_SETTER_ABILITIES = {
    "drizzle",
    "drought",
    "sand stream",
    "snow warning",
}

WEATHER_ABUSER_ABILITIES = {
    "swift swim",
    "chlorophyll",
    "sand rush",
    "slush rush",
    "solar power",
    "sand force",
    "ice body",
    "rain dish",
    "hydration",
    "protosynthesis",
}

WEATHER_MOVES = {
    "rain dance",
    "sunny day",
    "sandstorm",
    "snowscape",
    "hail",
}

PIVOT_MOVES = {
    "u turn",
    "volt switch",
    "flip turn",
    "parting shot",
    "baton pass",
    "teleport",
    "chilly reception",
}

RAIN_PAYOFF_MOVES = {
    "thunder",
    "hurricane",
    "electro shot",
    "weather ball",
}

SUN_PAYOFF_MOVES = {
    "solar beam",
    "solarblade",
    "solar blade",
    "weather ball",
    "heat wave",
    "eruption",
    "burn up",
}

SAND_PAYOFF_MOVES = {
    "weather ball",
    "shore up",
    "rock slide",
}

SNOW_PAYOFF_MOVES = {
    "aurora veil",
    "blizzard",
    "weather ball",
}

DEFENSIVE_ITEMS = {
    "leftovers",
    "sitrus berry",
    "assault vest",
    "safety goggles",
    "rocky helmet",
    "eviolite",
    "mental herb",
    "aguav berry",
    "figy berry",
    "wiki berry",
    "iapapa berry",
    "mago berry",
}

OFFENSIVE_ITEMS = {
    "life orb",
    "expert belt",
    "choice band",
    "choice specs",
    "choice scarf",
    "loaded dice",
    "booster energy",
    "weakness policy",
    "clear amulet",
}

CHOICE_ITEMS = {"choice band", "choice specs", "choice scarf"}


def normalize_name(value: Any) -> str:
    """Normalize Pokemon, move, item, ability, or nature names."""
    text = str(value or "").strip().lower()
    if not text or text in {"none", "nan", "null", "n/a"}:
        return ""

    replacements = {
        "â€™": "'",
        "’": "'",
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


def _safe_int(value: Any) -> int | None:
    if pd.isna(value):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_move_list(values: list[Any]) -> list[str]:
    return [normalized for normalized in (normalize_name(value) for value in values) if normalized]


def _coerce_member_move_lists(values: Any, member_count: int) -> list[list[str]]:
    if not member_count:
        return []

    if not isinstance(values, list):
        return [[] for _ in range(member_count)]

    if values and any(isinstance(entry, list) for entry in values):
        normalized_lists: list[list[str]] = []
        for index in range(member_count):
            entry = values[index] if index < len(values) else []
            if isinstance(entry, list):
                normalized_lists.append(_normalize_move_list(entry))
            elif entry:
                normalized_lists.append(_normalize_move_list([entry]))
            else:
                normalized_lists.append([])
        return normalized_lists

    flat_moves = _normalize_move_list(values)
    if not flat_moves:
        return [[] for _ in range(member_count)]

    distributed = [[] for _ in range(member_count)]
    for move_index, move in enumerate(flat_moves):
        distributed[move_index % member_count].append(move)
    return distributed


def _normalize_type_list(values: list[Any]) -> list[str]:
    return [normalized for normalized in (normalize_name(value) for value in values) if normalized]


@lru_cache(maxsize=1)
def load_pokemon_stat_lookup() -> dict[str, dict[str, Any]]:
    """Load Pokemon stat/type/ability data keyed by normalized lookup names."""
    lookup: dict[str, dict[str, Any]] = {}

    for record in load_pokemon_records():
        entry = {
            "species_name": record.get("species_display_name") or record.get("species_api_name") or "",
            "form_name": record.get("form_display_name") or record.get("form_api_name") or "",
            "types": _normalize_type_list(record.get("types", [])),
            "base_stats": {
                "hp": int((record.get("base_stats") or {}).get("hp", 0) or 0),
                "attack": int((record.get("base_stats") or {}).get("attack", 0) or 0),
                "defense": int((record.get("base_stats") or {}).get("defense", 0) or 0),
                "special_attack": int((record.get("base_stats") or {}).get("special_attack", 0) or 0),
                "special_defense": int((record.get("base_stats") or {}).get("special_defense", 0) or 0),
                "speed": int((record.get("base_stats") or {}).get("speed", 0) or 0),
            },
            "abilities": [
                normalize_name(ability.get("display_name") or ability.get("name"))
                for ability in record.get("abilities", [])
                if ability
            ],
        }

        possible_names = [
            record.get("species_display_name"),
            record.get("form_display_name"),
            record.get("species_api_name"),
            record.get("form_api_name"),
        ]

        for name in possible_names:
            for key in lookup_keys_for_value(name):
                normalized_key = normalize_name(key)
                if normalized_key and normalized_key not in lookup:
                    lookup[normalized_key] = entry

    return lookup


def _find_pokemon_lookup_entry(pokemon_name: str) -> dict[str, Any] | None:
    lookup = load_pokemon_stat_lookup()
    normalized = normalize_name(pokemon_name)
    if normalized in lookup:
        return lookup[normalized]

    compact = "".join(character for character in normalized if character.isalnum())
    for key, value in lookup.items():
        if "".join(character for character in key if character.isalnum()) == compact:
            return value

    return None


def _normalize_member_record(row: pd.Series) -> dict[str, Any]:
    moves = _normalize_move_list(
        [row.get("move_1"), row.get("move_2"), row.get("move_3"), row.get("move_4")]
    )
    return {
        "slot": _safe_int(row.get("team_slot")),
        "pokemon_name": normalize_name(row.get("pokemon")),
        "pokemon_display_name": str(row.get("pokemon") or "").strip(),
        "item": normalize_name(row.get("item")),
        "ability": normalize_name(row.get("ability")),
        "moves": moves,
        "tera_type": normalize_name(row.get("tera_type")),
    }


def _build_team_record_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    pokemon = payload.get("pokemon") or []
    raw_items = payload.get("items", []) or []
    raw_abilities = payload.get("abilities", []) or []

    items = [normalize_name(value) for value in raw_items if normalize_name(value)] if isinstance(raw_items, list) else []
    abilities = [normalize_name(value) for value in raw_abilities if normalize_name(value)] if isinstance(raw_abilities, list) else []
    member_moves = _coerce_member_move_lists(payload.get("moves", []), len(pokemon))
    members = []

    for index, pokemon_name in enumerate(pokemon, start=1):
        members.append(
            {
                "slot": index,
                "pokemon_name": normalize_name(pokemon_name),
                "pokemon_display_name": str(pokemon_name or "").strip(),
                "item": normalize_name(raw_items[index - 1]) if isinstance(raw_items, list) and index - 1 < len(raw_items) else "",
                "ability": normalize_name(raw_abilities[index - 1]) if isinstance(raw_abilities, list) and index - 1 < len(raw_abilities) else "",
                "moves": member_moves[index - 1] if index - 1 < len(member_moves) else [],
                "tera_type": "",
            }
        )

    data_quality_notes: list[str] = []
    if any(not member["moves"] for member in members):
        data_quality_notes.append("missing moves")
    if any(not member["item"] for member in members):
        data_quality_notes.append("missing items")
    if any(not member["ability"] for member in members):
        data_quality_notes.append("missing abilities")

    return {
        "team_id": "manual::input",
        "tournament_id": "",
        "player_id": "",
        "player_name": "manual",
        "tournament_name": "",
        "tournament_date": None,
        "placement": None,
        "wins": None,
        "losses": None,
        "player_record": None,
        "team_win_rate": None,
        "members": members,
        "pokemon_names": [member["pokemon_name"] for member in members if member["pokemon_name"]],
        "items": [member["item"] for member in members if member["item"]],
        "abilities": [member["ability"] for member in members if member["ability"]],
        "moves": [move for member in members for move in member["moves"]],
        "data_quality_notes": sorted(set(data_quality_notes + ["missing date", "missing win/loss data"])),
    }


def load_tournament_teams() -> list[dict[str, Any]]:
    """Load and normalize one team record per tournament team."""
    team_rows = load_tournament_team_rows()
    if team_rows.empty:
        return []

    normalized_records: list[dict[str, Any]] = []
    grouped = team_rows.groupby(["tournament_id", "player_id"], dropna=True, sort=False)

    for team_key, group in grouped:
        group = group.copy()
        if "team_slot" in group.columns:
            group["_slot_sort"] = pd.to_numeric(group["team_slot"], errors="coerce")
            group = group.sort_values(by=["_slot_sort", "pokemon"], na_position="last")

        first_row = group.iloc[0]
        members = [_normalize_member_record(row) for _, row in group.iterrows()]
        pokemon_names = [member["pokemon_name"] for member in members if member["pokemon_name"]]
        items = [member["item"] for member in members if member["item"]]
        abilities = [member["ability"] for member in members if member["ability"]]
        moves = [move for member in members for move in member["moves"]]

        wins = _safe_int(first_row.get("wins"))
        losses = _safe_int(first_row.get("losses"))
        placing = _safe_int(first_row.get("placing"))
        tournament_date = pd.to_datetime(first_row.get("tournament_date"), errors="coerce", utc=True)
        tournament_date_iso = None if pd.isna(tournament_date) else tournament_date.isoformat()
        team_win_rate = None
        if wins is not None and losses is not None and wins + losses > 0:
            team_win_rate = round(wins / (wins + losses), 4)

        data_quality_notes: list[str] = []
        if any(not member["moves"] for member in members):
            data_quality_notes.append("missing moves")
        if any(not member["item"] for member in members):
            data_quality_notes.append("missing items")
        if any(not member["ability"] for member in members):
            data_quality_notes.append("missing abilities")
        if tournament_date_iso is None:
            data_quality_notes.append("missing date")
        if wins is None or losses is None:
            data_quality_notes.append("missing win/loss data")

        normalized_records.append(
            {
                "team_id": f"{team_key[0]}::{team_key[1]}",
                "tournament_id": str(team_key[0] or ""),
                "player_id": str(team_key[1] or ""),
                "player_name": first_row.get("player_name"),
                "tournament_name": first_row.get("tournament_name"),
                "tournament_date": tournament_date_iso,
                "placement": placing,
                "wins": wins,
                "losses": losses,
                "player_record": None if wins is None or losses is None else f"{wins}-{losses}",
                "team_win_rate": team_win_rate,
                "members": members,
                "pokemon_names": pokemon_names,
                "items": items,
                "abilities": abilities,
                "moves": moves,
                "data_quality_notes": sorted(set(data_quality_notes)),
            }
        )

    return normalized_records


def _average(values: list[float]) -> float:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return 0.0
    return round(sum(usable) / len(usable), 3)


def _sum_or_zero(*values: int) -> int:
    return sum(int(value or 0) for value in values)


def extract_team_features(team_record: dict[str, Any]) -> dict[str, Any]:
    """Extract explainable team-level features from one normalized team record."""
    if "members" not in team_record and "pokemon" in team_record:
        team_record = _build_team_record_from_payload(team_record)

    type_counter: Counter[str] = Counter()
    notes = list(team_record.get("data_quality_notes", []))

    hp_values: list[float] = []
    attack_values: list[float] = []
    defense_values: list[float] = []
    sp_attack_values: list[float] = []
    sp_defense_values: list[float] = []
    speed_values: list[float] = []
    bulk_values: list[float] = []

    physical_attacker_count = 0
    special_attacker_count = 0
    mixed_attacker_count = 0
    fast_pokemon_count = 0
    slow_pokemon_count = 0
    bulky_pokemon_count = 0

    setup_move_count = 0
    recovery_move_count = 0
    status_move_count = 0
    priority_move_count = 0
    speed_control_count = 0
    trick_room_move_count = 0
    tailwind_move_count = 0
    weather_move_count = 0
    protect_count = 0
    pivot_move_count = 0

    weather_setter_count = 0
    weather_abuser_count = 0
    trick_room_setter_count = 0
    tailwind_setter_count = 0
    focus_sash_count = 0
    choice_item_count = 0
    defensive_item_count = 0
    offensive_item_count = 0

    rain_payoff_move_count = 0
    sun_payoff_move_count = 0
    sand_payoff_move_count = 0
    snow_payoff_move_count = 0

    members_with_missing_stats = 0

    for member in team_record.get("members", []):
        pokemon_entry = _find_pokemon_lookup_entry(member.get("pokemon_name", ""))
        if not pokemon_entry:
            members_with_missing_stats += 1
            continue

        stats = pokemon_entry["base_stats"]
        hp = int(stats.get("hp", 0) or 0)
        attack = int(stats.get("attack", 0) or 0)
        defense = int(stats.get("defense", 0) or 0)
        sp_attack = int(stats.get("special_attack", 0) or 0)
        sp_defense = int(stats.get("special_defense", 0) or 0)
        speed = int(stats.get("speed", 0) or 0)
        bulk = hp + defense + sp_defense

        hp_values.append(hp)
        attack_values.append(attack)
        defense_values.append(defense)
        sp_attack_values.append(sp_attack)
        sp_defense_values.append(sp_defense)
        speed_values.append(speed)
        bulk_values.append(bulk)

        for pokemon_type in pokemon_entry.get("types", []):
            type_counter[pokemon_type] += 1

        if attack >= 100 and attack >= sp_attack + 15:
            physical_attacker_count += 1
        elif sp_attack >= 100 and sp_attack >= attack + 15:
            special_attacker_count += 1
        elif max(attack, sp_attack) >= 95:
            mixed_attacker_count += 1

        if speed >= 100:
            fast_pokemon_count += 1
        if speed <= 70:
            slow_pokemon_count += 1
        if bulk >= 280:
            bulky_pokemon_count += 1

        ability = member.get("ability", "")
        moves = set(member.get("moves", []))
        item = member.get("item", "")

        setup_move_count += sum(1 for move in moves if move in SETUP_MOVES)
        recovery_move_count += sum(1 for move in moves if move in RECOVERY_MOVES)
        status_move_count += sum(1 for move in moves if move in STATUS_MOVES)
        priority_move_count += sum(1 for move in moves if move in PRIORITY_MOVES)
        speed_control_count += sum(1 for move in moves if move in SPEED_CONTROL_MOVES)
        trick_room_move_count += 1 if "trick room" in moves else 0
        tailwind_move_count += 1 if "tailwind" in moves else 0
        weather_move_count += sum(1 for move in moves if move in WEATHER_MOVES)
        protect_count += 1 if "protect" in moves else 0
        pivot_move_count += sum(1 for move in moves if move in PIVOT_MOVES)

        rain_payoff_move_count += sum(1 for move in moves if move in RAIN_PAYOFF_MOVES)
        sun_payoff_move_count += sum(1 for move in moves if move in SUN_PAYOFF_MOVES)
        sand_payoff_move_count += sum(1 for move in moves if move in SAND_PAYOFF_MOVES)
        snow_payoff_move_count += sum(1 for move in moves if move in SNOW_PAYOFF_MOVES)

        weather_setter_count += 1 if ability in WEATHER_SETTER_ABILITIES else 0
        weather_abuser_count += 1 if ability in WEATHER_ABUSER_ABILITIES else 0
        trick_room_setter_count += 1 if "trick room" in moves or member.get("pokemon_name") in TRICK_ROOM_SETTER_POKEMON else 0
        tailwind_setter_count += 1 if "tailwind" in moves or member.get("pokemon_name") in TAILWIND_SETTER_POKEMON else 0
        focus_sash_count += 1 if item == "focus sash" else 0
        choice_item_count += 1 if item in CHOICE_ITEMS else 0
        defensive_item_count += 1 if item in DEFENSIVE_ITEMS else 0
        offensive_item_count += 1 if item in OFFENSIVE_ITEMS else 0

    if members_with_missing_stats:
        notes.append("missing Pokémon stats")

    return {
        "team_id": team_record.get("team_id"),
        "pokemon_names": list(team_record.get("pokemon_names", [])),
        "type_distribution": dict(type_counter),
        "average_hp": _average(hp_values),
        "average_attack": _average(attack_values),
        "average_defense": _average(defense_values),
        "average_sp_attack": _average(sp_attack_values),
        "average_sp_defense": _average(sp_defense_values),
        "average_speed": _average(speed_values),
        "average_bulk": _average(bulk_values),
        "physical_attacker_count": physical_attacker_count,
        "special_attacker_count": special_attacker_count,
        "mixed_attacker_count": mixed_attacker_count,
        "fast_pokemon_count": fast_pokemon_count,
        "slow_pokemon_count": slow_pokemon_count,
        "bulky_pokemon_count": bulky_pokemon_count,
        "setup_move_count": setup_move_count,
        "recovery_move_count": recovery_move_count,
        "status_move_count": status_move_count,
        "priority_move_count": priority_move_count,
        "speed_control_count": speed_control_count,
        "trick_room_move_count": trick_room_move_count,
        "tailwind_move_count": tailwind_move_count,
        "weather_move_count": weather_move_count,
        "protect_count": protect_count,
        "pivot_move_count": pivot_move_count,
        "weather_setter_count": weather_setter_count,
        "weather_abuser_count": weather_abuser_count,
        "trick_room_setter_count": trick_room_setter_count,
        "tailwind_setter_count": tailwind_setter_count,
        "focus_sash_count": focus_sash_count,
        "choice_item_count": choice_item_count,
        "defensive_item_count": defensive_item_count,
        "offensive_item_count": offensive_item_count,
        "rain_payoff_move_count": rain_payoff_move_count,
        "sun_payoff_move_count": sun_payoff_move_count,
        "sand_payoff_move_count": sand_payoff_move_count,
        "snow_payoff_move_count": snow_payoff_move_count,
        "moves": list(team_record.get("moves", [])),
        "items": list(team_record.get("items", [])),
        "abilities": list(team_record.get("abilities", [])),
        "tournament_name": team_record.get("tournament_name"),
        "tournament_date": team_record.get("tournament_date"),
        "player_record": team_record.get("player_record"),
        "wins": team_record.get("wins"),
        "losses": team_record.get("losses"),
        "placement": team_record.get("placement"),
        "team_win_rate": team_record.get("team_win_rate"),
        "data_quality_notes": sorted(set(notes)),
    }


def build_feature_records() -> list[dict[str, Any]]:
    """Build extracted feature records for every normalized tournament team."""
    return [extract_team_features(team_record) for team_record in load_tournament_teams()]
