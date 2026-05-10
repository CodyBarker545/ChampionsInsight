"""Loads Pokemon battle data and prepares frontend-friendly Pokemon records."""

import json
from functools import lru_cache

from paths import (
    CHAMPIONS_ROSTER_ABILITIES_PATH,
    POKEMON_BATTLE_DATA_PATH,
    SPRITE_METADATA_PATH,
    SPRITE_ROOT,
)


POKEMON_DATA_PATH = POKEMON_BATTLE_DATA_PATH


# Converts display names into lookup keys that ignore spaces and punctuation.
def normalize_lookup_key(value):
    return "".join(character.lower() for character in str(value or "") if character.isalnum())


def alternate_lookup_names(value):
    name = str(value or "").strip()
    alternates = [name]
    regional_prefixes = {
        "Alolan ": " Alola",
        "Galarian ": " Galar",
        "Hisuian ": " Hisui",
    }
    for prefix, suffix in regional_prefixes.items():
        if name.startswith(prefix):
            alternates.append(f"{name.removeprefix(prefix)}{suffix}")
    rotom_forms = {"Heat", "Wash", "Frost", "Fan", "Mow"}
    parts = name.split()
    if len(parts) == 2 and parts[0] in rotom_forms and parts[1] == "Rotom":
        alternates.append(f"Rotom {parts[0]}")
    special_aliases = {
        "Basculegion-M": "Basculegion Male",
        "Basculegion-F": "Basculegion Female",
        "Meowstic-M": "Meowstic Male",
        "Meowstic-F": "Meowstic Female",
        "Eternal Floette": "Floette Eternal",
        "Paldean Tauros": "Tauros Paldea Combat Breed",
        "Paldean Tauros (Blaze)": "Tauros Paldea Blaze Breed",
        "Paldean Tauros (Aqua)": "Tauros Paldea Aqua Breed",
    }
    if name in special_aliases:
        alternates.append(special_aliases[name])
    return alternates


def lookup_keys_for_value(value):
    return [key for key in (normalize_lookup_key(name) for name in alternate_lookup_names(value)) if key]


# Converts API stat keys into frontend stat keys.
def format_stats(stats):
    return {
        "hp": stats.get("hp", 0),
        "attack": stats.get("attack", 0),
        "defense": stats.get("defense", 0),
        "specialAttack": stats.get("special_attack", 0),
        "specialDefense": stats.get("special_defense", 0),
        "speed": stats.get("speed", 0),
    }


# Converts a name into title case while preserving spaces.
def title_name(value):
    return str(value or "").replace("-", " ").title()


# Loads all Pokemon records once for repeated API calls.
@lru_cache(maxsize=1)
def load_pokemon_records():
    if not POKEMON_DATA_PATH.exists():
        return []

    return json.loads(POKEMON_DATA_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_champions_roster_abilities():
    if not CHAMPIONS_ROSTER_ABILITIES_PATH.exists():
        return {}

    return json.loads(CHAMPIONS_ROSTER_ABILITIES_PATH.read_text(encoding="utf-8"))


# Loads Champion sprite metadata once for repeated API calls.
@lru_cache(maxsize=1)
def load_sprite_records():
    if not SPRITE_METADATA_PATH.exists():
        return []

    return json.loads(SPRITE_METADATA_PATH.read_text(encoding="utf-8"))


# Builds a lookup map from common display names to Pokemon records.
@lru_cache(maxsize=1)
def build_pokemon_lookup():
    lookup = {}
    for record in load_pokemon_records():
        names = [
            record.get("form_display_name"),
            record.get("form_api_name"),
        ]
        for name in names:
            for key in lookup_keys_for_value(name):
                if key not in lookup:
                    lookup[key] = record

    for record in load_pokemon_records():
        names = [
            record.get("species_display_name"),
            record.get("form_display_name"),
            record.get("species_api_name"),
            record.get("form_api_name"),
        ]
        for name in names:
            for key in lookup_keys_for_value(name):
                if key not in lookup:
                    lookup[key] = record

    return lookup


# Builds a lookup map from common display names to Champion sprite records.
@lru_cache(maxsize=1)
def build_sprite_lookup():
    lookup = {}
    for record in load_sprite_records():
        names = [
            record.get("display_name"),
            record.get("form_display_name"),
            record.get("form_api_name"),
        ]
        for name in names:
            for key in lookup_keys_for_value(name):
                if key not in lookup:
                    lookup[key] = record

    for record in load_sprite_records():
        names = [
            record.get("display_name"),
            record.get("species_display_name"),
            record.get("form_display_name"),
            record.get("form_api_name"),
        ]
        for name in names:
            for key in lookup_keys_for_value(name):
                if key not in lookup:
                    lookup[key] = record

    return lookup


# Finds the best local data record for a detected or manually entered Pokemon name.
def find_pokemon_record(name):
    lookup = build_pokemon_lookup()
    for key in lookup_keys_for_value(name):
        if key in lookup:
            return lookup[key]
    return None


# Finds the best Champion sprite metadata for a detected or manually entered Pokemon name.
def find_sprite_record(name):
    lookup = build_sprite_lookup()
    for key in lookup_keys_for_value(name):
        if key in lookup:
            return lookup[key]
    return None


def find_form_options(record):
    if not record:
        return []

    species_key = normalize_lookup_key(
        record.get("species_api_name") or record.get("species_display_name")
    )
    if not species_key:
        return []

    options = []
    seen_names = set()
    for pokemon_record in load_pokemon_records():
        candidate_species_key = normalize_lookup_key(
            pokemon_record.get("species_api_name")
            or pokemon_record.get("species_display_name")
        )
        if candidate_species_key != species_key:
            continue

        display_name = (
            pokemon_record.get("form_display_name")
            or pokemon_record.get("species_display_name")
            or pokemon_record.get("form_api_name")
        )
        if not display_name or display_name in seen_names:
            continue

        seen_names.add(display_name)
        options.append({
            "name": display_name,
            "label": display_name,
            "isDefault": bool(pokemon_record.get("is_default")),
        })

    return options


def find_champions_roster_abilities(name, record):
    roster_abilities = load_champions_roster_abilities()
    roster_by_key = {normalize_lookup_key(roster_name): abilities for roster_name, abilities in roster_abilities.items()}
    exact_names = alternate_lookup_names(name)
    if record:
        exact_names.append(record.get("form_display_name"))

    for lookup_name in exact_names:
        abilities = roster_by_key.get(normalize_lookup_key(lookup_name))
        if abilities:
            return abilities

    if record and not record.get("is_default", True):
        return []

    species_names = []
    if record:
        species_names.append(record.get("species_display_name"))

    for lookup_name in species_names:
        abilities = roster_by_key.get(normalize_lookup_key(lookup_name))
        if abilities:
            return abilities

    return []


# Builds a safe sprite API URL from local Champion sprite metadata.
def build_sprite_url(sprite_record):
    if not sprite_record:
        return ""

    sprite_type = sprite_record.get("sprite_type") or "normal"
    filename = sprite_record.get("local_filename") or ""
    if not filename:
        return ""

    return f"/api/pokemon/sprite/{sprite_type}/{filename}"


# Builds a frontend Pokemon record with image, types, abilities, moves, and base stats.
def build_pokemon_summary(name):
    record = find_pokemon_record(name)
    sprite_record = find_sprite_record(name)
    if not record:
        return {
            "name": name,
            "image": build_sprite_url(sprite_record),
            "types": [],
            "baseStats": {},
            "abilities": [],
            "moves": [],
        }

    abilities = find_champions_roster_abilities(name, record) or [
        ability.get("display_name", title_name(ability.get("name"))) for ability in record.get("abilities", [])
    ]
    moves = [move.get("display_name", title_name(move.get("name"))) for move in record.get("moves", [])]

    return {
        "name": record.get("form_display_name") or record.get("species_display_name") or name,
        "species": record.get("species_display_name", ""),
        "form": record.get("form_display_name", ""),
        "formOptions": find_form_options(record),
        "image": build_sprite_url(sprite_record),
        "types": [title_name(type_name) for type_name in record.get("types", [])],
        "baseStats": format_stats(record.get("base_stats", {})),
        "abilities": abilities,
        "moves": moves,
    }


def search_pokemon_summaries(query, limit=12):
    normalized_query = normalize_lookup_key(query)
    if not normalized_query:
        return []

    results = []
    seen_names = set()

    for record in load_pokemon_records():
        display_name = (
            record.get("form_display_name")
            or record.get("species_display_name")
            or record.get("form_api_name")
            or record.get("species_api_name")
            or ""
        )
        search_names = [
            record.get("form_display_name"),
            record.get("species_display_name"),
            record.get("form_api_name"),
            record.get("species_api_name"),
        ]

        if not display_name or display_name in seen_names:
            continue

        if not any(normalized_query in normalize_lookup_key(name) for name in search_names if name):
            continue

        sprite_record = find_sprite_record(display_name)
        seen_names.add(display_name)
        results.append({
            "name": display_name,
            "species": record.get("species_display_name", ""),
            "form": record.get("form_display_name", ""),
            "image": build_sprite_url(sprite_record),
            "types": [title_name(type_name) for type_name in record.get("types", [])],
        })

        if len(results) >= limit:
            break

    return results


# Adds Pokemon data to each detected opponent slot.
def enrich_detected_team(detected_team):
    enriched_team = []
    for slot in detected_team:
        pokemon_name = slot.get("pokemonName", "")
        pokemon_summary = build_pokemon_summary(pokemon_name) if pokemon_name and pokemon_name != "unknown" else {}
        base_stats = pokemon_summary.get("baseStats", {})

        enriched_team.append({
            **slot,
            "pokemon": pokemon_summary,
            "image": pokemon_summary.get("image", ""),
            "types": pokemon_summary.get("types", []),
            "baseStats": base_stats,
            "statPoints": {stat_name: 0 for stat_name in base_stats},
            "stats": base_stats,
            "abilities": pokemon_summary.get("abilities", []),
            "moves": pokemon_summary.get("moves", []),
        })

    return enriched_team
