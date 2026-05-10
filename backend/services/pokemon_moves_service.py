"""Loads tournament-derived common move data for Pokemon."""

import json
from functools import lru_cache

from paths import TOP_MOVES_PATH
from services.pokemon_data_service import lookup_keys_for_value


@lru_cache(maxsize=1)
def load_top_moves_data():
    if not TOP_MOVES_PATH.exists():
        return {}

    return json.loads(TOP_MOVES_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def build_top_moves_lookup():
    lookup = {}

    for pokemon_name, moves in load_top_moves_data().items():
        for lookup_key in lookup_keys_for_value(pokemon_name):
            if lookup_key not in lookup:
                lookup[lookup_key] = {
                    "pokemon": pokemon_name,
                    "moves": moves,
                }

    return lookup


def get_top_tournament_moves(name, limit=4):
    lookup = build_top_moves_lookup()

    for lookup_key in lookup_keys_for_value(name):
        if lookup_key in lookup:
            match = lookup[lookup_key]
            moves = match.get("moves", [])[:limit]

            return {
                "name": name,
                "matchedName": match.get("pokemon", name),
                "moves": moves,
            }

    return {
        "name": name,
        "matchedName": None,
        "moves": [],
    }
