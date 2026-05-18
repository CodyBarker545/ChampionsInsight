from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from paths import DATA_DIR
from .utils import normalize_name


POKEMON_DATA_DIR = DATA_DIR / "pokemon"


def load_json_file(filename: str) -> dict[str, Any]:
    path = POKEMON_DATA_DIR / filename

    if not path.exists():
        print(f"DATA LOADER MISSING FILE: {path}")
        return {}

    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"DATA LOADER LOADED: {path} ({len(data)} records)")
    return data


@lru_cache(maxsize=1)
def load_move_dictionary() -> dict[str, Any]:
    return load_json_file("move_dictionary.json")


@lru_cache(maxsize=1)
def load_item_dictionary() -> dict[str, Any]:
    return load_json_file("item_dictionary.json")


@lru_cache(maxsize=1)
def load_ability_dictionary() -> dict[str, Any]:
    return load_json_file("ability_dictionary.json")


def get_move_data(name: str) -> dict[str, Any]:
    dictionary = load_move_dictionary()

    keys_to_try = [
        normalize_name(name),                    # Shadow Ball -> shadow-ball
        str(name or "").strip().lower(),         # shadow ball
        str(name or "").strip().lower().replace(" ", "-"),
        str(name or "").strip().lower().replace(" ", ""),
    ]

    for key in keys_to_try:
        if key in dictionary:
            return dictionary[key]

    print(f"MOVE NOT FOUND: {name} | tried {keys_to_try}")
    return {}


def get_item_data(name: str) -> dict[str, Any]:
    dictionary = load_item_dictionary()

    keys_to_try = [
        normalize_name(name),
        str(name or "").strip().lower(),
        str(name or "").strip().lower().replace(" ", "-"),
        str(name or "").strip().lower().replace(" ", ""),
    ]

    for key in keys_to_try:
        if key in dictionary:
            return dictionary[key]

    return {}


def get_ability_data(name: str) -> dict[str, Any]:
    dictionary = load_ability_dictionary()

    keys_to_try = [
        normalize_name(name),
        str(name or "").strip().lower(),
        str(name or "").strip().lower().replace(" ", "-"),
        str(name or "").strip().lower().replace(" ", ""),
    ]

    for key in keys_to_try:
        if key in dictionary:
            return dictionary[key]

    return {}