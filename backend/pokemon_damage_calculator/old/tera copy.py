from __future__ import annotations

from .old.models import BattlePokemon
from .old.utils import normalize_name


def get_effective_types(pokemon: BattlePokemon) -> list[str]:
    if pokemon.is_terastallized and pokemon.tera_type:
        return [normalize_name(pokemon.tera_type)]
    return [normalize_name(t) for t in pokemon.types]


def get_stab(attacker: BattlePokemon, move_type: str) -> float:
    move_type = normalize_name(move_type)
    ability = "" if attacker.ability_suppressed else normalize_name(attacker.ability)
    original_types = [normalize_name(t) for t in attacker.types]
    tera_type = normalize_name(attacker.tera_type)

    if tera_type == "stellar" and attacker.is_terastallized:
        if move_type in original_types:
            return 2.0 if ability == "adaptability" else 1.5
        return 1.2  # simplified Stellar first-use bonus; caller can override with state if tracking each type.

    if not attacker.is_terastallized:
        if ability in {"protean", "libero"} and not attacker.ability_triggered:
            return 2.0 if ability == "adaptability" else 1.5
        if move_type in original_types:
            return 2.0 if ability == "adaptability" else 1.5
        return 1.0

    if move_type == tera_type and move_type in original_types:
        return 2.25 if ability == "adaptability" else 2.0
    if move_type == tera_type:
        return 2.0 if ability == "adaptability" else 1.5
    if move_type in original_types:
        return 2.0 if ability == "adaptability" else 1.5
    return 1.0
