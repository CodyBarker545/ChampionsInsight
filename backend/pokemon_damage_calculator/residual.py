from __future__ import annotations

from math import floor

from .models import BattleField, BattlePokemon
from .type_chart import get_type_multiplier
from .utils import normalize_name


def stealth_rock_damage(pokemon: BattlePokemon) -> int:
    multiplier = get_type_multiplier("rock", pokemon.types)
    return max(1, floor(pokemon.max_hp * multiplier / 8))


def spikes_damage(pokemon: BattlePokemon, layers: int) -> int:
    if not pokemon.is_grounded or "flying" in [normalize_name(t) for t in pokemon.types] or normalize_name(pokemon.ability) == "levitate":
        return 0
    layers = max(0, min(3, layers))
    if layers == 1: return max(1, floor(pokemon.max_hp / 8))
    if layers == 2: return max(1, floor(pokemon.max_hp / 6))
    if layers == 3: return max(1, floor(pokemon.max_hp / 4))
    return 0


def end_turn_damage_or_healing(pokemon: BattlePokemon, field: BattleField) -> dict[str, int]:
    changes: dict[str, int] = {}
    ability = normalize_name(pokemon.ability)
    item = normalize_name(pokemon.item)
    types = [normalize_name(t) for t in pokemon.types]

    if pokemon.status == "burn" and ability != "magic-guard":
        changes["burn"] = -max(1, floor(pokemon.max_hp / 16))
    if pokemon.status == "poison" and ability == "poison-heal":
        changes["poison_heal"] = max(1, floor(pokemon.max_hp / 8))
    elif pokemon.status == "poison" and ability != "magic-guard":
        changes["poison"] = -max(1, floor(pokemon.max_hp / 8))
    ignores_weather_damage = ability in {"sand-veil", "sand-rush", "sand-force", "magic-guard", "overcoat"} or item == "safety-goggles"
    if field.weather == "sand" and not ({"rock", "ground", "steel"} & set(types)) and not ignores_weather_damage:
        changes["sand"] = -max(1, floor(pokemon.max_hp / 16))
    if field.weather == "snow" and ability == "ice-body":
        changes["ice_body"] = max(1, floor(pokemon.max_hp / 16))
    if field.weather in {"sun", "harsh-sun"} and ability == "dry-skin":
        changes["dry_skin_sun"] = -max(1, floor(pokemon.max_hp / 8))
    if field.weather in {"rain", "heavy-rain"} and ability == "dry-skin":
        changes["dry_skin_rain"] = max(1, floor(pokemon.max_hp / 8))
    if field.weather in {"sun", "harsh-sun"} and ability == "solar-power":
        changes["solar_power"] = -max(1, floor(pokemon.max_hp / 8))
    if item == "leftovers":
        changes["leftovers"] = max(1, floor(pokemon.max_hp / 16))
    if item == "black-sludge":
        changes["black_sludge"] = max(1, floor(pokemon.max_hp / 16)) if "poison" in types else -max(1, floor(pokemon.max_hp / 8))
    if field.terrain == "grassy" and pokemon.is_grounded:
        changes["grassy_terrain"] = max(1, floor(pokemon.max_hp / 16))
    return changes
