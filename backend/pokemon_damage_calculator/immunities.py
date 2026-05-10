from __future__ import annotations

from .models import BattleField, BattleMove, BattlePokemon
from .utils import defender_ability_is_ignored, has_active_item, normalize_name


def is_grounded(pokemon: BattlePokemon, field: BattleField) -> bool:
    if field.gravity:
        return True
    ability = "" if pokemon.ability_suppressed else normalize_name(pokemon.ability)
    item = "" if pokemon.item_suppressed or field.magic_room or ability == "klutz" else normalize_name(pokemon.item)
    if item == "iron-ball":
        return True
    if item == "air-balloon":
        return False
    if ability == "levitate":
        return False
    if "flying" in [normalize_name(t) for t in pokemon.types]:
        return False
    return pokemon.is_grounded


def type_immunity_overridden(attacker: BattlePokemon, defender: BattlePokemon, move: BattleMove, move_type: str, field: BattleField) -> bool:
    ability = "" if attacker.ability_suppressed else normalize_name(attacker.ability)
    defender_ability = "" if defender.ability_suppressed else normalize_name(defender.ability)
    item = "" if defender.item_suppressed or field.magic_room or defender_ability == "klutz" else normalize_name(defender.item)
    move_name = normalize_name(move.name)

    if item == "ring-target":
        return True
    if ability in {"scrappy", "minds-eye"} and move_type in {"normal", "fighting"}:
        return True
    if move_name == "thousand-arrows" and move_type == "ground":
        return True
    return False


def blocks_by_ability(attacker: BattlePokemon, defender: BattlePokemon, move: BattleMove, move_type: str, type_multiplier: float, field: BattleField) -> tuple[bool, str]:
    if defender_ability_is_ignored(attacker, defender, move, field):
        return False, ""

    ability = normalize_name(defender.ability)

    if ability == "levitate" and move_type == "ground" and not field.gravity and not has_active_item(defender, field, "iron-ball"):
        return True, "Levitate blocks Ground-type damage."
    if ability == "flash-fire" and move_type == "fire":
        return True, "Flash Fire blocks Fire-type damage."
    if ability in {"water-absorb", "storm-drain", "dry-skin"} and move_type == "water":
        return True, f"{defender.ability} blocks Water-type damage."
    if ability == "volt-absorb" and move_type == "electric":
        return True, "Volt Absorb blocks Electric-type damage."
    if ability == "motor-drive" and move_type == "electric":
        return True, "Motor Drive blocks Electric-type damage."
    if ability == "lightning-rod" and move_type == "electric":
        return True, "Lightning Rod blocks Electric-type damage."
    if ability == "sap-sipper" and move_type == "grass":
        return True, "Sap Sipper blocks Grass-type damage."
    if ability == "well-baked-body" and move_type == "fire":
        return True, "Well-Baked Body blocks Fire-type damage."
    if ability == "earth-eater" and move_type == "ground":
        return True, "Earth Eater blocks Ground-type damage."
    if ability == "bulletproof" and move.is_bullet:
        return True, "Bulletproof blocks bullet/ball moves."
    if ability == "soundproof" and move.is_sound:
        return True, "Soundproof blocks sound moves."
    if ability == "wonder-guard" and type_multiplier <= 1.0:
        return True, "Wonder Guard blocks non-super-effective direct damage."
    return False, ""


def blocks_priority(attacker: BattlePokemon, defender: BattlePokemon, move: BattleMove, field: BattleField) -> tuple[bool, str]:
    if move.priority <= 0:
        return False, ""
    if field.terrain == "psychic" and defender.is_grounded:
        return True, "Psychic Terrain blocks priority against grounded targets."
    ability = normalize_name(defender.ability)
    if ability in {"queenly-majesty", "armor-tail", "dazzling"} and not defender_ability_is_ignored(attacker, defender, move, field):
        return True, f"{defender.ability} blocks priority moves."
    return False, ""


def blocks_protection(attacker: BattlePokemon, defender: BattlePokemon, move: BattleMove) -> tuple[bool, str]:
    if not defender.protected:
        return False, ""
    if move.ignores_protect:
        return False, ""
    if normalize_name(attacker.ability) == "unseen-fist" and not attacker.ability_suppressed and move.makes_contact:
        return False, "Unseen Fist bypasses protection for contact moves."
    return True, "Target is protected."


def blocks_substitute(attacker: BattlePokemon, defender: BattlePokemon, move: BattleMove) -> tuple[bool, str]:
    if not defender.substitute:
        return False, ""
    if move.is_sound:
        return False, "Sound move bypasses Substitute."
    if normalize_name(attacker.ability) == "infiltrator" and not attacker.ability_suppressed:
        return False, "Infiltrator bypasses Substitute."
    # Damage should normally hit the substitute, not the real defender. For this calculator we report it.
    return False, "Target has Substitute; damage shown is theoretical damage before substitute HP handling."
