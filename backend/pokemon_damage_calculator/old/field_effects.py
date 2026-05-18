from __future__ import annotations

from .models import BattleField, BattleMove, BattlePokemon, DamageContext
from .utils import normalize_name


def effective_weather(field: BattleField, attacker: BattlePokemon, defender: BattlePokemon) -> str:
    attacker_ability = "" if attacker.ability_suppressed else normalize_name(attacker.ability)
    defender_ability = "" if defender.ability_suppressed else normalize_name(defender.ability)
    if attacker_ability in {"cloud-nine", "air-lock"}:
        return ""
    if defender_ability in {"cloud-nine", "air-lock"}:
        return ""
    return normalize_name(field.weather)


def weather_modifier(field: BattleField, attacker: BattlePokemon, defender: BattlePokemon, ctx: DamageContext) -> float:
    weather = effective_weather(field, attacker, defender)
    if weather == "rain":
        if ctx.move_type == "water": return 1.5
        if ctx.move_type == "fire": return 0.5
    if weather == "sun":
        if ctx.move_type == "fire": return 1.5
        if ctx.move_type == "water": return 0.5
    if weather == "harsh-sun" and ctx.move_type == "water": return 0.0
    if weather == "heavy-rain" and ctx.move_type == "fire": return 0.0
    return 1.0


def terrain_modifier(field: BattleField, attacker: BattlePokemon, defender: BattlePokemon, move: BattleMove, ctx: DamageContext) -> float:
    return 1.0


def screen_modifier(field: BattleField, move: BattleMove, ctx: DamageContext) -> float:
    if ctx.is_critical:
        return 1.0
    category = normalize_name(move.category)
    side = field.defender_side
    if side.aurora_veil and category in {"physical", "special"}:
        return 2 / 3 if field.is_doubles else 0.5
    if side.reflect and category == "physical":
        return 2 / 3 if field.is_doubles else 0.5
    if side.light_screen and category == "special":
        return 2 / 3 if field.is_doubles else 0.5
    return 1.0


def spread_modifier(field: BattleField, move: BattleMove) -> float:
    return 0.75 if field.is_doubles and move.is_spread else 1.0


def helping_hand_modifier(field: BattleField) -> float:
    return 1.0


def friend_guard_modifier(field: BattleField) -> float:
    return 0.75 if field.defender_side.friend_guard else 1.0


def charge_modifier(attacker: BattlePokemon, field: BattleField, ctx: DamageContext) -> float:
    if (attacker.charge_active or field.charge) and ctx.move_type == "electric":
        return 2.0
    return 1.0
