from __future__ import annotations

from .models import BattleField, BattleMove, BattlePokemon, DamageContext
from .utils import defender_ability_is_ignored, normalize_name


def attacker_ability_modifier(attacker: BattlePokemon, defender: BattlePokemon, move: BattleMove, field: BattleField, ctx: DamageContext) -> float:
    ability = normalize_name(attacker.ability)
    if attacker.ability_suppressed:
        return 1.0

    mod = 1.0

    if ability == "tinted-lens" and 0 < ctx.type_multiplier < 1:
        mod *= 2.0; ctx.notes.append("Tinted Lens")
    if ability == "sniper" and ctx.is_critical:
        mod *= 1.5; ctx.notes.append("Sniper")
    if ability == "neuroforce" and ctx.type_multiplier > 1:
        mod *= 1.25; ctx.notes.append("Neuroforce")

    return mod


def defender_ability_modifier(attacker: BattlePokemon, defender: BattlePokemon, move: BattleMove, field: BattleField, ctx: DamageContext) -> float:
    if defender_ability_is_ignored(attacker, defender, move, field):
        return 1.0

    ability = normalize_name(defender.ability)
    mod = 1.0

    if ability in {"multiscale", "shadow-shield"} and defender.hp() == defender.max_hp:
        mod *= 0.5; ctx.notes.append(defender.ability)
    if ability in {"filter", "solid-rock", "prism-armor"} and ctx.type_multiplier > 1:
        mod *= 0.75; ctx.notes.append(defender.ability)
    attacker_ability = "" if attacker.ability_suppressed else normalize_name(attacker.ability)
    if ability == "fluffy" and move.makes_contact and attacker_ability != "long-reach":
        mod *= 0.5; ctx.notes.append("Fluffy contact reduction")
    if ability == "fluffy" and ctx.move_type == "fire":
        mod *= 2.0; ctx.notes.append("Fluffy Fire weakness")
    if ability == "punk-rock" and move.is_sound:
        mod *= 0.5; ctx.notes.append("Punk Rock sound resistance")
    if ability == "ice-scales" and ctx.category == "special":
        mod *= 0.5; ctx.notes.append("Ice Scales")
    if ability == "tera-shell" and defender.hp() == defender.max_hp and defender.is_terastallized:
        mod *= 0.5; ctx.notes.append("Tera Shell approximation")

    return mod
