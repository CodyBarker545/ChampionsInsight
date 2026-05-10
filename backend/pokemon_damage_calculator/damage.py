from __future__ import annotations

from math import floor
from statistics import mean

from .abilities import attacker_ability_modifier, defender_ability_modifier
from .field_effects import (
    charge_modifier,
    friend_guard_modifier,
    helping_hand_modifier,
    screen_modifier,
    spread_modifier,
    terrain_modifier,
    weather_modifier,
)
from .immunities import blocks_by_ability, blocks_priority, blocks_protection, blocks_substitute, type_immunity_overridden
from .accuracy import calculate_accuracy
from .items import apply_survival_items, contact_damage_to_attacker, defender_item_modifier, item_damage_modifier
from .models import BattleField, BattleMove, BattlePokemon, DamageContext
from .moves import get_base_power, get_hit_count, get_move_type
from .residual import end_turn_damage_or_healing, spikes_damage, stealth_rock_damage
from .stats import get_attack_stat, get_defense_stat
from .tera import get_effective_types, get_stab
from .type_chart import get_type_multiplier
from .utils import (
    attacker_ignores_target_ability,
    ability_shield_active,
    critical_hits,
    defender_ability_can_be_ignored,
    defender_ability_is_ignored,
    move_ignores_target_ability,
    neutralizing_gas_can_suppress,
    normalize_name,
    poke_round,
)


def calculate_base_damage(level: int, power: int, attack: int, defense: int) -> int:
    step1 = floor((2 * level) / 5 + 2)
    step2 = floor(step1 * power * attack / max(1, defense))
    step3 = floor(step2 / 50)
    return floor(step3 + 2)


def percent_range(damage_values: list[int], defender: BattlePokemon) -> dict:
    if not damage_values or defender.max_hp <= 0:
        return {"minPercent": 0.0, "maxPercent": 0.0, "rangeText": "0% - 0%"}
    min_percent = round(min(damage_values) / defender.max_hp * 100, 1)
    max_percent = round(max(damage_values) / defender.max_hp * 100, 1)
    return {"minPercent": min_percent, "maxPercent": max_percent, "rangeText": f"{min_percent}% - {max_percent}%"}


def estimate_ko_chance(damage_values: list[int], defender: BattlePokemon) -> str:
    hp = defender.hp()
    if not damage_values or hp <= 0:
        return "Unknown"
    one_hit = sum(1 for damage in damage_values if damage >= hp)
    total = len(damage_values)
    if one_hit == total:
        return "Guaranteed OHKO"
    if one_hit > 0:
        return f"{round(one_hit / total * 100, 1)}% chance to OHKO"
    if min(damage_values) * 2 >= hp:
        return "Guaranteed 2HKO"
    if max(damage_values) * 2 >= hp:
        return "Possible 2HKO"
    if min(damage_values) * 3 >= hp:
        return "Guaranteed 3HKO"
    if max(damage_values) * 3 >= hp:
        return "Possible 3HKO"
    return "Likely 4HKO or worse"


def apply_burn_modifier(attacker: BattlePokemon, move: BattleMove) -> float:
    if normalize_name(attacker.status) == "burn" and normalize_name(move.category) == "physical":
        if normalize_name(attacker.ability) == "guts" and not attacker.ability_suppressed:
            return 1.0
        if normalize_name(move.name) == "facade":
            return 1.0
        return 0.5
    return 1.0


def critical_modifier(attacker: BattlePokemon, defender: BattlePokemon, field: BattleField) -> float:
    return 1.5 if critical_hits(attacker, defender, field) else 1.0


def apply_ability_suppression(attacker: BattlePokemon, defender: BattlePokemon, move: BattleMove, field: BattleField) -> None:
    attacker_ability = "" if attacker.ability_suppressed else normalize_name(attacker.ability)
    defender_ability = "" if defender.ability_suppressed else normalize_name(defender.ability)

    if (attacker_ignores_target_ability(attacker) or move_ignores_target_ability(move)) and defender_ability_can_be_ignored(defender):
        if not ability_shield_active(defender, field):
            defender.ability_suppressed = True

    if attacker_ability == "neutralizing-gas" and defender_ability and neutralizing_gas_can_suppress(defender, field):
        defender.ability_suppressed = True
        defender_ability = ""

    if defender_ability == "neutralizing-gas" and attacker_ability and neutralizing_gas_can_suppress(attacker, field):
        attacker.ability_suppressed = True


def make_context(attacker: BattlePokemon, defender: BattlePokemon, move: BattleMove, field: BattleField) -> tuple[DamageContext, list[str]]:
    notes: list[str] = []
    move_type, type_notes = get_move_type(move, attacker, field)
    notes.extend(type_notes)
    base_power, power_notes = get_base_power(move, attacker, defender, field, move_type)
    notes.extend(power_notes)

    effective_types = get_effective_types(defender)
    type_multiplier = get_type_multiplier(move_type, effective_types)
    if normalize_name(field.weather) in {"strong-winds", "delta-stream"} and "flying" in effective_types and move_type in {"electric", "ice", "rock"} and type_multiplier > 1:
        type_multiplier /= 2
        notes.append("Delta Stream removes Flying-type super-effective weakness.")

    if type_multiplier == 0.0 and type_immunity_overridden(attacker, defender, move, move_type, field):
        type_multiplier = 1.0
        notes.append("Type immunity overridden by ability/item/move.")

    ctx = DamageContext(
        move_type=move_type,
        original_move_type=normalize_name(move.type),
        category=normalize_name(move.category),
        base_power=base_power,
        type_multiplier=type_multiplier,
        is_critical=critical_hits(attacker, defender, field),
        is_super_effective=type_multiplier > 1,
        is_not_very_effective=0 < type_multiplier < 1,
        notes=notes,
    )
    return ctx, effective_types


def calculate_single_hit_damage(attacker: BattlePokemon, defender: BattlePokemon, move: BattleMove, field: BattleField, random_roll: int) -> tuple[int, DamageContext]:
    apply_ability_suppression(attacker, defender, move, field)
    ctx, effective_types = make_context(attacker, defender, move, field)

    if normalize_name(move.category) == "status" or ctx.base_power <= 0:
        return 0, ctx

    blocked, reason = blocks_protection(attacker, defender, move)
    if blocked:
        ctx.notes.append(reason)
        return 0, ctx
    elif reason:
        ctx.notes.append(reason)

    blocked, reason = blocks_priority(attacker, defender, move, field)
    if blocked:
        ctx.notes.append(reason)
        return 0, ctx

    blocked, reason = blocks_substitute(attacker, defender, move)
    if reason:
        ctx.notes.append(reason)

    blocked, reason = blocks_by_ability(attacker, defender, move, ctx.move_type, ctx.type_multiplier, field)
    if blocked:
        ctx.notes.append(reason)
        return 0, ctx

    defender_ability = normalize_name(defender.ability)
    defender_ignored = defender_ability_is_ignored(attacker, defender, move, field)
    if defender_ability == "disguise" and not defender.ability_triggered and not defender_ignored:
        ctx.notes.append("Disguise blocks this hit.")
        return 0, ctx
    if defender_ability == "ice-face" and normalize_name(move.category) == "physical" and not defender.ability_triggered and not defender_ignored:
        ctx.notes.append("Ice Face blocks this physical hit.")
        return 0, ctx

    if ctx.type_multiplier == 0.0:
        ctx.notes.append("No damage due to type immunity.")
        return 0, ctx

    attack, attack_name, attack_notes = get_attack_stat(attacker, defender, move, field, ctx.move_type)
    defense, defense_name, defense_notes = get_defense_stat(defender, attacker, move, field)
    ctx.attack_stat = attack
    ctx.defense_stat = defense
    ctx.notes.extend(attack_notes)
    ctx.notes.extend(defense_notes)

    base_damage = calculate_base_damage(attacker.level, ctx.base_power, attack, defense)
    ctx.stab = get_stab(attacker, ctx.move_type)

    modifiers = {
        "spread": spread_modifier(field, move),
        "weather": weather_modifier(field, attacker, defender, ctx),
        "critical": critical_modifier(attacker, defender, field),
        "random": random_roll / 100,
        "stab": ctx.stab,
        "type": ctx.type_multiplier,
        "burn": apply_burn_modifier(attacker, move),
        "screen": screen_modifier(field, move, ctx),
        "terrain": terrain_modifier(field, attacker, defender, move, ctx),
        "charge": charge_modifier(attacker, field, ctx),
        "attackerItem": item_damage_modifier(attacker, defender, move, field, ctx),
        "defenderItem": defender_item_modifier(attacker, defender, move, field, ctx),
        "attackerAbility": attacker_ability_modifier(attacker, defender, move, field, ctx),
        "defenderAbility": defender_ability_modifier(attacker, defender, move, field, ctx),
        "helpingHand": helping_hand_modifier(field),
        "friendGuard": friend_guard_modifier(field),
    }

    damage = base_damage
    rounded_final_modifiers = {"attackerItem", "defenderItem", "attackerAbility", "defenderAbility", "friendGuard"}
    for key, modifier in modifiers.items():
        if key in rounded_final_modifiers:
            damage = poke_round(damage * modifier)
        else:
            damage = floor(damage * modifier)

    if modifiers["weather"] == 0 or ctx.type_multiplier == 0:
        damage = 0
    else:
        damage = max(1, damage)

    # Focus Sash/Sturdy are reported for the high roll, but this does not mutate defender state.
    adjusted, survival_note = apply_survival_items(defender, damage, field)
    if survival_note:
        ctx.notes.append(survival_note)
        damage = adjusted

    ctx.flags["modifiers"] = modifiers
    ctx.flags["effectiveDefenderTypes"] = effective_types
    ctx.flags["attackStatName"] = attack_name
    ctx.flags["defenseStatName"] = defense_name
    ctx.flags["baseDamage"] = base_damage
    return damage, ctx


def calculate_damage_range(attacker: BattlePokemon, defender: BattlePokemon, move: BattleMove, field: BattleField) -> dict:
    if normalize_name(move.category) == "status" or not move.power:
        return {
            "move": move.name,
            "minDamage": 0,
            "maxDamage": 0,
            "damageValues": [],
            "koChance": "Status move",
            "message": "Status moves do not deal direct damage.",
        }

    apply_ability_suppression(attacker, defender, move, field)
    hit_counts, hit_probs, hit_notes = get_hit_count(move, attacker)
    all_totals: list[int] = []
    contexts: list[DamageContext] = []

    for hit_count in hit_counts:
        for random_roll in range(85, 101):
            single, ctx = calculate_single_hit_damage(attacker, defender, move, field, random_roll)
            contexts.append(ctx)
            total = single * hit_count
            if normalize_name(attacker.ability) == "parental-bond" and not attacker.ability_suppressed and single > 0:
                total = single + max(1, floor(single * 0.25))
                ctx.notes.append("Parental Bond second hit approximated at 25%.")
            all_totals.append(total)

    ctx = contexts[-1] if contexts else DamageContext()
    ctx.notes.extend(hit_notes)
    percents = percent_range(all_totals, defender)

    entry_hazard_damage = 0
    hazard_notes: list[str] = []
    if field.defender_side.stealth_rock:
        dmg = stealth_rock_damage(defender)
        entry_hazard_damage += dmg
        hazard_notes.append(f"Stealth Rock would deal {dmg} on switch-in.")
    if field.defender_side.spikes_layers:
        dmg = spikes_damage(defender, field.defender_side.spikes_layers)
        entry_hazard_damage += dmg
        if dmg:
            hazard_notes.append(f"Spikes would deal {dmg} on switch-in.")

    after_hazards_values = [d + entry_hazard_damage for d in all_totals]
    residual = end_turn_damage_or_healing(defender, field)
    contact_damage = contact_damage_to_attacker(attacker, defender, move)

    accuracy = calculate_accuracy(attacker, defender, move, field)

    result = {
        "move": move.name,
        "moveType": ctx.move_type,
        "category": normalize_name(move.category),
        "powerUsed": ctx.base_power,
        "attackStatUsed": ctx.attack_stat,
        "defenseStatUsed": ctx.defense_stat,
        "attackStatName": ctx.flags.get("attackStatName"),
        "defenseStatName": ctx.flags.get("defenseStatName"),
        "baseDamage": ctx.flags.get("baseDamage"),
        "modifiers": ctx.flags.get("modifiers", {}),
        "effectiveDefenderTypes": ctx.flags.get("effectiveDefenderTypes", []),
        "minDamage": min(all_totals) if all_totals else 0,
        "maxDamage": max(all_totals) if all_totals else 0,
        "averageDamage": round(mean(all_totals), 2) if all_totals else 0,
        "damageValues": sorted(all_totals),
        "damageRange": f"{min(all_totals)} - {max(all_totals)}" if all_totals else "0 - 0",
        "percentRange": percents["rangeText"],
        "minPercent": percents["minPercent"],
        "maxPercent": percents["maxPercent"],
        "koChance": estimate_ko_chance(all_totals, defender),
        "hitCountsConsidered": hit_counts,
        "entryHazardDamage": entry_hazard_damage,
        "afterHazards": {
            "damageValues": sorted(after_hazards_values),
            "koChance": estimate_ko_chance(after_hazards_values, defender),
        },
        "accuracy": accuracy,
        "residualOnDefenderEndTurn": residual,
        "contactDamageToAttacker": contact_damage,
        "notes": sorted(set(ctx.notes + hazard_notes + accuracy.get("notes", []))),
    }
    return result
