from __future__ import annotations

from math import floor
from .models import BattleField, BattleMove, BattlePokemon
from .utils import clamp, normalize_name


def stage_multiplier(stage: int) -> float:
    stage = clamp(stage, -6, 6)
    if stage >= 0:
        return (3 + stage) / 3
    return 3 / (3 - stage)


def calculate_accuracy(attacker: BattlePokemon, defender: BattlePokemon, move: BattleMove, field: BattleField) -> dict:
    if move.accuracy is None:
        return {"accuracy": None, "chance": 1.0, "notes": ["Move bypasses accuracy check."]}

    ability_a = "" if attacker.ability_suppressed else normalize_name(attacker.ability)
    ability_d = "" if defender.ability_suppressed else normalize_name(defender.ability)
    item_a = "" if attacker.item_suppressed or field.magic_room or ability_a == "klutz" else normalize_name(attacker.item)
    item_d = "" if defender.item_suppressed or field.magic_room or ability_d == "klutz" else normalize_name(defender.item)

    if ability_a == "no-guard" or ability_d == "no-guard":
        return {"accuracy": None, "chance": 1.0, "notes": ["No Guard bypasses accuracy checks."]}

    acc = float(move.accuracy)
    notes: list[str] = []

    acc *= stage_multiplier(attacker.boosts.get("accuracy", 0))
    if ability_a not in {"keen-eye", "minds-eye"}:
        acc /= stage_multiplier(defender.boosts.get("evasion", 0))
    elif defender.boosts.get("evasion", 0) > 0:
        notes.append(f"{attacker.ability} ignores evasion boosts")

    if ability_a == "compound-eyes":
        acc *= 1.3; notes.append("Compound Eyes")
    if ability_a == "victory-star":
        acc *= 1.1; notes.append("Victory Star")
    if ability_a == "hustle" and normalize_name(move.category) == "physical":
        acc *= 0.8; notes.append("Hustle accuracy penalty")
    if item_a == "wide-lens":
        acc *= 1.1; notes.append("Wide Lens")
    if item_a == "zoom-lens" and attacker.has_acted:
        acc *= 1.2; notes.append("Zoom Lens")
    if item_a == "micle-berry" and attacker.used_item:
        acc *= 1.2; notes.append("Micle Berry")

    if item_d == "bright-powder":
        acc *= 0.9; notes.append("Bright Powder")
    if item_d == "lax-incense":
        acc *= 0.9; notes.append("Lax Incense")
    if ability_d == "sand-veil" and field.weather == "sand":
        acc *= 0.8; notes.append("Sand Veil")
    if ability_d == "snow-cloak" and field.weather == "snow":
        acc *= 0.8; notes.append("Snow Cloak")

    acc = max(0.0, min(100.0, acc))
    return {"accuracy": round(acc, 2), "chance": round(acc / 100, 4), "notes": notes}
