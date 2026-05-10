from __future__ import annotations

from math import floor

from .models import BattleField, BattlePokemon, SideConditions
from .stats import apply_stat_stage
from .utils import normalize_name


def get_modified_speed(pokemon: BattlePokemon, field: BattleField, side: SideConditions | None = None) -> dict:
    ability = "" if pokemon.ability_suppressed else normalize_name(pokemon.ability)
    item = normalize_name(pokemon.item) if not pokemon.item_suppressed and not field.magic_room and ability != "klutz" else ""
    status = normalize_name(pokemon.status)

    speed = pokemon.stats.get("speed", 0)
    stage = pokemon.boosts.get("speed", 0)
    if ability == "simple":
        stage *= 2
    modified = apply_stat_stage(speed, stage)
    modifiers: list[str] = []

    if item == "choice-scarf":
        modified = floor(modified * 1.5); modifiers.append("Choice Scarf")
    if item == "quick-powder" and normalize_name(pokemon.name) == "ditto":
        modified = floor(modified * 2); modifiers.append("Quick Powder")
    if item == "iron-ball":
        modified = floor(modified * 0.5); modifiers.append("Iron Ball")
    if item == "macho-brace" or item.startswith("power-"):
        modified = floor(modified * 0.5); modifiers.append(pokemon.item)
    if pokemon.used_item and item == "salac-berry":
        modified = floor(modified * 1.5); modifiers.append("Salac Berry activated")
    if pokemon.used_item and item == "adrenaline-orb":
        modified = floor(modified * 1.5); modifiers.append("Adrenaline Orb activated")
    if ability == "slow-start" and not pokemon.ability_triggered:
        modified = floor(modified * 0.5); modifiers.append("Slow Start")
    if status == "paralysis" and ability != "quick-feet":
        modified = floor(modified * 0.5); modifiers.append("Paralysis")
    if ability == "quick-feet" and status:
        modified = floor(modified * 1.5); modifiers.append("Quick Feet")
    if ability == "swift-swim" and field.weather in {"rain", "heavy-rain"}:
        modified = floor(modified * 2); modifiers.append("Swift Swim")
    if ability == "chlorophyll" and field.weather in {"sun", "harsh-sun"}:
        modified = floor(modified * 2); modifiers.append("Chlorophyll")
    if ability == "sand-rush" and field.weather == "sand":
        modified = floor(modified * 2); modifiers.append("Sand Rush")
    if ability == "slush-rush" and field.weather == "snow":
        modified = floor(modified * 2); modifiers.append("Slush Rush")
    if ability == "surge-surfer" and field.terrain == "electric":
        modified = floor(modified * 2); modifiers.append("Surge Surfer")
    if ability == "unburden" and pokemon.used_item:
        modified = floor(modified * 2); modifiers.append("Unburden")
    proto_active = ability == "protosynthesis" and (field.weather in {"sun", "harsh-sun"} or pokemon.booster_energy_active or item == "booster-energy")
    quark_active = ability == "quark-drive" and (field.terrain == "electric" or pokemon.booster_energy_active or item == "booster-energy")
    if quark_active and pokemon.quark_drive_stat == "speed":
        modified = floor(modified * 1.5); modifiers.append("Quark Drive Speed")
    if proto_active and pokemon.protosynthesis_stat == "speed":
        modified = floor(modified * 1.5); modifiers.append("Protosynthesis Speed")
    side = side or field.attacker_side
    if side.tailwind:
        modified = floor(modified * 2); modifiers.append("Tailwind")

    return {"baseSpeed": speed, "boostStage": stage, "modifiedSpeed": max(0, modified), "modifiers": modifiers}


def compare_speed(user_pokemon: BattlePokemon, opponent_pokemon: BattlePokemon, field: BattleField) -> dict:
    user_speed = get_modified_speed(user_pokemon, field, field.attacker_side)
    opponent_speed = get_modified_speed(opponent_pokemon, field, field.defender_side)
    u = user_speed["modifiedSpeed"]
    o = opponent_speed["modifiedSpeed"]

    if field.trick_room:
        if u < o:
            result = f"{user_pokemon.name} moves first under Trick Room."
        elif o < u:
            result = f"{opponent_pokemon.name} moves first under Trick Room."
        else:
            result = "Speed tie."
    else:
        if u > o:
            result = f"{user_pokemon.name} moves first."
        elif o > u:
            result = f"{opponent_pokemon.name} moves first."
        else:
            result = "Speed tie."

    return {"user": user_speed, "opponent": opponent_speed, "trickRoom": field.trick_room, "result": result}
