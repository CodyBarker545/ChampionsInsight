from __future__ import annotations

from math import floor

MOLD_BREAKER_ABILITIES = {"mold-breaker", "teravolt", "turboblaze"}

DEFENDER_ABILITY_IGNORE_TARGETS = {
    "armor-tail", "aroma-veil", "aura-break", "battle-armor", "big-pecks", "bulletproof",
    "clear-body", "contrary", "damp", "dazzling", "disguise", "dry-skin", "earth-eater",
    "filter", "flash-fire", "flower-gift", "flower-veil", "fluffy", "friend-guard",
    "fur-coat", "good-as-gold", "grass-pelt", "guard-dog", "heatproof", "heavy-metal",
    "hyper-cutter", "ice-face", "ice-scales", "illuminate", "immunity", "inner-focus",
    "insomnia", "keen-eye", "leaf-guard", "levitate", "light-metal", "lightning-rod",
    "limber", "magic-bounce", "magma-armor", "marvel-scale", "minds-eye", "mirror-armor",
    "motor-drive", "multiscale", "oblivious", "overcoat", "own-tempo", "pastel-veil",
    "punk-rock", "purifying-salt", "queenly-majesty", "sand-veil", "sap-sipper",
    "shell-armor", "shield-dust", "simple", "snow-cloak", "solid-rock", "soundproof",
    "sticky-hold", "storm-drain", "sturdy", "suction-cups", "sweet-veil", "tangled-feet",
    "telepathy", "tera-shell", "thermal-exchange", "thick-fat", "unaware", "vital-spirit",
    "volt-absorb", "water-absorb", "water-bubble", "water-veil", "well-baked-body",
    "white-smoke", "wind-rider", "wonder-guard", "wonder-skin",
}

ABILITY_IGNORE_MOVES = {
    "g-max-drum-solo", "g-max-fire-ball", "g-max-hydrosnipe", "light-that-burns-the-sky",
    "menacing-moonraze-maelstrom", "moongeist-beam", "photon-geyser",
    "searing-sunraze-smash", "sunsteel-strike",
}

NEUTRALIZING_GAS_IGNORED_ABILITIES = {
    "as-one-(glastrier)", "as-one-(spectrier)", "battle-bond", "comatose", "disguise",
    "gulp-missile", "ice-face", "multitype", "neutralizing-gas", "power-construct",
    "rks-system", "schooling", "shields-down", "stance-change", "tera-shift", "zen-mode",
    "zero-to-hero",
}


def normalize_name(value: object) -> str:
    return str(value or "").strip().lower().replace(" ", "-").replace("_", "-").replace("'", "")


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def poke_round(value: float) -> int:
    # Pokemon's damage engine rounds .5 down.
    return floor(value) if value % 1 <= 0.5 else floor(value) + 1


def chain_mods(mods: list[int], lower_bound: int = 41, upper_bound: int = 2_097_152) -> int:
    chained = 4096
    for mod in mods:
        if mod != 4096:
            chained = (chained * mod + 2048) >> 12
    return clamp(chained, lower_bound, upper_bound)


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def has_active_item(pokemon, field, item_name: str) -> bool:
    ability = "" if getattr(pokemon, "ability_suppressed", False) else normalize_name(getattr(pokemon, "ability", ""))
    if getattr(pokemon, "item_suppressed", False) or getattr(field, "magic_room", False) or ability == "klutz":
        return False
    return normalize_name(getattr(pokemon, "item", "")) == item_name


def ability_shield_active(pokemon, field) -> bool:
    return has_active_item(pokemon, field, "ability-shield")


def move_ignores_target_ability(move) -> bool:
    return normalize_name(getattr(move, "name", "")) in ABILITY_IGNORE_MOVES


def defender_ability_can_be_ignored(defender) -> bool:
    return normalize_name(getattr(defender, "ability", "")) in DEFENDER_ABILITY_IGNORE_TARGETS


def attacker_ignores_target_ability(attacker) -> bool:
    return (
        not getattr(attacker, "ability_suppressed", False)
        and normalize_name(getattr(attacker, "ability", "")) in MOLD_BREAKER_ABILITIES
    )


def defender_ability_is_ignored(attacker, defender, move=None, field=None) -> bool:
    if getattr(defender, "ability_suppressed", False):
        return True
    if field is not None and ability_shield_active(defender, field):
        return False
    if not defender_ability_can_be_ignored(defender):
        return False
    return attacker_ignores_target_ability(attacker) or (move is not None and move_ignores_target_ability(move))


def neutralizing_gas_can_suppress(pokemon, field) -> bool:
    if ability_shield_active(pokemon, field):
        return False
    return normalize_name(getattr(pokemon, "ability", "")) not in NEUTRALIZING_GAS_IGNORED_ABILITIES


def critical_hits(attacker, defender, field) -> bool:
    if not getattr(field, "critical", False):
        return False

    defender_ability = normalize_name(getattr(defender, "ability", ""))
    attacker_ability = normalize_name(getattr(attacker, "ability", ""))

    if defender_ability in {"battle-armor", "shell-armor"} and not defender_ability_is_ignored(attacker, defender, None, field):
        return False

    return True
