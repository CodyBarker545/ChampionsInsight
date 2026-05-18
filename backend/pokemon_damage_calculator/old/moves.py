from __future__ import annotations

from math import floor

from .items import GEMS, SPECIES_TYPE_BOOST_ITEMS, TYPE_BOOST_ITEMS
from .models import BattleField, BattleMove, BattlePokemon
from .utils import chain_mods, defender_ability_is_ignored, normalize_name, poke_round

PLATE_TYPES = {
    "flame-plate": "fire", "splash-plate": "water", "zap-plate": "electric", "meadow-plate": "grass",
    "icicle-plate": "ice", "fist-plate": "fighting", "toxic-plate": "poison", "earth-plate": "ground",
    "sky-plate": "flying", "mind-plate": "psychic", "insect-plate": "bug", "stone-plate": "rock",
    "spooky-plate": "ghost", "draco-plate": "dragon", "dread-plate": "dark", "iron-plate": "steel",
    "pixie-plate": "fairy",
}
MEMORY_TYPES = {
    "fighting-memory": "fighting", "flying-memory": "flying", "poison-memory": "poison",
    "ground-memory": "ground", "rock-memory": "rock", "bug-memory": "bug", "ghost-memory": "ghost",
    "steel-memory": "steel", "fire-memory": "fire", "water-memory": "water", "grass-memory": "grass",
    "electric-memory": "electric", "psychic-memory": "psychic", "ice-memory": "ice",
    "dragon-memory": "dragon", "dark-memory": "dark", "fairy-memory": "fairy",
}
ABILITY_TYPE_CHANGE_BLOCKED_MOVES = {
    "judgment", "multi-attack", "natural-gift", "nature-power", "revelation-dance",
    "struggle", "techno-blast", "terrain-pulse", "weather-ball",
}


def get_move_type(move: BattleMove, attacker: BattlePokemon, field: BattleField) -> tuple[str, list[str]]:
    name = normalize_name(move.name)
    base_type = normalize_name(move.type)
    ability = "" if attacker.ability_suppressed else normalize_name(attacker.ability)
    item = normalize_name(attacker.item) if not attacker.item_suppressed and not field.magic_room and ability != "klutz" else ""
    notes: list[str] = []
    type_change_blocked = name in ABILITY_TYPE_CHANGE_BLOCKED_MOVES or (name == "tera-blast" and attacker.tera_type)

    # Ability-based type conversions.
    if not type_change_blocked and ability == "pixilate" and base_type == "normal":
        return "fairy", ["Pixilate changes Normal to Fairy."]
    if not type_change_blocked and ability == "refrigerate" and base_type == "normal":
        return "ice", ["Refrigerate changes Normal to Ice."]
    if not type_change_blocked and ability == "aerilate" and base_type == "normal":
        return "flying", ["Aerilate changes Normal to Flying."]
    if not type_change_blocked and ability == "galvanize" and base_type == "normal":
        return "electric", ["Galvanize changes Normal to Electric."]
    if not type_change_blocked and ability == "liquid-voice" and move.is_sound:
        return "water", ["Liquid Voice changes sound moves to Water."]
    if not type_change_blocked and ability == "dragonize" and base_type == "normal":
        return "dragon", ["Dragonize changes Normal to Dragon."]
    if not type_change_blocked and ability == "normalize":
        return "normal", ["Normalize changes move type to Normal."]

    if name == "weather-ball":
        if field.weather == "rain": return "water", ["Weather Ball becomes Water in rain."]
        if field.weather == "sun": return "fire", ["Weather Ball becomes Fire in sun."]
        if field.weather == "sand": return "rock", ["Weather Ball becomes Rock in sand."]
        if field.weather == "snow": return "ice", ["Weather Ball becomes Ice in snow."]
        if field.weather == "harsh-sun": return "fire", ["Weather Ball becomes Fire in harsh sun."]
        if field.weather == "heavy-rain": return "water", ["Weather Ball becomes Water in heavy rain."]

    if name == "terrain-pulse" and attacker.is_grounded:
        if field.terrain == "electric": return "electric", ["Terrain Pulse becomes Electric."]
        if field.terrain == "grassy": return "grass", ["Terrain Pulse becomes Grass."]
        if field.terrain == "psychic": return "psychic", ["Terrain Pulse becomes Psychic."]
        if field.terrain == "misty": return "fairy", ["Terrain Pulse becomes Fairy."]

    if name == "tera-blast" and attacker.is_terastallized and attacker.tera_type:
        return normalize_name(attacker.tera_type), ["Tera Blast becomes the user's Tera type."]
    if name == "judgment" and item in PLATE_TYPES:
        return PLATE_TYPES[item], ["Judgment type follows held Plate."]
    if name == "multi-attack" and item in MEMORY_TYPES:
        return MEMORY_TYPES[item], ["Multi-Attack type follows held Memory."]

    return base_type, notes


def weight_power(weight_kg: float | None) -> int:
    w = weight_kg or 0
    if w >= 200: return 120
    if w >= 100: return 100
    if w >= 50: return 80
    if w >= 25: return 60
    if w >= 10: return 40
    return 20


def relative_weight_power(attacker_weight: float | None, defender_weight: float | None) -> int:
    a = attacker_weight or 1
    d = defender_weight or 1
    ratio = d / a
    if ratio >= 5: return 120
    if ratio >= 4: return 100
    if ratio >= 3: return 80
    if ratio >= 2: return 60
    return 40


def defender_bp_modifier_ability_ignored(attacker: BattlePokemon, defender: BattlePokemon, move: BattleMove, field: BattleField) -> bool:
    return defender_ability_is_ignored(attacker, defender, move, field)


def apply_base_power_modifiers(
    base_power: int,
    move: BattleMove,
    attacker: BattlePokemon,
    defender: BattlePokemon,
    field: BattleField,
    move_type: str,
    notes: list[str],
) -> int:
    if base_power <= 0:
        return base_power

    name = normalize_name(move.name)
    category = normalize_name(move.category)
    attacker_ability = "" if attacker.ability_suppressed else normalize_name(attacker.ability)
    defender_ability = "" if defender.ability_suppressed else normalize_name(defender.ability)
    item = normalize_name(attacker.item) if not attacker.item_suppressed and not field.magic_room and attacker_ability != "klutz" else ""
    mods: list[int] = []

    def add(mod: int, note: str) -> None:
        mods.append(mod)
        notes.append(note)

    if field.attacker_side.helping_hand:
        add(6144, "Helping Hand")
    if field.attacker_side.battery and category == "special":
        add(5325, "Battery")
    if field.attacker_side.power_spot:
        add(5325, "Power Spot")

    if attacker.is_grounded:
        if field.terrain in {"electric", "grassy", "psychic"} and move_type == field.terrain:
            add(5325, f"{field.terrain.title()} Terrain")
    if defender.is_grounded:
        if (field.terrain == "misty" and move_type == "dragon") or (field.terrain == "grassy" and name in {"earthquake", "bulldoze", "magnitude"}):
            add(2048, f"{field.terrain.title()} Terrain")

    if attacker_ability:
        if (
            (attacker_ability == "technician" and base_power <= 60)
            or (attacker_ability == "flare-boost" and normalize_name(attacker.status) == "burn" and category == "special")
            or (attacker_ability == "toxic-boost" and normalize_name(attacker.status) in {"poison", "bad-poison"} and category == "physical")
            or (attacker_ability == "mega-launcher" and move.is_pulse)
            or (attacker_ability == "strong-jaw" and move.is_bite)
            or (attacker_ability == "steely-spirit" and move_type == "steel")
            or (attacker_ability == "sharpness" and move.is_slicing)
        ):
            add(6144, attacker.ability)
        elif (
            (attacker_ability == "sheer-force" and move.has_secondary_effect)
            or (attacker_ability == "sand-force" and field.weather == "sand" and move_type in {"rock", "ground", "steel"})
            or (attacker_ability == "analytic" and attacker.has_acted)
            or (attacker_ability == "tough-claws" and move.makes_contact)
            or (attacker_ability == "punk-rock" and move.is_sound)
        ):
            add(5325, attacker.ability)

        attacker_gender = normalize_name(attacker.gender)
        defender_gender = normalize_name(defender.gender)
        if attacker_ability == "rivalry" and attacker_gender and defender_gender and attacker_gender not in {"genderless", "unknown", "n"} and defender_gender not in {"genderless", "unknown", "n"}:
            add(5120 if attacker_gender == defender_gender else 3072, "Rivalry")

        type_change_blocked = name in ABILITY_TYPE_CHANGE_BLOCKED_MOVES or (name == "tera-blast" and attacker.tera_type)
        if attacker_ability in {"pixilate", "refrigerate", "aerilate", "galvanize", "dragonize"} and normalize_name(move.type) == "normal" and move_type != "normal":
            add(4915, f"{attacker.ability} damage boost")
        if attacker_ability == "normalize" and not type_change_blocked:
            add(4915, "Normalize damage boost")

        if (attacker_ability == "reckless" and move.is_recoil) or (attacker_ability == "iron-fist" and move.is_punch):
            add(4915, attacker.ability)

        if attacker_ability == "supreme-overlord" and attacker.fainted_allies:
            add([4096, 4506, 4915, 5325, 5734, 6144][min(5, max(0, attacker.fainted_allies))], "Supreme Overlord")

    if field.attacker_side.steely_spirit and move_type == "steel":
        add(6144, "Ally Steely Spirit")

    attacker_aura = attacker_ability
    defender_aura = defender_ability
    aura_ability = f"{move_type}-aura"
    if move_type in {"fairy", "dark"} and aura_ability in {attacker_aura, defender_aura}:
        aura_break = "aura-break" in {attacker_aura, defender_aura}
        add(3072 if aura_break else 5448, "Aura Break" if aura_break else aura_ability.replace("-", " ").title())

    if not defender_bp_modifier_ability_ignored(attacker, defender, move, field) and defender_ability == "dry-skin" and move_type == "fire":
        add(5120, "Dry Skin Fire weakness")

    if GEMS.get(item) == move_type:
        add(5325, f"{attacker.item} gem boost")
    elif TYPE_BOOST_ITEMS.get(item) == move_type:
        add(4915, f"{attacker.item} type boost")
    else:
        species_boosts = SPECIES_TYPE_BOOST_ITEMS.get(item, {})
        boosted_types = species_boosts.get(normalize_name(attacker.name), set())
        if move_type in boosted_types:
            add(4915, f"{attacker.item} species boost")
        elif item == "muscle-band" and category == "physical":
            add(4505, "Muscle Band")
        elif item == "wise-glasses" and category == "special":
            add(4505, "Wise Glasses")
        elif item == "punching-glove" and move.is_punch:
            add(4506, "Punching Glove")

    if not mods:
        return base_power
    return max(1, poke_round(base_power * chain_mods(mods) / 4096))


def get_base_power(move: BattleMove, attacker: BattlePokemon, defender: BattlePokemon, field: BattleField, move_type: str | None = None) -> tuple[int, list[str]]:
    name = normalize_name(move.name)
    power = int(move.power or 0)
    notes: list[str] = []
    resolved_move_type = move_type or normalize_name(move.type)

    def finalize(base_power: int, current_notes: list[str]) -> tuple[int, list[str]]:
        return apply_base_power_modifiers(base_power, move, attacker, defender, field, resolved_move_type, current_notes), current_notes

    if power <= 0 and name not in {
        "low-kick", "grass-knot", "heavy-slam", "heat-crash", "electro-ball", "gyro-ball",
        "flail", "reversal", "eruption", "water-spout", "crush-grip", "wring-out",
        "stored-power", "power-trip", "punishment", "beat-up", "return", "frustration",
        "trump-card", "natural-gift", "fling"
    }:
        return 0, notes

    if name in {"low-kick", "grass-knot"}:
        return finalize(weight_power(defender.weight_kg), [f"{move.name} power is based on defender weight."])
    if name in {"heavy-slam", "heat-crash"}:
        return finalize(relative_weight_power(attacker.weight_kg, defender.weight_kg), [f"{move.name} power is based on relative weight."])
    if name == "electro-ball":
        a = max(1, attacker.stats.get("speed", 1))
        d = max(1, defender.stats.get("speed", 1))
        ratio = a / d
        if ratio >= 4: return finalize(150, ["Electro Ball power from speed ratio."])
        if ratio >= 3: return finalize(120, ["Electro Ball power from speed ratio."])
        if ratio >= 2: return finalize(80, ["Electro Ball power from speed ratio."])
        if ratio >= 1: return finalize(60, ["Electro Ball power from speed ratio."])
        return finalize(40, ["Electro Ball power from speed ratio."])
    if name == "gyro-ball":
        p = min(150, max(1, floor(25 * max(1, defender.stats.get("speed", 1)) / max(1, attacker.stats.get("speed", 1))) + 1))
        return finalize(p, ["Gyro Ball power from inverse speed ratio."])
    if name in {"eruption", "water-spout"}:
        hp_ratio = attacker.hp() / max(1, attacker.max_hp)
        return finalize(max(1, floor(150 * hp_ratio)), [f"{move.name} power scales with user's current HP."])
    if name in {"flail", "reversal"}:
        ratio = attacker.hp() / max(1, attacker.max_hp)
        if ratio <= 1/48: return finalize(200, [f"{move.name} power scales with low HP."])
        if ratio <= 4/48: return finalize(150, [f"{move.name} power scales with low HP."])
        if ratio <= 9/48: return finalize(100, [f"{move.name} power scales with low HP."])
        if ratio <= 16/48: return finalize(80, [f"{move.name} power scales with low HP."])
        if ratio <= 32/48: return finalize(40, [f"{move.name} power scales with low HP."])
        return finalize(20, [f"{move.name} power scales with low HP."])
    if name in {"crush-grip", "wring-out"}:
        return finalize(max(1, floor(120 * defender.hp() / max(1, defender.max_hp))), [f"{move.name} power scales with target HP."])
    if name in {"stored-power", "power-trip"}:
        positive_boosts = sum(max(0, s) for s in attacker.boosts.values())
        return finalize(20 + 20 * positive_boosts, [f"{move.name} power scales with positive stat boosts."])
    if name == "punishment":
        positive_boosts = sum(max(0, s) for s in defender.boosts.values())
        return finalize(min(200, 60 + 20 * positive_boosts), ["Punishment power scales with defender boosts."])
    if name == "return":
        return finalize(max(1, floor(attacker.happiness * 10 / 25)), ["Return power uses happiness."])
    if name == "frustration":
        return finalize(max(1, floor((255 - attacker.happiness) * 10 / 25)), ["Frustration power uses inverse happiness."])
    if name == "facade" and attacker.status in {"burn", "poison", "bad-poison", "paralysis"}:
        return finalize(power * 2, ["Facade doubles when the user has a major status."])
    if name == "hex" and defender.status:
        return finalize(power * 2, ["Hex doubles against a statused target."])
    if name == "venoshock" and defender.status in {"poison", "bad-poison"}:
        return finalize(power * 2, ["Venoshock doubles against poisoned target."])
    if name == "brine" and defender.hp() <= defender.max_hp / 2:
        return finalize(power * 2, ["Brine doubles below half HP."])
    if name == "acrobatics" and not attacker.item:
        return finalize(power * 2, ["Acrobatics doubles with no held item."])
    if name == "knock-off" and defender.item and not defender.item_suppressed and not field.magic_room:
        return finalize(floor(power * 1.5), ["Knock Off boosts when target has a removable item."])
    if name == "weather-ball" and field.weather:
        return finalize(100, ["Weather Ball power is 100 during weather."])
    if name == "terrain-pulse" and field.terrain and attacker.is_grounded:
        return finalize(100, ["Terrain Pulse power is 100 on terrain when grounded."])
    if name == "expanding-force" and field.terrain == "psychic" and attacker.is_grounded:
        return finalize(floor(power * 1.5), ["Expanding Force is boosted on Psychic Terrain."])
    if name == "misty-explosion" and field.terrain == "misty" and attacker.is_grounded:
        return finalize(floor(power * 1.5), ["Misty Explosion is boosted on Misty Terrain."])
    if name == "rising-voltage" and field.terrain == "electric" and defender.is_grounded:
        return finalize(power * 2, ["Rising Voltage doubles on Electric Terrain against grounded target."])
    if name == "solar-beam" and field.weather in {"rain", "sand", "snow"}:
        return finalize(floor(power * 0.5), ["Solar Beam is weakened by this weather."])

    return finalize(power, notes)


def get_hit_count(move: BattleMove, attacker: BattlePokemon) -> tuple[list[int], list[float], list[str]]:
    name = normalize_name(move.name)
    ability = "" if attacker.ability_suppressed else normalize_name(attacker.ability)
    item = normalize_name(attacker.item) if not attacker.item_suppressed and ability != "klutz" else ""
    notes: list[str] = []

    if ability == "parental-bond" and normalize_name(move.category) != "status":
        return [1], [1.0], ["Parental Bond second-hit damage is approximated separately."]
    if isinstance(move.hits, int) and move.hits > 1:
        return [move.hits], [1.0], notes
    if name in {"double-kick", "dual-chop", "bonemerang", "gear-grind", "twin-beam"}:
        return [2], [1.0], ["Fixed two-hit move."]
    if name == "triple-axel":
        return [1, 2, 3], [1/3, 1/3, 1/3], ["Triple Axel hit count simplified."]
    if name == "population-bomb":
        if item == "loaded-dice":
            return [4, 5, 6, 7, 8, 9, 10], [1/7] * 7, ["Population Bomb with Loaded Dice simplified."]
        return list(range(1, 11)), [0.1] * 10, ["Population Bomb hit count simplified."]
    if name in {"bullet-seed", "icicle-spear", "pin-missile", "rock-blast", "scale-shot", "tail-slap", "water-shuriken"}:
        if ability == "skill-link":
            return [5], [1.0], ["Skill Link maximizes 2-5 hit moves."]
        if item == "loaded-dice":
            return [4, 5], [0.5, 0.5], ["Loaded Dice causes mostly 4-5 hits."]
        return [2, 3, 4, 5], [3/8, 3/8, 1/8, 1/8], ["Standard 2-5 hit distribution."]
    return [1], [1.0], notes
