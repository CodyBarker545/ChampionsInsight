from __future__ import annotations

from math import floor

from .models import BattleField, BattleMove, BattlePokemon
from .utils import chain_mods, clamp, critical_hits, defender_ability_is_ignored, normalize_name, poke_round


def apply_stat_stage(stat: int, stage: int) -> int:
    stage = clamp(stage, -6, 6)
    if stage >= 0:
        return max(1, floor(stat * (2 + stage) / 2))
    return max(1, floor(stat * 2 / (2 - stage)))


def apply_paradox_stat_modifier(pokemon: BattlePokemon, stat_name: str, stat: int, field: BattleField, notes: list[str]) -> int:
    if pokemon.ability_suppressed:
        return stat
    ability = normalize_name(pokemon.ability)
    item = normalize_name(pokemon.item) if not pokemon.item_suppressed and not field.magic_room else ""

    proto_active = ability == "protosynthesis" and (field.weather in {"sun", "harsh-sun"} or pokemon.booster_energy_active or item == "booster-energy")
    quark_active = ability == "quark-drive" and (field.terrain == "electric" or pokemon.booster_energy_active or item == "booster-energy")

    proto_stat = str(pokemon.protosynthesis_stat or "").strip().lower().replace("-", "_")
    quark_stat = str(pokemon.quark_drive_stat or "").strip().lower().replace("-", "_")
    if proto_active and proto_stat == stat_name:
        multiplier = 1.5 if stat_name == "speed" else 1.3
        notes.append(f"Protosynthesis boosts {stat_name}.")
        return floor(stat * multiplier)
    if quark_active and quark_stat == stat_name:
        multiplier = 1.5 if stat_name == "speed" else 1.3
        notes.append(f"Quark Drive boosts {stat_name}.")
        return floor(stat * multiplier)
    return stat


def is_species(pokemon: BattlePokemon, *names: str) -> bool:
    pokemon_name = normalize_name(pokemon.name)
    return pokemon_name in {normalize_name(name) for name in names}


def terrain_seed_boosts(item: str, field: BattleField, defense_key: str) -> bool:
    return (
        (item == "electric-seed" and field.terrain == "electric" and defense_key == "defense")
        or (item == "grassy-seed" and field.terrain == "grassy" and defense_key == "defense")
        or (item == "misty-seed" and field.terrain == "misty" and defense_key == "special_defense")
        or (item == "psychic-seed" and field.terrain == "psychic" and defense_key == "special_defense")
    )


def triggered_offensive_item_stage(item: str, stat_name: str) -> int:
    if item == "weakness-policy" and stat_name in {"attack", "special_attack"}:
        return 2
    if item == "liechi-berry" and stat_name == "attack":
        return 1
    if item == "petaya-berry" and stat_name == "special_attack":
        return 1
    if item == "cell-battery" and stat_name == "attack":
        return 1
    if item == "snowball" and stat_name == "attack":
        return 1
    if item == "absorb-bulb" and stat_name == "special_attack":
        return 1
    return 0


def triggered_defensive_item_stage(item: str, defense_key: str) -> int:
    if item == "ganlon-berry" and defense_key == "defense":
        return 1
    if item == "apicot-berry" and defense_key == "special_defense":
        return 1
    if item == "kee-berry" and defense_key == "defense":
        return 1
    if item == "maranga-berry" and defense_key == "special_defense":
        return 1
    if item == "luminous-moss" and defense_key == "special_defense":
        return 1
    return 0


def defender_attack_modifier_ability_ignored(attacker: BattlePokemon, defender: BattlePokemon, move: BattleMove, field: BattleField) -> bool:
    return defender_ability_is_ignored(attacker, defender, move, field)


def defender_ability_ignored(attacker: BattlePokemon, defender: BattlePokemon, move: BattleMove, field: BattleField) -> bool:
    return defender_ability_is_ignored(attacker, defender, move, field)


def apply_attack_stat_modifiers(
    stat: int,
    attacker: BattlePokemon,
    defender: BattlePokemon,
    move: BattleMove,
    field: BattleField,
    move_type: str,
    notes: list[str],
) -> int:
    name = normalize_name(move.name)
    category = normalize_name(move.category)
    ability = "" if attacker.ability_suppressed else normalize_name(attacker.ability)
    item = normalize_name(attacker.item) if not attacker.item_suppressed and not field.magic_room and ability != "klutz" else ""
    mods: list[int] = []

    def add(mod: int, note: str) -> None:
        mods.append(mod)
        notes.append(note)

    low_hp_type_boosts = {
        "overgrow": "grass",
        "blaze": "fire",
        "torrent": "water",
        "swarm": "bug",
    }
    is_physical_attack = category == "physical" or name in {"foul-play", "body-press"}
    if is_physical_attack:
        if item == "choice-band":
            add(6144, "Choice Band")
        if item == "light-ball" and is_species(attacker, "pikachu"):
            add(8192, "Light Ball")
        if item == "thick-club" and is_species(attacker, "cubone", "marowak", "marowak-alola", "marowak-alola-totem"):
            add(8192, "Thick Club")
        if ability in {"huge-power", "pure-power"}:
            add(8192, "Huge/Pure Power")
        if ability == "guts" and attacker.status:
            add(6144, "Guts")
        if ability == "hustle":
            add(6144, "Hustle")
        if ability == "gorilla-tactics":
            add(6144, "Gorilla Tactics")
        if ability == "defeatist" and attacker.hp() <= attacker.max_hp / 2:
            add(2048, "Defeatist")
        if ability == "slow-start" and not attacker.ability_triggered:
            add(2048, "Slow Start")
        if ability == "flower-gift" and field.weather in {"sun", "harsh-sun"}:
            add(6144, "Flower Gift")
        if ability == "intrepid-sword" and not attacker.ability_triggered:
            add(6144, "Intrepid Sword")
        if ability == "download" and defender.stats.get("defense", 1) < defender.stats.get("special_defense", 1):
            add(6144, "Download")
        if ability == "orichalcum-pulse" and field.weather in {"sun", "harsh-sun"}:
            add(5461, "Orichalcum Pulse")
        if field.attacker_side.flower_gift and ability != "flower-gift" and field.weather in {"sun", "harsh-sun"}:
            add(6144, "Ally Flower Gift")
    else:
        if item == "choice-specs":
            add(6144, "Choice Specs")
        if item == "light-ball" and is_species(attacker, "pikachu"):
            add(8192, "Light Ball")
        if item == "deep-sea-tooth" and is_species(attacker, "clamperl"):
            add(8192, "Deep Sea Tooth")
        if ability == "solar-power" and field.weather in {"sun", "harsh-sun"}:
            add(6144, "Solar Power")
        if ability == "hadron-engine" and field.terrain == "electric":
            add(5461, "Hadron Engine")
        if ability == "defeatist" and attacker.hp() <= attacker.max_hp / 2:
            add(2048, "Defeatist")
        if ability == "download" and defender.stats.get("special_defense", 1) <= defender.stats.get("defense", 1):
            add(6144, "Download")
        if ability in {"plus", "minus"} and attacker.ability_on:
            add(6144, attacker.ability)

    if low_hp_type_boosts.get(ability) == move_type and attacker.hp() <= attacker.max_hp / 3:
        add(6144, attacker.ability)
    if ability == "flash-fire" and (attacker.flash_fire_active or attacker.ability_on) and move_type == "fire":
        add(6144, "Flash Fire active")
    if ability in {"steelworker", "steely-spirit"} and move_type == "steel":
        add(6144, attacker.ability)
    if ability == "dragons-maw" and move_type == "dragon":
        add(6144, "Dragon's Maw")
    if ability == "rocky-payload" and move_type == "rock":
        add(6144, "Rocky Payload")
    if ability == "transistor" and move_type == "electric":
        add(5325, "Transistor")
    if ability == "stakeout" and (defender.switched_in_this_turn or attacker.ability_on):
        add(8192, "Stakeout")
    if ability == "water-bubble" and move_type == "water":
        add(8192, "Water Bubble")

    if not defender_attack_modifier_ability_ignored(attacker, defender, move, field):
        defender_ability = normalize_name(defender.ability)
        if defender_ability == "thick-fat" and move_type in {"fire", "ice"}:
            add(2048, "Thick Fat")
        if defender_ability == "heatproof" and move_type == "fire":
            add(2048, "Heatproof")
        if defender_ability == "water-bubble" and move_type == "fire":
            add(2048, "Water Bubble")
        if defender_ability == "purifying-salt" and move_type == "ghost":
            add(2048, "Purifying Salt")

    defender_ability = "" if defender.ability_suppressed else normalize_name(defender.ability)
    if (defender_ability == "tablets-of-ruin" and is_physical_attack) or (defender_ability == "vessel-of-ruin" and not is_physical_attack):
        add(3072, defender.ability)

    if not mods:
        return stat
    return max(1, poke_round(stat * chain_mods(mods, 410, 131072) / 4096))


def get_attack_stat(attacker: BattlePokemon, defender: BattlePokemon, move: BattleMove, field: BattleField, move_type: str | None = None) -> tuple[int, str, list[str]]:
    name = normalize_name(move.name)
    category = normalize_name(move.category)
    notes: list[str] = []

    if name == "foul-play":
        raw = defender.stats.get("attack", 1)
        stage = defender.boosts.get("attack", 0)
        stat_name = "defender_attack"
        notes.append("Foul Play uses defender Attack.")
    elif name == "body-press":
        raw = attacker.stats.get("defense", 1)
        stage = attacker.boosts.get("defense", 0)
        stat_name = "attacker_defense"
        notes.append("Body Press uses attacker Defense as the attacking stat.")
    elif name in {"psyshock", "psystrike", "secret-sword"}:
        raw = attacker.stats.get("special_attack", 1)
        stage = attacker.boosts.get("special_attack", 0)
        stat_name = "special_attack"
    elif category == "physical":
        raw = attacker.stats.get("attack", 1)
        stage = attacker.boosts.get("attack", 0)
        stat_name = "attack"
    else:
        raw = attacker.stats.get("special_attack", 1)
        stage = attacker.boosts.get("special_attack", 0)
        stat_name = "special_attack"

    if ability := ("" if attacker.ability_suppressed else normalize_name(attacker.ability)):
        if ability == "simple":
            stage *= 2

    if normalize_name(defender.ability) == "unaware" and not defender_ability_ignored(attacker, defender, move, field):
        if stage > 0:
            notes.append("Unaware ignores attacker's positive offensive boosts.")
            stage = 0

    if critical_hits(attacker, defender, field) and stage < 0:
        notes.append("Critical hit ignores negative offensive stages.")
        stage = 0

    stat = apply_stat_stage(raw, stage)
    attacker_ability = "" if attacker.ability_suppressed else normalize_name(attacker.ability)
    item = normalize_name(attacker.item) if not attacker.item_suppressed and not field.magic_room and attacker_ability != "klutz" else ""
    triggered_stage = triggered_offensive_item_stage(item, stat_name) if attacker.used_item else 0
    if triggered_stage:
        stat = apply_stat_stage(stat, triggered_stage); notes.append(f"{attacker.item} activated")

    # Booster Energy / Protosynthesis / Quark Drive can boost offensive stats after nature/EV/IV stats are built.
    stat = apply_paradox_stat_modifier(attacker, stat_name, stat, field, notes)
    stat = apply_attack_stat_modifiers(stat, attacker, defender, move, field, move_type or normalize_name(move.type), notes)

    return max(1, stat), stat_name, notes


def get_defense_stat(defender: BattlePokemon, attacker: BattlePokemon, move: BattleMove, field: BattleField) -> tuple[int, str, list[str]]:
    name = normalize_name(move.name)
    category = normalize_name(move.category)
    notes: list[str] = []

    if field.wonder_room:
        defense_key = "special_defense" if category == "physical" else "defense"
        notes.append("Wonder Room swaps Defense and Sp. Def.")
    elif name in {"psyshock", "psystrike", "secret-sword"}:
        defense_key = "defense"
        notes.append(f"{move.name} targets Defense.")
    elif category == "physical":
        defense_key = "defense"
    else:
        defense_key = "special_defense"

    raw = defender.stats.get(defense_key, 1)
    stage = defender.boosts.get(defense_key, 0)
    ability_ignored = defender_ability_ignored(attacker, defender, move, field)
    if normalize_name(defender.ability) == "simple" and not ability_ignored:
        stage *= 2

    if normalize_name(attacker.ability) == "unaware" and not attacker.ability_suppressed:
        if stage > 0:
            notes.append("Unaware ignores defender's positive defensive boosts.")
            stage = 0

    if critical_hits(attacker, defender, field) and stage > 0:
        notes.append("Critical hit ignores positive defensive stages.")
        stage = 0

    stat = apply_stat_stage(raw, stage)
    ability = "" if ability_ignored else normalize_name(defender.ability)
    item = normalize_name(defender.item) if not defender.item_suppressed and not field.magic_room and ability != "klutz" else ""

    triggered_stage = triggered_defensive_item_stage(item, defense_key) if defender.used_item else 0
    if triggered_stage:
        stat = apply_stat_stage(stat, triggered_stage); notes.append(f"{defender.item} activated")

    mods: list[int] = []

    def add(mod: int, note: str) -> None:
        mods.append(mod)
        notes.append(note)

    if defense_key == "special_defense":
        if item == "assault-vest":
            add(6144, "Assault Vest")
        if item == "deep-sea-scale" and is_species(defender, "clamperl"):
            add(8192, "Deep Sea Scale")
        if field.weather == "sand" and "rock" in [normalize_name(t) for t in defender.types]:
            stat = poke_round(stat * 1.5); notes.append("Sandstorm Rock Sp. Def boost")
        if ability == "flower-gift" and field.weather in {"sun", "harsh-sun"}:
            add(6144, "Flower Gift")
        if field.defender_side.flower_gift and ability != "flower-gift" and field.weather in {"sun", "harsh-sun"}:
            add(6144, "Ally Flower Gift")
    else:
        if field.weather == "snow" and "ice" in [normalize_name(t) for t in defender.types]:
            stat = poke_round(stat * 1.5); notes.append("Snow Ice Defense boost")
        if ability == "grass-pelt" and field.terrain == "grassy":
            add(6144, "Grass Pelt")
        if ability == "marvel-scale" and defender.status:
            add(6144, "Marvel Scale")
        if item == "metal-powder" and is_species(defender, "ditto"):
            add(8192, "Metal Powder")

    if item == "eviolite":
        add(6144, "Eviolite")
    if ability == "fur-coat" and defense_key == "defense":
        add(8192, "Fur Coat")
    if ability == "dauntless-shield" and defense_key == "defense" and not defender.ability_triggered:
        add(6144, "Dauntless Shield")
    if terrain_seed_boosts(item, field, defense_key):
        add(6144, defender.item)

    attacker_ability = "" if attacker.ability_suppressed else normalize_name(attacker.ability)
    if (attacker_ability == "sword-of-ruin" and defense_key == "defense") or (attacker_ability == "beads-of-ruin" and defense_key == "special_defense"):
        add(3072, attacker.ability)

    stat = apply_paradox_stat_modifier(defender, defense_key, stat, field, notes)
    if mods:
        stat = max(1, poke_round(stat * chain_mods(mods, 410, 131072) / 4096))

    return max(1, stat), defense_key, notes
