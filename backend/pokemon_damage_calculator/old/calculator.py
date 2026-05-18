from __future__ import annotations

from .damage import calculate_damage_range
from .data_loader import get_move_data
from .models import BattleField, BattleMove, BattlePokemon, SideConditions
from .move_flags import enrich_flags
from .speed import compare_speed
from .stat_calculator import calculate_all_stats, highest_non_hp_stat
from .utils import normalize_name, safe_int


def _get_bool(data: dict, camel: str, snake: str | None = None, default: bool = False) -> bool:
    snake = snake or camel
    return bool(data.get(camel, data.get(snake, default)))


def normalize_stat_keys(stats: dict) -> dict:
    if not stats:
        return {}

    return {
        "hp": safe_int(stats.get("hp", 0), 0),
        "attack": safe_int(stats.get("attack", 0), 0),
        "defense": safe_int(stats.get("defense", 0), 0),
        "special_attack": safe_int(stats.get("specialAttack", stats.get("special_attack", 0)), 0),
        "special_defense": safe_int(stats.get("specialDefense", stats.get("special_defense", 0)), 0),
        "speed": safe_int(stats.get("speed", 0), 0),
    }


def form_has_triggered_once_per_battle_ability(name: str, ability: str) -> bool:
    normalized_name = normalize_name(name)
    normalized_ability = normalize_name(ability)

    if normalized_ability == "disguise" and "busted" in normalized_name:
        return True

    if normalized_ability == "ice-face" and "noice" in normalized_name:
        return True

    return False


def build_pokemon(data: dict) -> BattlePokemon:
    level = safe_int(data.get("level", 50), 50)
    stats = normalize_stat_keys(data.get("stats", {}) or {})
    base_stats = normalize_stat_keys(data.get("baseStats", data.get("base_stats", {})) or {})
    ivs = data.get("ivs", {}) or {}
    evs = data.get("evs", {}) or {}
    nature = data.get("nature", "") or ""

    # Supports two input styles:
    # 1. final calculated stats in "stats"
    # 2. baseStats + IVs + EVs + nature, calculated here
    if not stats and base_stats:
        stats = calculate_all_stats(base_stats=base_stats, ivs=ivs, evs=evs, level=level, nature=nature)

    max_hp = data.get("maxHp", data.get("max_hp"))
    if max_hp is None:
        max_hp = stats.get("hp", 1)
    max_hp = safe_int(max_hp, 1)

    current_hp = data.get("currentHp", data.get("current_hp"))
    if current_hp is None:
        current_hp = max_hp
    current_hp = safe_int(current_hp, max_hp)

    protosynthesis_stat = data.get("protosynthesisStat", data.get("protosynthesis_stat", "")) or ""
    quark_drive_stat = data.get("quarkDriveStat", data.get("quark_drive_stat", "")) or ""
    if not protosynthesis_stat and stats:
        protosynthesis_stat = highest_non_hp_stat(stats)
    if not quark_drive_stat and stats:
        quark_drive_stat = highest_non_hp_stat(stats)

    ability = data.get("ability", "") or ""
    ability_triggered = _get_bool(data, "abilityTriggered", "ability_triggered", False)
    if form_has_triggered_once_per_battle_ability(data.get("name", ""), ability):
        ability_triggered = True

    return BattlePokemon(
        name=data["name"],
        level=level,
        types=data.get("types", []) or [],
        stats=stats,
        max_hp=max_hp,
        current_hp=current_hp,
        ability=ability,
        item=data.get("item", "") or "",
        status=data.get("status", "") or "",
        nature=nature,
        base_stats=base_stats,
        ivs=ivs,
        evs=evs,
        boosts=data.get("boosts", {}) or {},
        gender=data.get("gender", "") or "",
        is_grounded=_get_bool(data, "isGrounded", "is_grounded", True),
        has_acted=_get_bool(data, "hasActed", "has_acted", False),
        tera_type=data.get("teraType", data.get("tera_type", "")) or "",
        is_terastallized=_get_bool(data, "isTerastallized", "is_terastallized", False),
        weight_kg=data.get("weightKg", data.get("weight_kg")),
        happiness=safe_int(data.get("happiness", 255), 255),
        flash_fire_active=_get_bool(data, "flashFireActive", "flash_fire_active", False),
        ability_on=_get_bool(data, "abilityOn", "ability_on", _get_bool(data, "flashFireActive", "flash_fire_active", False)),
        helping_hand_active=_get_bool(data, "helpingHandActive", "helping_hand_active", False),
        charge_active=_get_bool(data, "chargeActive", "charge_active", False),
        used_item=_get_bool(data, "usedItem", "used_item", False),
        ability_triggered=ability_triggered,
        switched_in_this_turn=_get_bool(data, "switchedInThisTurn", "switched_in_this_turn", False),
        last_damage_taken=safe_int(data.get("lastDamageTaken", data.get("last_damage_taken", 0)), 0),
        fainted_allies=safe_int(data.get("faintedAllies", data.get("fainted_allies", 0)), 0),
        booster_energy_active=_get_bool(data, "boosterEnergyActive", "booster_energy_active", False),
        protosynthesis_stat=protosynthesis_stat,
        quark_drive_stat=quark_drive_stat,
        ability_suppressed=_get_bool(data, "abilitySuppressed", "ability_suppressed", False),
        item_suppressed=_get_bool(data, "itemSuppressed", "item_suppressed", False),
        protected=_get_bool(data, "protected", "protected", False),
        substitute=_get_bool(data, "substitute", "substitute", False),
        metronome_turns=safe_int(data.get("metronomeTurns", data.get("metronome_turns", 0)), 0),
    )


def build_move(data: dict) -> BattleMove:
    if isinstance(data, str):
        data = {"name": data}

    data = data or {}
    move_name = data.get("name") or data.get("display_name") or ""

    if not move_name:
        raise ValueError("Move name is required.")

    lookup = get_move_data(move_name) or {}

    move_type = data.get("type") or lookup.get("type") or "normal"
    category = data.get("category") or lookup.get("category") or "physical"
    power = data.get("power")
    if power is None:
        power = lookup.get("power")

    accuracy = data.get("accuracy")
    if accuracy is None:
        accuracy = lookup.get("accuracy")

    priority = data.get("priority")
    if priority is None:
        priority = lookup.get("priority", 0)

    move = BattleMove(
        name=lookup.get("display_name") or data.get("display_name") or move_name,
        type=move_type,
        category=category,
        power=power,
        accuracy=accuracy,
        priority=safe_int(priority, 0),
        hits=data.get("hits", lookup.get("hits", 1)),
        target=data.get("target", lookup.get("target", "normal")),
        makes_contact=_get_bool(data, "makesContact", "makes_contact", bool(lookup.get("makes_contact", False))),
        is_spread=_get_bool(data, "isSpread", "is_spread", bool(lookup.get("is_spread", False))),
        is_sound=_get_bool(data, "isSound", "is_sound", bool(lookup.get("is_sound", False))),
        is_punch=_get_bool(data, "isPunch", "is_punch", bool(lookup.get("is_punch", False))),
        is_bite=_get_bool(data, "isBite", "is_bite", bool(lookup.get("is_bite", False))),
        is_bullet=_get_bool(data, "isBullet", "is_bullet", bool(lookup.get("is_bullet", False))),
        is_pulse=_get_bool(data, "isPulse", "is_pulse", bool(lookup.get("is_pulse", False))),
        is_recoil=_get_bool(data, "isRecoil", "is_recoil", bool(lookup.get("is_recoil", False))),
        is_slicing=_get_bool(data, "isSlicing", "is_slicing", bool(lookup.get("is_slicing", False))),
        has_secondary_effect=_get_bool(
            data,
            "hasSecondaryEffect",
            "has_secondary_effect",
            bool(lookup.get("has_secondary_effect", False)),
        ),
        ignores_protect=_get_bool(data, "ignoresProtect", "ignores_protect", bool(lookup.get("ignores_protect", False))),
    )

    print("MOVE LOOKUP DEBUG:", {
        "input": move_name,
        "lookup": lookup,
        "resolved": {
            "name": move.name,
            "type": move.type,
            "category": move.category,
            "power": move.power,
            "accuracy": move.accuracy,
        },
    })

    enrich_flags(move)
    return move

def build_side_conditions(data: dict | None) -> SideConditions:
    data = data or {}
    return SideConditions(
        reflect=bool(data.get("reflect", False)),
        light_screen=bool(data.get("lightScreen", data.get("light_screen", False))),
        aurora_veil=bool(data.get("auroraVeil", data.get("aurora_veil", False))),
        tailwind=bool(data.get("tailwind", False)),
        friend_guard=bool(data.get("friendGuard", data.get("friend_guard", False))),
        helping_hand=bool(data.get("helpingHand", data.get("helping_hand", False))),
        battery=bool(data.get("battery", data.get("isBattery", data.get("is_battery", False)))),
        power_spot=bool(data.get("powerSpot", data.get("power_spot", data.get("isPowerSpot", data.get("is_power_spot", False))))),
        flower_gift=bool(data.get("flowerGift", data.get("flower_gift", data.get("isFlowerGift", data.get("is_flower_gift", False))))),
        steely_spirit=bool(data.get("steelySpirit", data.get("steely_spirit", data.get("isSteelySpirit", data.get("is_steely_spirit", False))))),
        stealth_rock=bool(data.get("stealthRock", data.get("stealth_rock", False))),
        spikes_layers=safe_int(data.get("spikesLayers", data.get("spikes_layers", 0)), 0),
        toxic_spikes_layers=safe_int(data.get("toxicSpikesLayers", data.get("toxic_spikes_layers", 0)), 0),
        sticky_web=bool(data.get("stickyWeb", data.get("sticky_web", False))),
    )


def build_field(data: dict | None) -> BattleField:
    data = data or {}
    return BattleField(
        weather=data.get("weather", "") or "",
        terrain=data.get("terrain", "") or "",
        critical=bool(data.get("critical", False)),
        critical_stage=safe_int(data.get("criticalStage", data.get("critical_stage", 0)), 0),
        is_doubles=bool(data.get("isDoubles", data.get("is_doubles", False))),
        gravity=bool(data.get("gravity", False)),
        trick_room=bool(data.get("trickRoom", data.get("trick_room", False))),
        wonder_room=bool(data.get("wonderRoom", data.get("wonder_room", False))),
        magic_room=bool(data.get("magicRoom", data.get("magic_room", False))),
        charge=bool(data.get("charge", False)),
        attacker_side=build_side_conditions(data.get("attackerSide", data.get("attacker_side"))),
        defender_side=build_side_conditions(data.get("defenderSide", data.get("defender_side"))),
    )


def validate_payload(payload: dict) -> None:
    if not payload.get("attacker"):
        raise ValueError("Attacker is required.")
    if not payload.get("defender"):
        raise ValueError("Defender is required.")
    if not payload.get("move"):
        raise ValueError("Move is required.")


def apply_entry_field_abilities(attacker: BattlePokemon, defender: BattlePokemon, field: BattleField) -> None:
    if any(normalize_name(pokemon.ability) == "teraform-zero" for pokemon in (attacker, defender)):
        field.weather = ""
        field.terrain = ""
        return

    if field.weather == "":
        weather_abilities = {
            "drizzle": "rain",
            "drought": "sun",
            "sand-stream": "sand",
            "snow-warning": "snow",
            "delta-stream": "strong-winds",
        }
        for pokemon in (attacker, defender):
            weather = weather_abilities.get(normalize_name(pokemon.ability))
            if weather:
                field.weather = weather

    if field.terrain == "":
        terrain_abilities = {
            "electric-surge": "electric",
            "grassy-surge": "grassy",
            "psychic-surge": "psychic",
            "misty-surge": "misty",
            "hadron-engine": "electric",
        }
        for pokemon in (attacker, defender):
            terrain = terrain_abilities.get(normalize_name(pokemon.ability))
            if terrain:
                field.terrain = terrain


def apply_entry_stat_abilities(attacker: BattlePokemon, defender: BattlePokemon) -> None:
    defender_ability = normalize_name(defender.ability)
    attacker_ability = normalize_name(attacker.ability)
    if defender_ability == "intimidate" and attacker_ability not in {"clear-body", "white-smoke", "full-metal-body", "hyper-cutter"}:
        if attacker_ability == "guard-dog":
            attacker.boosts["attack"] = attacker.boosts.get("attack", 0) + 1
        else:
            attacker.boosts["attack"] = attacker.boosts.get("attack", 0) - 1
    if attacker_ability == "intimidate" and defender_ability not in {"clear-body", "white-smoke", "full-metal-body", "hyper-cutter"}:
        if defender_ability == "guard-dog":
            defender.boosts["attack"] = defender.boosts.get("attack", 0) + 1
        else:
            defender.boosts["attack"] = defender.boosts.get("attack", 0) - 1


def analyze_battle(payload: dict) -> dict:
    validate_payload(payload)

    attacker = build_pokemon(payload["attacker"])
    defender = build_pokemon(payload["defender"])
    move = build_move(payload["move"])
    field = build_field(payload.get("field"))

    apply_entry_field_abilities(attacker, defender, field)
    apply_entry_stat_abilities(attacker, defender)

    speed_result = compare_speed(attacker, defender, field)
    damage_result = calculate_damage_range(attacker, defender, move, field)

    min_damage = damage_result.get("minDamage", 0)
    max_damage = damage_result.get("maxDamage", 0)

    damage_result["range"] = f"{min_damage} - {max_damage}"

    return {
        "attacker": attacker.name,
        "defender": defender.name,
        "move": move.name,
        "speed": speed_result,
        "damage": damage_result,
    }
