import json
import math

from paths import POKEMON_BATTLE_DATA_PATH


with open(POKEMON_BATTLE_DATA_PATH, "r", encoding="utf-8") as file:
    POKEMON_DATA = json.load(file)


NATURES = {
    "hardy": {},
    "lonely": {"up": "attack", "down": "defense"},
    "brave": {"up": "attack", "down": "speed"},
    "adamant": {"up": "attack", "down": "special_attack"},
    "naughty": {"up": "attack", "down": "special_defense"},

    "bold": {"up": "defense", "down": "attack"},
    "docile": {},
    "relaxed": {"up": "defense", "down": "speed"},
    "impish": {"up": "defense", "down": "special_attack"},
    "lax": {"up": "defense", "down": "special_defense"},

    "timid": {"up": "speed", "down": "attack"},
    "hasty": {"up": "speed", "down": "defense"},
    "serious": {},
    "jolly": {"up": "speed", "down": "special_attack"},
    "naive": {"up": "speed", "down": "special_defense"},

    "modest": {"up": "special_attack", "down": "attack"},
    "mild": {"up": "special_attack", "down": "defense"},
    "quiet": {"up": "special_attack", "down": "speed"},
    "bashful": {},
    "rash": {"up": "special_attack", "down": "special_defense"},

    "calm": {"up": "special_defense", "down": "attack"},
    "gentle": {"up": "special_defense", "down": "defense"},
    "sassy": {"up": "special_defense", "down": "speed"},
    "careful": {"up": "special_defense", "down": "special_attack"},
    "quirky": {},
}


def normalize_name(name):
    return "".join(char.lower() for char in str(name) if char.isalnum())


def nature_multiplier(stat_name, nature_name):
    nature = NATURES.get(str(nature_name).lower(), {})

    if nature.get("up") == stat_name:
        return 1.1

    if nature.get("down") == stat_name:
        return 0.9

    return 1.0


def calculate_hp(base_hp, level=50, iv=31):
    return math.floor(((2 * base_hp + iv) * level) / 100) + level + 10


def calculate_stat(base_stat, stat_name, nature_name, level=50, iv=31):
    raw_stat = math.floor(((2 * base_stat + iv) * level) / 100) + 5
    return math.floor(raw_stat * nature_multiplier(stat_name, nature_name))


def get_value(pokemon, possible_keys, default=None):
    for key in possible_keys:
        if key in pokemon and pokemon[key] is not None:
            return pokemon[key]

    return default


def get_base_stats_from_pokemon(pokemon):
    nested_stats = (
        pokemon.get("baseStats")
        or pokemon.get("base_stats")
        or pokemon.get("stats")
        or {}
    )

    hp = get_value(pokemon, ["hp", "base_hp", "HP"])
    attack = get_value(pokemon, ["attack", "base_attack", "Attack"])
    defense = get_value(pokemon, ["defense", "base_defense", "Defense"])
    special_attack = get_value(
        pokemon,
        ["special_attack", "specialAttack", "sp_attack", "spAtk", "Sp. Atk"]
    )
    special_defense = get_value(
        pokemon,
        ["special_defense", "specialDefense", "sp_defense", "spDef", "Sp. Def"]
    )
    speed = get_value(pokemon, ["speed", "base_speed", "Speed"])

    if hp is None:
        hp = get_value(nested_stats, ["hp", "base_hp", "HP"])
    if attack is None:
        attack = get_value(nested_stats, ["attack", "base_attack", "Attack"])
    if defense is None:
        defense = get_value(nested_stats, ["defense", "base_defense", "Defense"])
    if special_attack is None:
        special_attack = get_value(
            nested_stats,
            ["special_attack", "specialAttack", "sp_attack", "spAtk", "Sp. Atk"]
        )
    if special_defense is None:
        special_defense = get_value(
            nested_stats,
            ["special_defense", "specialDefense", "sp_defense", "spDef", "Sp. Def"]
        )
    if speed is None:
        speed = get_value(nested_stats, ["speed", "base_speed", "Speed"])

    missing = []
    values = {
        "hp": hp,
        "attack": attack,
        "defense": defense,
        "special_attack": special_attack,
        "special_defense": special_defense,
        "speed": speed,
    }

    for stat_name, value in values.items():
        if value is None:
            missing.append(stat_name)

    if missing:
        raise KeyError(
            f"Missing base stat fields for {pokemon.get('name') or pokemon.get('species_display_name')}: {missing}"
        )

    return {
        "hp": int(hp),
        "attack": int(attack),
        "defense": int(defense),
        "special_attack": int(special_attack),
        "special_defense": int(special_defense),
        "speed": int(speed),
    }


def get_types_from_pokemon(pokemon):
    types = pokemon.get("types", [])

    if isinstance(types, str):
        return [
            pokemon_type.strip().lower()
            for pokemon_type in types.split(",")
            if pokemon_type.strip()
        ]

    if isinstance(types, list):
        return [str(pokemon_type).lower() for pokemon_type in types]

    type_1 = pokemon.get("type1") or pokemon.get("type_1")
    type_2 = pokemon.get("type2") or pokemon.get("type_2")

    result = []
    if type_1:
        result.append(str(type_1).lower())
    if type_2:
        result.append(str(type_2).lower())

    return result


def find_pokemon(name):
    target = normalize_name(name)

    for pokemon in POKEMON_DATA:
        possible_names = [
            pokemon.get("name", ""),
            pokemon.get("pokemonName", ""),
            pokemon.get("species_display_name", ""),
            pokemon.get("form_display_name", ""),
            pokemon.get("form_api_name", ""),
            pokemon.get("displayName", ""),
        ]

        for possible_name in possible_names:
            if normalize_name(possible_name) == target:
                return pokemon

    return None


def get_level_50_stats(name, nature="hardy"):
    pokemon = find_pokemon(name)

    if not pokemon:
        return None

    level = 50
    iv = 31

    base_stats = get_base_stats_from_pokemon(pokemon)

    final_stats = {
        "hp": calculate_hp(base_stats["hp"], level, iv),
        "attack": calculate_stat(base_stats["attack"], "attack", nature, level, iv),
        "defense": calculate_stat(base_stats["defense"], "defense", nature, level, iv),
        "special_attack": calculate_stat(
            base_stats["special_attack"],
            "special_attack",
            nature,
            level,
            iv,
        ),
        "special_defense": calculate_stat(
            base_stats["special_defense"],
            "special_defense",
            nature,
            level,
            iv,
        ),
        "speed": calculate_stat(base_stats["speed"], "speed", nature, level, iv),
    }

    return {
        "name": (
            pokemon.get("name")
            or pokemon.get("pokemonName")
            or pokemon.get("form_display_name")
            or pokemon.get("species_display_name")
        ),
        "form": pokemon.get("form_display_name") or pokemon.get("form"),
        "dexNumber": pokemon.get("dex_number") or pokemon.get("dexNumber"),
        "types": get_types_from_pokemon(pokemon),
        "level": level,
        "nature": nature,
        "ivs": {
            "hp": iv,
            "attack": iv,
            "defense": iv,
            "special_attack": iv,
            "special_defense": iv,
            "speed": iv,
        },
        "baseStats": base_stats,
        "finalStats": final_stats,
        "image": pokemon.get("image") or pokemon.get("sprite") or pokemon.get("spritePath"),
    }
