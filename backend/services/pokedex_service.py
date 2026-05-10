import json
import re
from functools import lru_cache
from pathlib import Path

from paths import POKEMON_DATA_DIR
from services.tournament_usage_service import get_pokemon_usage_summary


DATA_DIR = POKEMON_DATA_DIR

CHAMPIONS_SPRITES_DIR = DATA_DIR / "champions_sprites" / "normal"
POKEMON_LOOKUP_BY_DEX_PATH = DATA_DIR / "pokemon_lookup_by_dex.json"
LEARNSET_PATH = DATA_DIR / "learnset.json"


STAT_KEY_MAP = {
    "hp": "hp",
    "attack": "attack",
    "defense": "defense",
    "special_attack": "specialAttack",
    "special-attack": "specialAttack",
    "specialAttack": "specialAttack",
    "sp_atk": "specialAttack",
    "special_defense": "specialDefense",
    "special-defense": "specialDefense",
    "specialDefense": "specialDefense",
    "sp_def": "specialDefense",
    "speed": "speed",
}


def load_json_file(path: Path):
    if not path.exists():
        return None

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_stats(stats):
    normalized = {
        "hp": 0,
        "attack": 0,
        "defense": 0,
        "specialAttack": 0,
        "specialDefense": 0,
        "speed": 0,
    }

    if not isinstance(stats, dict):
        return normalized

    for key, value in stats.items():
        mapped_key = STAT_KEY_MAP.get(key)

        if mapped_key:
            normalized[mapped_key] = int(value or 0)

    return normalized


def extract_dex_number_from_sprite(filename: str):
    match = re.search(r"Menu_CP_(\d+)", filename)

    if not match:
        return 0

    return int(match.group(1))


def extract_form_suffix_from_sprite(filename: str):
    """
    Examples:
    Menu_CP_0003.png -> ""
    Menu_CP_0003-Mega.png -> "mega"
    Menu_CP_0006-Mega_X.png -> "mega x"
    Menu_CP_0026-Alola.png -> "alola"
    """

    stem = Path(filename).stem
    match = re.match(r"Menu_CP_\d+(?:-(.+))?$", stem)

    if not match or not match.group(1):
        return ""

    return match.group(1).replace("_", " ").replace("-", " ").strip().lower()


def build_sprite_url_from_filename(filename: str):
    return f"/api/pokemon/sprite/normal/{filename}"


@lru_cache(maxsize=1)
def get_champions_sprite_files():
    if not CHAMPIONS_SPRITES_DIR.exists():
        raise FileNotFoundError(f"Missing Champions sprites folder: {CHAMPIONS_SPRITES_DIR}")

    image_extensions = {".png", ".jpg", ".jpeg", ".webp"}

    return tuple(
        sorted(
            file
            for file in CHAMPIONS_SPRITES_DIR.rglob("*")
            if file.is_file() and file.suffix.lower() in image_extensions
        )
    )


@lru_cache(maxsize=1)
def get_lookup_by_dex():
    data = load_json_file(POKEMON_LOOKUP_BY_DEX_PATH)

    if not isinstance(data, dict):
        return {}

    return data


@lru_cache(maxsize=1)
def get_learnset_data():
    data = load_json_file(LEARNSET_PATH)

    if not isinstance(data, dict):
        return {}

    return data


def normalize_api_name(value: str):
    return str(value or "").strip().lower().replace("_", "-")


def choose_form_for_sprite(dex_entries: list, sprite_filename: str):
    if not dex_entries:
        return None

    form_suffix = extract_form_suffix_from_sprite(sprite_filename)

    if not form_suffix:
        for entry in dex_entries:
            if entry.get("is_default") is True:
                return entry

        return dex_entries[0]

    form_suffix = form_suffix.lower()

    for entry in dex_entries:
        form_display = str(entry.get("form_display_name", "")).lower()
        form_api = str(entry.get("form_api_name", "")).lower().replace("-", " ")

        if form_suffix in form_display or form_suffix in form_api:
            return entry

    return dex_entries[0]


def get_moves_for_form(form_entry):
    learnset_data = get_learnset_data()

    form_api_name = normalize_api_name(form_entry.get("form_api_name"))
    species_name = normalize_api_name(form_entry.get("species_display_name"))

    possible_keys = [
        form_api_name,
        species_name,
        form_api_name.replace("-", ""),
        species_name.replace("-", ""),
    ]

    learnset_entry = None

    for key in possible_keys:
        if key in learnset_data:
            learnset_entry = learnset_data[key]
            break

    if not learnset_entry:
        return []

    raw_learnset = learnset_entry.get("learnset", {})

    if not isinstance(raw_learnset, dict):
        return []

    return sorted(raw_learnset.keys())


def normalize_abilities(abilities):
    if not isinstance(abilities, list):
        return []

    normalized = []

    for ability in abilities:
        if isinstance(ability, dict):
            normalized.append(
                {
                    "name": ability.get("name", ""),
                    "displayName": ability.get("display_name", ability.get("name", "")),
                    "isHidden": bool(ability.get("is_hidden", False)),
                }
            )
        else:
            normalized.append(
                {
                    "name": str(ability),
                    "displayName": str(ability),
                    "isHidden": False,
                }
            )

    return normalized


def normalize_pokemon_from_lookup(form_entry, sprite_filename, dex_number):
    name = (
        form_entry.get("form_display_name")
        or form_entry.get("species_display_name")
        or f"Pokemon #{dex_number}"
    )

    return {
        "id": dex_number,
        "name": name,
        "speciesName": form_entry.get("species_display_name", name),
        "formApiName": form_entry.get("form_api_name", ""),
        "types": form_entry.get("types", []),
        "abilities": normalize_abilities(form_entry.get("abilities", [])),
        # Keep grid fast. Moves are added only in get_pokedex_entry_detail_by_name().
        "moves": [],
        "baseStats": normalize_stats(form_entry.get("base_stats", {})),
        "spriteUrl": build_sprite_url_from_filename(sprite_filename),
        "spriteFilename": sprite_filename,
        "isMega": "mega" in name.lower() or "mega" in sprite_filename.lower(),
    }


@lru_cache(maxsize=1)
def get_all_pokedex_entries():
    lookup_by_dex = get_lookup_by_dex()
    sprite_files = get_champions_sprite_files()

    entries = []

    for sprite_file in sprite_files:
        dex_number = extract_dex_number_from_sprite(sprite_file.name)

        if dex_number <= 0:
            continue

        dex_key = str(dex_number).zfill(4)
        dex_entries = lookup_by_dex.get(dex_key, [])

        form_entry = choose_form_for_sprite(dex_entries, sprite_file.name)

        if form_entry:
            entries.append(
                normalize_pokemon_from_lookup(
                    form_entry=form_entry,
                    sprite_filename=sprite_file.name,
                    dex_number=dex_number,
                )
            )
        else:
            entries.append(
                {
                    "id": dex_number,
                    "name": f"Pokemon #{dex_number}",
                    "speciesName": f"Pokemon #{dex_number}",
                    "formApiName": "",
                    "types": [],
                    "abilities": [],
                    "moves": [],
                    "baseStats": normalize_stats({}),
                    "spriteUrl": build_sprite_url_from_filename(sprite_file.name),
                    "spriteFilename": sprite_file.name,
                    "isMega": "mega" in sprite_file.name.lower(),
                }
            )

    return tuple(sorted(entries, key=lambda pokemon: (pokemon["id"], pokemon["name"])))


def to_pokedex_list_entry(pokemon):
    return {
        "id": pokemon["id"],
        "name": pokemon["name"],
        "speciesName": pokemon.get("speciesName", pokemon["name"]),
        "formApiName": pokemon.get("formApiName", ""),
        "types": pokemon.get("types", []),
        "abilities": pokemon.get("abilities", []),
        "baseStats": pokemon.get("baseStats", {}),
        "spriteUrl": pokemon.get("spriteUrl", ""),
        "spriteFilename": pokemon.get("spriteFilename", ""),
        "isMega": pokemon.get("isMega", False),
    }


def get_pokedex_grid_entries():
    return [
        to_pokedex_list_entry(pokemon)
        for pokemon in get_all_pokedex_entries()
    ]


def get_pokedex_entry_by_name(name: str):
    if not name:
        return None

    target = name.strip().lower()

    for pokemon in get_all_pokedex_entries():
        possible_names = {
            pokemon["name"].lower(),
            pokemon.get("speciesName", "").lower(),
            pokemon.get("formApiName", "").lower(),
            pokemon.get("formApiName", "").replace("-", " ").lower(),
        }

        if target in possible_names:
            return dict(pokemon)

    return None


def get_form_entry_for_pokemon(pokemon):
    lookup_by_dex = get_lookup_by_dex()

    dex_key = str(pokemon["id"]).zfill(4)
    dex_entries = lookup_by_dex.get(dex_key, [])

    for entry in dex_entries:
        if entry.get("form_api_name") == pokemon.get("formApiName"):
            return entry

    return None


def get_pokedex_entry_detail_by_name(name: str):
    pokemon = get_pokedex_entry_by_name(name)

    if pokemon is None:
        return None

    pokemon_detail = dict(pokemon)

    form_entry = get_form_entry_for_pokemon(pokemon_detail)

    if form_entry:
        pokemon_detail["moves"] = get_moves_for_form(form_entry)
    else:
        pokemon_detail["moves"] = []

    usage_name = pokemon_detail.get("speciesName") or pokemon_detail["name"]
    pokemon_detail["usage"] = get_pokemon_usage_summary(usage_name)

    return pokemon_detail
