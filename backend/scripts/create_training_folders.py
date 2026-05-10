"""Create training folders for red slot Pokémon crops and type combo crops.

Run from backend:
    python scripts/create_training_folders.py
"""

import json
import re
import sys
from itertools import combinations
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from paths import SPRITE_METADATA_PATH, SPRITE_ROOT


OUTPUT_ROOT = BACKEND_DIR / "data" / "training_dataset"
SLOT_POKEMON_DIR = OUTPUT_ROOT / "slot_pokemon"
TYPE_COMBO_DIR = OUTPUT_ROOT / "type_combos"
REVIEW_DIR = OUTPUT_ROOT / "review"

POKEMON_TYPES = [
    "normal",
    "fire",
    "water",
    "electric",
    "grass",
    "ice",
    "fighting",
    "poison",
    "ground",
    "flying",
    "psychic",
    "bug",
    "rock",
    "ghost",
    "dragon",
    "dark",
    "steel",
    "fairy",
]


def slugify(value):
    value = str(value or "").strip().lower()
    value = value.replace("♀", "f").replace("♂", "m")
    value = value.replace("'", "")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "unknown"


def load_json(path):
    path = Path(path)

    if not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def extract_pokemon_names_from_metadata(metadata):
    names = set()

    if metadata is None:
        return names

    if isinstance(metadata, list):
        records = metadata
    elif isinstance(metadata, dict):
        if isinstance(metadata.get("pokemon"), list):
            records = metadata["pokemon"]
        elif isinstance(metadata.get("sprites"), list):
            records = metadata["sprites"]
        elif isinstance(metadata.get("records"), list):
            records = metadata["records"]
        else:
            records = list(metadata.values())
    else:
        records = []

    for record in records:
        if not isinstance(record, dict):
            continue

        for key in (
            "display_name",
            "form_display_name",
            "species_display_name",
            "name",
            "pokemonName",
            "species",
            "form",
        ):
            name = record.get(key)
            if name:
                names.add(str(name).strip())

    return names


def extract_pokemon_names_from_sprite_files():
    names = set()

    for sprite_dir in (SPRITE_ROOT / "normal", SPRITE_ROOT / "shiny"):
        if not sprite_dir.exists():
            continue

        for path in sprite_dir.rglob("*"):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue

            clean = path.stem
            clean = clean.replace("Menu_CP_", "")
            clean = re.sub(r"^\d+[-_]?", "", clean)
            clean = clean.replace("_", " ").replace("-", " ").strip()

            if clean:
                names.add(clean.title())

    return names


def create_slot_pokemon_folders():
    metadata = load_json(SPRITE_METADATA_PATH)
    names = extract_pokemon_names_from_metadata(metadata)

    if not names:
        names = extract_pokemon_names_from_sprite_files()

    SLOT_POKEMON_DIR.mkdir(parents=True, exist_ok=True)

    for name in sorted(names, key=slugify):
        folder = SLOT_POKEMON_DIR / slugify(name)
        folder.mkdir(parents=True, exist_ok=True)

        label_path = folder / "_label.txt"
        if not label_path.exists():
            label_path.write_text(name, encoding="utf-8")

        readme_path = folder / "_README.txt"
        if not readme_path.exists():
            readme_path.write_text(
                (
                    f"Label: {name}\n\n"
                    "Put full red opponent slot crops for this Pokémon here.\n"
                    "Example image: opponent-slot-1.jpg\n\n"
                    "The image should include the Pokémon, red card background, "
                    "type icons, and other slot UI.\n"
                    "Only use manually verified examples.\n"
                ),
                encoding="utf-8",
            )

    return len(names)


def create_type_combo_folders():
    TYPE_COMBO_DIR.mkdir(parents=True, exist_ok=True)

    combos = []

    for type_name in POKEMON_TYPES:
        combos.append((type_name,))

    for first, second in combinations(POKEMON_TYPES, 2):
        combos.append((first, second))

    for combo in combos:
        combo_name = "_".join(combo)
        folder = TYPE_COMBO_DIR / combo_name
        folder.mkdir(parents=True, exist_ok=True)

        label_path = folder / "_label.txt"
        if not label_path.exists():
            label_path.write_text(combo_name, encoding="utf-8")

        readme_path = folder / "_README.txt"
        if not readme_path.exists():
            readme_path.write_text(
                (
                    f"Label: {combo_name}\n\n"
                    "Put type cluster crops for this type combo here.\n"
                    "Example image: opponent-type-cluster-1.jpg\n\n"
                    "The crop should show the full type icon area, not the full Pokémon slot.\n"
                    "Only use manually verified examples.\n"
                ),
                encoding="utf-8",
            )

    return len(combos)


def create_review_folders():
    folders = [
        REVIEW_DIR / "needs_review",
        REVIEW_DIR / "bad_crops",
        REVIEW_DIR / "unsorted_slot_pokemon",
        REVIEW_DIR / "unsorted_type_combos",
    ]

    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)

    return len(folders)


def create_root_readme():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    (OUTPUT_ROOT / "_README.txt").write_text(
        (
            "ChampionsInsight Training Dataset\n"
            "=================================\n\n"
            "Use slot_pokemon/ for full red slot crops labeled by Pokémon.\n"
            "Use type_combos/ for type cluster crops labeled by type combo.\n\n"
            "Do not train on review/needs_review, review/bad_crops, or unsorted folders.\n"
        ),
        encoding="utf-8",
    )


def main():
    create_root_readme()

    pokemon_count = create_slot_pokemon_folders()
    combo_count = create_type_combo_folders()
    review_count = create_review_folders()

    print("Training folders created.")
    print(f"Root: {OUTPUT_ROOT}")
    print(f"Slot Pokémon folders: {pokemon_count}")
    print(f"Type combo folders: {combo_count}")
    print(f"Review/helper folders: {review_count}")


if __name__ == "__main__":
    main()