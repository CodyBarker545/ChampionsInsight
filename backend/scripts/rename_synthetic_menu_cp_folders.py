"""
Repair synthetic Pokémon folder names.

This scans every image in slot_pokemon_synthetic, reads the Menu_CP ID from
the filename, maps it using pokemon_battle_data.csv, and moves the image into
the correct Pokémon-name folder.

Run from backend:
python scripts/repair_synthetic_folder_names.py
"""

import csv
import re
import shutil
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]

SYNTHETIC_DIR = BACKEND_DIR / "data" / "training_dataset" / "slot_pokemon_synthetic"
POKEMON_CSV = BACKEND_DIR / "data" / "pokemon" / "pokemon_battle_data.csv"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def normalize_label(value: str) -> str:
    value = str(value or "").strip().lower()

    value = value.replace("♀", "_f")
    value = value.replace("♂", "_m")
    value = value.replace("'", "")
    value = value.replace(".", "")
    value = value.replace(":", "")
    value = value.replace("(", "")
    value = value.replace(")", "")
    value = value.replace("[", "")
    value = value.replace("]", "")
    value = value.replace("/", "_")
    value = value.replace("\\", "_")
    value = value.replace("-", "_")
    value = value.replace(" ", "_")

    value = re.sub(r"[^a-z0-9_]+", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("_")

    return value


def parse_menu_cp_from_name(name: str) -> tuple[int | None, str | None]:
    """
    Examples:
    Menu_CP_0003_synthetic_0000.jpg              -> (3, None)
    Menu_CP_0003_shiny_synthetic_0000.jpg        -> (3, None)
    Menu_CP_0026-Alola_synthetic_0000.jpg        -> (26, "alola")
    Menu_CP_0026-Alola_shiny_synthetic_0000.jpg  -> (26, "alola")
    Menu_CP_0479-Wash_synthetic_0000.jpg         -> (479, "wash")
    Menu_CP_0681-Blade_shiny_synthetic_0000.jpg  -> (681, "blade")
    """

    stem = Path(name).stem

    stem = re.sub(r"_synthetic_\d+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_shiny$", "", stem, flags=re.IGNORECASE)
    stem = stem.replace("-shiny", "")

    match = re.match(r"Menu_CP_(\d+)(?:-([A-Za-z0-9_]+))?$", stem)

    if not match:
        return None, None

    dex_number = int(match.group(1))
    form_suffix = match.group(2)

    if form_suffix:
        form_suffix = normalize_label(form_suffix)

    return dex_number, form_suffix


def remove_species_prefix(label: str, species: str) -> str:
    label = normalize_label(label)
    species = normalize_label(species)

    if label == species:
        return ""

    prefix = species + "_"

    if label.startswith(prefix):
        return label[len(prefix):]

    return label


def load_mapping() -> tuple[dict[tuple[int, str | None], str], dict[int, str]]:
    """
    Returns:
    mapping[(dex, None)] = base species label
    mapping[(dex, form_suffix)] = form label
    species_by_dex[dex] = species label
    """

    if not POKEMON_CSV.exists():
        raise FileNotFoundError(f"Missing CSV: {POKEMON_CSV}")

    mapping: dict[tuple[int, str | None], str] = {}
    species_by_dex: dict[int, str] = {}

    with POKEMON_CSV.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            dex_text = row.get("dex_number_int") or row.get("dex_number")

            if not dex_text:
                continue

            try:
                dex_number = int(str(dex_text).strip())
            except ValueError:
                continue

            species = normalize_label(row.get("species_display_name") or "")
            form_display = normalize_label(row.get("form_display_name") or "")
            form_identifier = normalize_label(row.get("form_identifier") or "")

            if not species:
                continue

            if "mega" in species or "mega" in form_display or "mega" in form_identifier:
                continue

            # Always save base species fallback.
            species_by_dex[dex_number] = species
            mapping.setdefault((dex_number, None), species)

            # If this row is the base form, keep base mapping.
            if (
                not form_display
                or form_display == species
                or not form_identifier
                or form_identifier == species
            ):
                mapping[(dex_number, None)] = species
                continue

            # Form label should usually be form_identifier, e.g. rotom_wash.
            if form_identifier and form_identifier != species:
                form_label = form_identifier
            else:
                form_label = f"{species}_{form_display}"

            form_label = normalize_label(form_label)

            if "mega" in form_label:
                continue

            # Build possible suffixes that might appear after Menu_CP_####-
            suffixes = set()

            display_suffix = remove_species_prefix(form_display, species)
            identifier_suffix = remove_species_prefix(form_identifier, species)
            label_suffix = remove_species_prefix(form_label, species)

            for suffix in [display_suffix, identifier_suffix, label_suffix]:
                if suffix:
                    suffixes.add(suffix)

            # Also add final piece aliases.
            # Example: paldea_aqua -> aqua
            for suffix in list(suffixes):
                parts = suffix.split("_")
                if len(parts) > 1:
                    suffixes.add(parts[-1])

            for suffix in suffixes:
                mapping[(dex_number, suffix)] = form_label

    return mapping, species_by_dex


def unique_target_path(target_dir: Path, filename: str) -> Path:
    target_path = target_dir / filename

    if not target_path.exists():
        return target_path

    stem = target_path.stem
    suffix = target_path.suffix
    counter = 1

    while True:
        candidate = target_dir / f"{stem}__dup{counter}{suffix}"

        if not candidate.exists():
            return candidate

        counter += 1


def cleanup_empty_dirs(root: Path) -> int:
    removed = 0

    folders = [p for p in root.rglob("*") if p.is_dir()]
    folders.sort(key=lambda p: len(p.parts), reverse=True)

    for folder in folders:
        try:
            folder.rmdir()
            removed += 1
        except OSError:
            pass

    return removed


def main() -> None:
    if not SYNTHETIC_DIR.exists():
        raise FileNotFoundError(f"Missing synthetic folder: {SYNTHETIC_DIR}")

    mapping, species_by_dex = load_mapping()

    print(f"Loaded mapping entries: {len(mapping)}")
    print(f"Loaded base species entries: {len(species_by_dex)}")
    print(f"Synthetic folder: {SYNTHETIC_DIR}")
    print()

    image_files = [
        path for path in SYNTHETIC_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    moved = 0
    already_correct = 0
    skipped = []

    for file_path in image_files:
        dex_number, form_suffix = parse_menu_cp_from_name(file_path.name)

        if dex_number is None:
            skipped.append((str(file_path), "no Menu_CP id in filename"))
            continue

        label = mapping.get((dex_number, form_suffix))

        if not label and form_suffix:
            base_species = species_by_dex.get(dex_number)

            if base_species:
                # Safe fallback for missing forms:
                # Menu_CP_0026-Alola -> raichu_alola
                label = normalize_label(f"{base_species}_{form_suffix}")

        if not label:
            label = mapping.get((dex_number, None))

        if not label:
            label = species_by_dex.get(dex_number)

        if not label:
            skipped.append((str(file_path), f"no mapping for dex={dex_number}, form={form_suffix}"))
            continue

        target_dir = SYNTHETIC_DIR / label
        target_dir.mkdir(parents=True, exist_ok=True)

        if file_path.parent == target_dir:
            already_correct += 1
            continue

        target_path = unique_target_path(target_dir, file_path.name)
        shutil.move(str(file_path), str(target_path))
        moved += 1

    removed_empty_dirs = cleanup_empty_dirs(SYNTHETIC_DIR)

    print("Done.")
    print(f"Images scanned: {len(image_files)}")
    print(f"Images moved: {moved}")
    print(f"Already correct: {already_correct}")
    print(f"Skipped images: {len(skipped)}")
    print(f"Empty folders removed: {removed_empty_dirs}")

    if skipped:
        print()
        print("First 50 skipped:")
        for path, reason in skipped[:50]:
            print(f"  - {reason}: {path}")


if __name__ == "__main__":
    main()