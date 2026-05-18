"""Repair training folder labels for synthetic Pokemon and type-combo data.

Pokemon synthetic folders are generated from Menu_CP sprite filenames. This
script maps those filenames through pokemon_lookup_by_dex.json and moves images
into normalized Pokemon form labels such as:

    Menu_CP_0003 -> venusaur
    Menu_CP_0026-Alola -> raichu_alola

Type-combo folders are checked for invalid type words and fixed when a safe
spelling correction is known.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from urllib.parse import unquote


BACKEND_DIR = Path(__file__).resolve().parents[2]
POKEMON_LOOKUP_BY_DEX = BACKEND_DIR / "data" / "pokemon" / "pokemon_lookup_by_dex.json"

DEFAULT_POKEMON_DIRS = [
    BACKEND_DIR / "data" / "training_dataset" / "slot_pokemon_synthetic",
    BACKEND_DIR / "data" / "training_dataset" / "slot_pokemon_synthetic_real_like",
    BACKEND_DIR / "data" / "training_dataset" / "slot_pokemon_synthetic_real_like_visible",
]

DEFAULT_TYPE_DIRS = [
    BACKEND_DIR / "data" / "training_dataset" / "type_combos",
    BACKEND_DIR / "data" / "training_dataset" / "type_combos_synthetic_visible",
    BACKEND_DIR / "data" / "training_dataset" / "review" / "type_combos_synthetic_preview",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VALID_TYPES = {
    "bug",
    "dark",
    "dragon",
    "electric",
    "fairy",
    "fighting",
    "fire",
    "flying",
    "ghost",
    "grass",
    "ground",
    "ice",
    "normal",
    "poison",
    "psychic",
    "rock",
    "steel",
    "water",
}

TYPE_WORD_FIXES = {
    "dargon": "dragon",
}


def is_mega_label(value: str) -> bool:
    normalized = normalize_label(value)
    return "mega" in normalized.split("_")


def normalize_label(value: str) -> str:
    value = str(value or "").strip().lower()
    value = value.replace("â™€", "_f")
    value = value.replace("â™‚", "_m")
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
    return value.strip("_")


def parse_menu_cp_from_name(name: str) -> tuple[int | None, str | None]:
    stem = unquote(Path(name).stem)
    stem = re.sub(r"_synthetic_\d+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"_shiny$", "", stem, flags=re.IGNORECASE)
    stem = stem.replace("-shiny", "")

    match = re.match(r"menu_cp_(\d+)(?:_([a-z0-9_]+))?$", normalize_label(stem))
    if not match:
        return None, None

    form_suffix = match.group(2)
    return int(match.group(1)), normalize_label(form_suffix) if form_suffix else None


def load_pokemon_mapping() -> tuple[dict[tuple[int, str | None], str], dict[int, str]]:
    if not POKEMON_LOOKUP_BY_DEX.exists():
        raise FileNotFoundError(f"Missing lookup: {POKEMON_LOOKUP_BY_DEX}")

    lookup = json.loads(POKEMON_LOOKUP_BY_DEX.read_text(encoding="utf-8"))
    mapping: dict[tuple[int, str | None], str] = {}
    base_by_dex: dict[int, str] = {}

    for dex_key, forms in lookup.items():
        dex_number = int(dex_key)
        default_label = None

        for form in forms:
            form_api_name = normalize_label(form.get("form_api_name", ""))
            species_api_name = normalize_label(
                form.get("species_api_name") or form.get("species_display_name") or ""
            )
            form_display = normalize_label(form.get("form_display_name", ""))

            if not form_api_name:
                continue
            if is_mega_label(form_api_name) or is_mega_label(form_display):
                continue

            if form.get("is_default") or form_api_name == species_api_name:
                default_label = form_api_name
                mapping[(dex_number, None)] = form_api_name
                base_by_dex[dex_number] = form_api_name

            suffixes = set()
            for candidate in [form_api_name, form_display]:
                candidate = normalize_label(candidate)
                if not candidate:
                    continue
                suffixes.add(candidate)
                if species_api_name and candidate.startswith(species_api_name + "_"):
                    suffixes.add(candidate[len(species_api_name) + 1 :])
                parts = candidate.split("_")
                if len(parts) > 1:
                    suffixes.add(parts[-1])

            for suffix in suffixes:
                mapping[(dex_number, suffix)] = form_api_name

        if default_label:
            mapping.setdefault((dex_number, None), default_label)
            base_by_dex.setdefault(dex_number, default_label)

    return mapping, base_by_dex


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
    folders = sorted(
        [path for path in root.rglob("*") if path.is_dir()],
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for folder in folders:
        try:
            folder.rmdir()
            removed += 1
        except OSError:
            pass
    return removed


def repair_pokemon_folder(root: Path, dry_run: bool = False) -> dict[str, object]:
    mapping, base_by_dex = load_pokemon_mapping()
    image_files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    moved = 0
    already_correct = 0
    skipped: list[tuple[str, str]] = []

    for file_path in image_files:
        dex_number, form_suffix = parse_menu_cp_from_name(file_path.name)
        if dex_number is None:
            skipped.append((str(file_path), "no Menu_CP id in filename"))
            continue

        label = mapping.get((dex_number, form_suffix))
        if not label and form_suffix and dex_number in base_by_dex:
            label = normalize_label(f"{base_by_dex[dex_number]}_{form_suffix}")
        if not label:
            label = mapping.get((dex_number, None)) or base_by_dex.get(dex_number)
        if not label:
            skipped.append((str(file_path), f"no mapping for dex={dex_number}, form={form_suffix}"))
            continue

        target_dir = root / label
        if file_path.parent == target_dir:
            already_correct += 1
            continue

        moved += 1
        if not dry_run:
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = unique_target_path(target_dir, file_path.name)
            shutil.move(str(file_path), str(target_path))

    removed_empty_dirs = 0 if dry_run else cleanup_empty_dirs(root)

    return {
        "root": str(root),
        "images": len(image_files),
        "moved": moved,
        "alreadyCorrect": already_correct,
        "skipped": skipped,
        "removedEmptyDirs": removed_empty_dirs,
    }


def fixed_type_folder_name(name: str) -> tuple[str | None, list[str]]:
    parts = [part for part in normalize_label(name).split("_") if part]
    fixed_parts = [TYPE_WORD_FIXES.get(part, part) for part in parts]
    invalid = [part for part in fixed_parts if part not in VALID_TYPES]
    if invalid:
        return None, invalid
    return "_".join(fixed_parts), []


def merge_folder(source_dir: Path, target_dir: Path, dry_run: bool = False) -> int:
    moved = 0
    if source_dir == target_dir:
        return moved

    for child in source_dir.iterdir():
        moved += 1
        if dry_run:
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / child.name
        if child.is_file():
            target_path = unique_target_path(target_dir, child.name)
            shutil.move(str(child), str(target_path))
        elif child.is_dir():
            merge_folder(child, target_path, dry_run=False)

    if not dry_run:
        try:
            source_dir.rmdir()
        except OSError:
            pass

    return moved


def repair_type_folder(root: Path, dry_run: bool = False) -> dict[str, object]:
    renamed: list[tuple[str, str, int]] = []
    invalid: list[tuple[str, list[str]]] = []

    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        fixed_name, invalid_parts = fixed_type_folder_name(folder.name)
        if invalid_parts:
            invalid.append((folder.name, invalid_parts))
            continue
        if fixed_name and fixed_name != folder.name:
            target_dir = root / fixed_name
            moved_items = merge_folder(folder, target_dir, dry_run=dry_run)
            renamed.append((folder.name, fixed_name, moved_items))

    return {
        "root": str(root),
        "renamed": renamed,
        "invalid": invalid,
    }


def existing_dirs(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pokemon-dir", action="append", type=Path, default=[])
    parser.add_argument("--type-dir", action="append", type=Path, default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    pokemon_dirs = args.pokemon_dir or existing_dirs(DEFAULT_POKEMON_DIRS)
    type_dirs = args.type_dir or existing_dirs(DEFAULT_TYPE_DIRS)

    print(f"Dry run: {args.dry_run}")
    print()

    for root in pokemon_dirs:
        result = repair_pokemon_folder(root, dry_run=args.dry_run)
        print(f"Pokemon folder: {result['root']}")
        print(f"  Images scanned: {result['images']}")
        print(f"  Images moved: {result['moved']}")
        print(f"  Already correct: {result['alreadyCorrect']}")
        print(f"  Skipped: {len(result['skipped'])}")
        print(f"  Empty folders removed: {result['removedEmptyDirs']}")
        for path, reason in result["skipped"][:20]:
            print(f"    - {reason}: {path}")
        print()

    for root in type_dirs:
        result = repair_type_folder(root, dry_run=args.dry_run)
        print(f"Type folder: {result['root']}")
        print(f"  Renamed/merged: {len(result['renamed'])}")
        for old_name, new_name, moved_items in result["renamed"][:30]:
            print(f"    - {old_name} -> {new_name} ({moved_items} items)")
        print(f"  Invalid names remaining: {len(result['invalid'])}")
        for name, invalid_parts in result["invalid"][:30]:
            print(f"    - {name}: {', '.join(invalid_parts)}")
        print()


if __name__ == "__main__":
    main()

