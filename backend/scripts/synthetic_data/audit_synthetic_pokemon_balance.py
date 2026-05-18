"""Audit normal/shiny balance for the synthetic Pokemon slot dataset."""

from __future__ import annotations

import argparse
from pathlib import Path


def count_variants(class_dir: Path) -> tuple[int, int]:
    normal = 0
    shiny = 0

    for image_path in class_dir.glob("*_synthetic_*.jpg"):
        if "_shiny_" in image_path.name.lower():
            shiny += 1
        else:
            normal += 1

    return normal, shiny


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="backend/data/training_dataset/slot_pokemon_synthetic",
    )
    parser.add_argument("--target-per-variant", type=int, default=1000)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    target = args.target_per_variant
    rows = []

    for class_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        normal, shiny = count_variants(class_dir)
        rows.append((class_dir.name, normal, shiny))

    complete = [row for row in rows if row[1] == target and row[2] == target]
    incomplete = [row for row in rows if row not in complete]

    print(f"Dataset: {dataset_dir}")
    print(f"Target: {target} normal + {target} shiny per Pokemon")
    print(f"Classes: {len(rows)}")
    print(f"Complete: {len(complete)}")
    print(f"Incomplete: {len(incomplete)}")

    if incomplete:
        print()
        print("Needs work:")
        for label, normal, shiny in incomplete:
            print(f"{label}: normal={normal}, shiny={shiny}")


if __name__ == "__main__":
    main()
