"""Generate a balanced, incremental training set for newly synced Champions sprites."""
import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))
from scripts.synthetic_data.generate_synthetic_pokemon_cards import generate_dataset, parse_label


def prepare(manifest, output, target_per_class=250, workers=4, trim_excess=False):
    if target_per_class < 2 or target_per_class % 2:
        raise ValueError("Target per class must be a positive even number for normal/shiny balance")
    records = json.loads(manifest.read_text(encoding="utf-8"))["added_sprites"]
    sprites = [(record, parse_label(Path(record["local_filename"]))) for record in records]
    sprites = [(record, label) for record, label in sprites if label is not None]
    labels = {label for _, label in sprites}
    for label in labels:
        variants = [record["is_shiny"] for record, name in sprites if name == label]
        if sorted(variants) != [False, True]:
            raise ValueError(f"Expected one normal and one shiny source for {label}")
    target = target_per_class // 2
    if trim_excess:
        for record, label in sprites:
            folder = (output / label).resolve()
            if not folder.is_relative_to(output.resolve()):
                raise ValueError(f"Unsafe output folder: {folder}")
            # Only files made from this update's new sources can be removed.
            pattern = Path(record["local_filename"]).stem + "_synthetic_*.jpg"
            for path in sorted(folder.glob(pattern))[target:]:
                path.unlink()
    generate_dataset(BACKEND / "data/pokemon/champions_sprites", output,
                     None, target, 10, 0, None, manifest=manifest, workers=workers)
    counts = {}
    for record, label in sprites:
        pattern = Path(record["local_filename"]).stem + "_synthetic_*.jpg"
        count = len(list((output / label).glob(pattern)))
        counts.setdefault(label, {})["shiny" if record["is_shiny"] else "normal"] = count
        if count != target:
            raise RuntimeError(f"{record['local_filename']}: expected {target} images, found {count}")
    report = dict(classes=len(labels), target_per_class=target_per_class,
                  total_images=sum(sum(x.values()) for x in counts.values()), class_counts=counts)
    path = BACKEND / "data/training_dataset/champions_training_update_summary.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "class_counts"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=BACKEND / "data/pokemon/champions_catalog_update.json")
    parser.add_argument("--output", type=Path, default=BACKEND / "data/training_dataset/slot_pokemon_synthetic")
    parser.add_argument("--target-per-class", type=int, default=250)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--trim-excess", action="store_true", help="Trim only crops generated from this update's new sprites")
    args = parser.parse_args()
    prepare(args.manifest, args.output, args.target_per_class, args.workers, args.trim_excess)
