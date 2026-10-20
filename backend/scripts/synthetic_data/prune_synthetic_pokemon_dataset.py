"""Trim synthetic Pokemon slot images down to a balanced per-variant cap.

This script prunes each class folder in the synthetic slot dataset to a fixed
number of normal images and shiny images. It removes low-quality images first
based on how much usable Champion spirit content is detected in the crop. If a
class still has more images than the target after weak images are exhausted, it
removes the remaining excess at random.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services import cv_service


@dataclass
class ImageScore:
    path: Path
    score: float
    crop_quality: float
    source: str
    width_ratio: float
    height_ratio: float
    area_ratio: float
    weak_reason: str


def is_synthetic_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"} and "_synthetic_" in path.name.lower()


def is_shiny_image(path: Path) -> bool:
    return "_shiny_" in path.name.lower()


def score_image(image_path: Path, min_spirit_score: float) -> ImageScore:
    try:
        image = cv_service.read_cv_image(image_path)
        height, width = image.shape[:2]
        sprite_object = cv_service.detect_pokemon_sprite_object(image)

        if not sprite_object or cv_service.is_empty_image(sprite_object.get("image")):
            return ImageScore(
                path=image_path,
                score=0.0,
                crop_quality=0.0,
                source="no_sprite_detected",
                width_ratio=0.0,
                height_ratio=0.0,
                area_ratio=0.0,
                weak_reason="no_sprite_detected",
            )

        box = sprite_object.get("box") or {}
        width_ratio = float(box.get("width", 0) or 0) / max(1, width)
        height_ratio = float(box.get("height", 0) or 0) / max(1, height)
        area_ratio = float((box.get("width", 0) or 0) * (box.get("height", 0) or 0)) / max(
            1,
            width * height,
        )
        crop_quality = float(sprite_object.get("cropQuality", 0.0) or 0.0)

        weak_reason = ""
        if crop_quality < min_spirit_score:
            weak_reason = "low_spirit_score"

        return ImageScore(
            path=image_path,
            score=round(crop_quality, 4),
            crop_quality=round(crop_quality, 4),
            source=str(sprite_object.get("source") or "unknown"),
            width_ratio=round(width_ratio, 4),
            height_ratio=round(height_ratio, 4),
            area_ratio=round(area_ratio, 4),
            weak_reason=weak_reason,
        )
    except Exception as exc:
        return ImageScore(
            path=image_path,
            score=0.0,
            crop_quality=0.0,
            source="error",
            width_ratio=0.0,
            height_ratio=0.0,
            area_ratio=0.0,
            weak_reason=f"error:{type(exc).__name__}",
        )


def choose_deletions(
    images: list[Path],
    *,
    label: str,
    variant: str,
    target: int,
    min_spirit_score: float,
    seed: int,
) -> tuple[list[ImageScore], list[ImageScore], list[ImageScore]]:
    scored_images = [score_image(path, min_spirit_score=min_spirit_score) for path in sorted(images)]
    excess = max(0, len(scored_images) - target)
    if excess == 0:
        return scored_images, [], []

    rng = random.Random(f"{seed}:{label}:{variant}")
    weak_images = [item for item in scored_images if item.weak_reason]
    weak_images.sort(key=lambda item: (item.score, item.path.name.lower()))

    deleted: list[ImageScore] = weak_images[:excess]

    if len(deleted) < excess:
        deleted_paths = {item.path for item in deleted}
        strong_images = [item for item in scored_images if item.path not in deleted_paths]
        deleted.extend(rng.sample(strong_images, excess - len(deleted)))

    deleted_paths = {item.path for item in deleted}
    kept = [item for item in scored_images if item.path not in deleted_paths]
    deleted.sort(key=lambda item: (item.score, item.path.name.lower()))
    kept.sort(key=lambda item: item.path.name.lower())
    return scored_images, kept, deleted


def ensure_within_dataset(dataset_dir: Path, image_path: Path) -> None:
    dataset_root = dataset_dir.resolve()
    resolved = image_path.resolve()
    if not resolved.is_relative_to(dataset_root):
        raise ValueError(f"Refusing to delete outside dataset: {resolved}")


def write_report(report_path: Path, report_rows: list[dict]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report_rows, indent=2), encoding="utf-8")


def prune_dataset(
    dataset_dir: Path,
    *,
    target_per_variant: int,
    min_spirit_score: float,
    seed: int,
    dry_run: bool,
    report_path: Path | None,
) -> None:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_dir}")

    class_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    report_rows: list[dict] = []
    deleted_total = 0

    print(f"Dataset: {dataset_dir}")
    print(f"Target per variant: {target_per_variant}")
    print(f"Minimum spirit score before pruning priority: {min_spirit_score}")
    print(f"Dry run: {dry_run}")
    print(f"Classes: {len(class_dirs)}")
    print()

    for class_dir in class_dirs:
        synthetic_images = [path for path in class_dir.iterdir() if is_synthetic_image(path)]
        normal_images = [path for path in synthetic_images if not is_shiny_image(path)]
        shiny_images = [path for path in synthetic_images if is_shiny_image(path)]

        variant_results = []
        class_deleted = 0

        for variant_name, variant_images in (("normal", normal_images), ("shiny", shiny_images)):
            scored, kept, deleted = choose_deletions(
                variant_images,
                label=class_dir.name,
                variant=variant_name,
                target=target_per_variant,
                min_spirit_score=min_spirit_score,
                seed=seed,
            )
            weak_count = sum(1 for item in scored if item.weak_reason)

            if not dry_run:
                for item in deleted:
                    ensure_within_dataset(dataset_dir, item.path)
                    item.path.unlink(missing_ok=True)

            class_deleted += len(deleted)
            deleted_total += len(deleted)
            variant_results.append(
                {
                    "variant": variant_name,
                    "before": len(scored),
                    "after": len(kept),
                    "deleted": len(deleted),
                    "weak_candidates": weak_count,
                    "deleted_weak_first": sum(1 for item in deleted if item.weak_reason),
                    "deleted_random": sum(1 for item in deleted if not item.weak_reason),
                    "deleted_files": [
                        {
                            **asdict(item),
                            "path": str(item.path),
                        }
                        for item in deleted
                    ],
                }
            )

        if class_deleted:
            print(
                f"{class_dir.name}: "
                + ", ".join(
                    (
                        f"{result['variant']} {result['before']}->{result['after']}"
                        f" (deleted {result['deleted']}, weak {result['deleted_weak_first']}, random {result['deleted_random']})"
                    )
                    for result in variant_results
                )
            )

        report_rows.append(
            {
                "class": class_dir.name,
                "variants": variant_results,
            }
        )

    print()
    print(f"Total deleted: {deleted_total}")
    if report_path:
        write_report(report_path, report_rows)
        print(f"Report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="backend/data/training_dataset/slot_pokemon_synthetic",
        help="Synthetic slot dataset root.",
    )
    parser.add_argument(
        "--target-per-variant",
        type=int,
        default=500,
        help="Keep at most this many normal and this many shiny images per class.",
    )
    parser.add_argument(
        "--min-spirit-score",
        type=float,
        default=0.38,
        help="Images below this score are pruned before random elimination.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Seed for deterministic random deletion among otherwise acceptable images.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview deletions without removing files.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON report path.",
    )
    args = parser.parse_args()

    prune_dataset(
        dataset_dir=Path(args.dataset),
        target_per_variant=int(args.target_per_variant),
        min_spirit_score=float(args.min_spirit_score),
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
        report_path=Path(args.report) if args.report else None,
    )


if __name__ == "__main__":
    main()
