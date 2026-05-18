"""Cluster unsorted Champions Insight training crops.

This script can cluster:
    1. Full slot crops
    2. PokÃ©mon sprite crops
    3. Type combo crops
    4. Single type icon crops

Run from backend:

    python scripts/cluster_slot_cards.py --kind slots
    python scripts/cluster_slot_cards.py --kind pokemon
    python scripts/cluster_slot_cards.py --kind type-combos
    python scripts/cluster_slot_cards.py --kind type-icons

Examples:

    python scripts/cluster_slot_cards.py --kind pokemon --clusters 80

    python scripts/cluster_slot_cards.py --kind type-combos --clusters 30

    python scripts/cluster_slot_cards.py --kind type-icons --clusters 18

Custom input/output:

    python scripts/cluster_slot_cards.py ^
      --input data/training_dataset/review/unsorted_pokemon ^
      --output data/training_dataset/review/clustered_pokemon ^
      --kind pokemon ^
      --clusters 80
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


BACKEND_DIR = Path(__file__).resolve().parents[2]

REVIEW_DIR = BACKEND_DIR / "data" / "training_dataset" / "review"

DEFAULT_PATHS = {
    "slots": {
        "input": REVIEW_DIR / "unsorted_slots",
        "output": REVIEW_DIR / "clustered_slots",
        "clusters": 40,
    },
    "pokemon": {
        "input": REVIEW_DIR / "unsorted_pokemon",
        "output": REVIEW_DIR / "clustered_pokemon",
        "clusters": 80,
    },
    "type-combos": {
        "input": REVIEW_DIR / "unsorted_type_combos",
        "output": REVIEW_DIR / "clustered_type_combos",
        "clusters": 36,
    },
    "type-icons": {
        "input": REVIEW_DIR / "unsorted_single_type_icons",
        "output": REVIEW_DIR / "clustered_single_type_icons",
        "clusters": 18,
    },
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def read_image(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        print(f"[WARN] Could not read: {path}")

    return image


def clear_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def normalize_feature(feature: np.ndarray) -> np.ndarray:
    feature = feature.astype(np.float32)

    norm = np.linalg.norm(feature)

    if norm > 0:
        feature = feature / norm

    return feature


def resize_with_padding(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize image while preserving aspect ratio, then pad to target size."""

    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError("Invalid image dimensions.")

    scale = min(target_width / width, target_height / height)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    padded = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    padded[
        y_offset : y_offset + new_height,
        x_offset : x_offset + new_width,
    ] = resized

    return padded


def make_histograms(image: np.ndarray, h_bins=32, s_bins=32, v_bins=32) -> list[np.ndarray]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [h_bins], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [s_bins], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [v_bins], [0, 256]).flatten()

    return [hist_h, hist_s, hist_v]


def make_edge_feature(image: np.ndarray, width: int, height: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_equalized = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray_equalized, 60, 160)
    edge_small = cv2.resize(edges, (width, height), interpolation=cv2.INTER_AREA)

    return edge_small.flatten().astype(np.float32) / 255.0


def make_pixel_thumbnail(image: np.ndarray, width: int, height: int) -> np.ndarray:
    thumbnail = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return thumbnail.flatten().astype(np.float32) / 255.0


def make_slot_feature(image: np.ndarray) -> np.ndarray:
    """Feature for full slot cards."""

    card = cv2.resize(image, (224, 96), interpolation=cv2.INTER_AREA)
    card = cv2.GaussianBlur(card, (3, 3), 0)

    height, width = card.shape[:2]

    hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)

    left = hsv[:, : int(width * 0.45)]
    middle = hsv[:, int(width * 0.25) : int(width * 0.70)]
    right = hsv[:, int(width * 0.55) :]

    region_features = []

    for region in [left, middle, right]:
        region_h = cv2.calcHist([region], [0], None, [24], [0, 180]).flatten()
        region_s = cv2.calcHist([region], [1], None, [24], [0, 256]).flatten()
        region_v = cv2.calcHist([region], [2], None, [24], [0, 256]).flatten()
        region_features.extend([region_h, region_s, region_v])

    feature = np.concatenate(
        [
            *make_histograms(card, 32, 32, 32),
            *region_features,
            make_edge_feature(card, 56, 24),
            make_pixel_thumbnail(card, 56, 24),
        ]
    )

    return normalize_feature(feature)


def make_pokemon_feature(image: np.ndarray) -> np.ndarray:
    """Feature for PokÃ©mon-only sprite crops."""

    sprite = resize_with_padding(image, 128, 128)
    sprite = cv2.GaussianBlur(sprite, (3, 3), 0)

    height, width = sprite.shape[:2]

    # Focus slightly more on center where sprite usually appears.
    center = sprite[
        int(height * 0.12) : int(height * 0.88),
        int(width * 0.12) : int(width * 0.88),
    ]

    feature = np.concatenate(
        [
            *make_histograms(sprite, 32, 32, 32),
            *make_histograms(center, 32, 32, 32),
            make_edge_feature(sprite, 64, 64),
            make_pixel_thumbnail(sprite, 64, 64),
        ]
    )

    return normalize_feature(feature)


def make_type_combo_feature(image: np.ndarray) -> np.ndarray:
    """Feature for one or two type icons together."""

    type_crop = resize_with_padding(image, 160, 80)
    type_crop = cv2.GaussianBlur(type_crop, (3, 3), 0)

    height, width = type_crop.shape[:2]

    left = type_crop[:, : width // 2]
    right = type_crop[:, width // 2 :]

    feature = np.concatenate(
        [
            *make_histograms(type_crop, 36, 36, 36),
            *make_histograms(left, 24, 24, 24),
            *make_histograms(right, 24, 24, 24),
            make_edge_feature(type_crop, 80, 40),
            make_pixel_thumbnail(type_crop, 80, 40),
        ]
    )

    return normalize_feature(feature)


def make_type_icon_feature(image: np.ndarray) -> np.ndarray:
    """Feature for a single type icon."""

    icon = resize_with_padding(image, 96, 96)
    icon = cv2.GaussianBlur(icon, (3, 3), 0)

    feature = np.concatenate(
        [
            *make_histograms(icon, 36, 36, 36),
            make_edge_feature(icon, 48, 48),
            make_pixel_thumbnail(icon, 48, 48),
        ]
    )

    return normalize_feature(feature)


def make_feature(image: np.ndarray, kind: str) -> np.ndarray:
    if kind == "slots":
        return make_slot_feature(image)

    if kind == "pokemon":
        return make_pokemon_feature(image)

    if kind == "type-combos":
        return make_type_combo_feature(image)

    if kind == "type-icons":
        return make_type_icon_feature(image)

    raise ValueError(f"Unsupported crop kind: {kind}")


def reduce_features(features: np.ndarray) -> np.ndarray:
    scaled = StandardScaler().fit_transform(features)

    component_count = min(50, scaled.shape[0] - 1, scaled.shape[1])

    if component_count >= 2:
        return PCA(n_components=component_count, random_state=42).fit_transform(scaled)

    return scaled


def cluster_kmeans(features: np.ndarray, cluster_count: int) -> np.ndarray:
    reduced = reduce_features(features)

    model = KMeans(
        n_clusters=cluster_count,
        random_state=42,
        n_init="auto",
    )

    return model.fit_predict(reduced)


def cluster_dbscan(features: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    reduced = reduce_features(features)

    model = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric="euclidean",
    )

    return model.fit_predict(reduced)


def save_results(
    image_paths: list[Path],
    labels: np.ndarray,
    output_dir: Path,
    input_dir: Path,
    kind: str,
    method: str,
) -> None:
    clear_output_dir(output_dir)

    report = []
    summary: dict[str, int] = {}

    for path, label in zip(image_paths, labels):
        label = int(label)

        if label == -1:
            cluster_name = "noise_or_mixed"
        else:
            cluster_name = f"cluster_{label:03d}"

        cluster_dir = output_dir / cluster_name
        cluster_dir.mkdir(parents=True, exist_ok=True)

        destination = cluster_dir / path.name
        shutil.copy2(path, destination)

        summary[cluster_name] = summary.get(cluster_name, 0) + 1

        report.append(
            {
                "source": str(path),
                "cluster": cluster_name,
                "copied_to": str(destination),
            }
        )

    summary_path = output_dir / "cluster_summary.json"
    report_path = output_dir / "cluster_report.json"

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(dict(sorted(summary.items())), file, indent=2)

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print()
    print("Cluster summary:")
    for cluster_name, count in sorted(summary.items()):
        print(f"  {cluster_name}: {count}")

    print()
    print("Settings:")
    print(f"  kind:   {kind}")
    print(f"  method: {method}")

    print()
    print("Input:")
    print(f"  {input_dir}")

    print()
    print("Output:")
    print(f"  {output_dir}")

    print()
    print("Reports:")
    print(f"  {summary_path}")
    print(f"  {report_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--kind",
        choices=["slots", "pokemon", "type-combos", "type-icons"],
        default="slots",
        help="Crop type to cluster.",
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Folder containing unsorted crop images.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Folder where clustered folders will be saved.",
    )

    parser.add_argument(
        "--method",
        choices=["kmeans", "dbscan"],
        default="kmeans",
        help="Clustering method.",
    )

    parser.add_argument(
        "--clusters",
        type=int,
        default=None,
        help="Number of KMeans clusters.",
    )

    parser.add_argument(
        "--eps",
        type=float,
        default=6.0,
        help="DBSCAN eps value.",
    )

    parser.add_argument(
        "--min-samples",
        type=int,
        default=3,
        help="DBSCAN minimum samples.",
    )

    return parser.parse_args()


def resolve_path(path: Path | None, default_path: Path) -> Path:
    if path is None:
        return default_path

    if path.is_absolute():
        return path

    return BACKEND_DIR / path


def main() -> None:
    args = parse_args()

    defaults = DEFAULT_PATHS[args.kind]

    input_dir = resolve_path(args.input, defaults["input"])
    output_dir = resolve_path(args.output, defaults["output"])
    cluster_count = args.clusters if args.clusters is not None else defaults["clusters"]

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    image_paths = list_images(input_dir)

    if not image_paths:
        print(f"No images found in: {input_dir}")
        return

    print(f"Found {len(image_paths)} images.")
    print(f"Crop kind: {args.kind}")

    valid_paths: list[Path] = []
    features: list[np.ndarray] = []

    for index, path in enumerate(image_paths, start=1):
        image = read_image(path)

        if image is None:
            continue

        try:
            feature = make_feature(image, args.kind)
        except Exception as error:
            print(f"[WARN] Skipping {path}: {error}")
            continue

        valid_paths.append(path)
        features.append(feature)

        if index % 25 == 0 or index == len(image_paths):
            print(f"Processed {index}/{len(image_paths)}")

    if len(features) < 2:
        print("Need at least 2 valid images to cluster.")
        return

    feature_array = np.vstack(features)

    if args.method == "kmeans":
        cluster_count = min(cluster_count, len(valid_paths))

        print(f"\nClustering with KMeans: clusters={cluster_count}")
        labels = cluster_kmeans(feature_array, cluster_count)

    else:
        print(
            f"\nClustering with DBSCAN: "
            f"eps={args.eps}, min_samples={args.min_samples}"
        )
        labels = cluster_dbscan(feature_array, args.eps, args.min_samples)

    save_results(
        image_paths=valid_paths,
        labels=labels,
        output_dir=output_dir,
        input_dir=input_dir,
        kind=args.kind,
        method=args.method,
    )


if __name__ == "__main__":
    main()
