"""Builds canonical real Pokemon type-combo reference images."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from paths import SPRITE_ROOT, TYPE_COMBO_REFERENCE_DIR, TYPE_ICON_REFERENCE_DIR
from services import cv_service


DEFAULT_TYPE_DATABASE_PATH = SPRITE_ROOT / "champions_sprite_database.csv"
DEFAULT_TYPE_ICON_DIR = TYPE_ICON_REFERENCE_DIR
DEFAULT_OUTPUT_DIR = TYPE_COMBO_REFERENCE_DIR
DEFAULT_METADATA_PATH = DEFAULT_OUTPUT_DIR / "type_combo_metadata.json"
DEFAULT_BACKGROUND_BGR = cv_service.REFERENCE_CARD_BACKGROUND_BGR


def normalize_type_name(value):
    return str(value or "").strip().lower()


def parse_type_list(value):
    return [
        normalize_type_name(type_name)
        for type_name in str(value or "").split("|")
        if normalize_type_name(type_name)
    ]


def combo_key(types):
    return tuple(sorted(types))


def combo_slug(types):
    return "_".join(types)


def load_real_type_combos(type_database_path):
    combos_by_key = {}

    with Path(type_database_path).open(newline="", encoding="utf-8") as csv_file:
        for row in csv.DictReader(csv_file):
            types = parse_type_list(row.get("types", ""))

            if not types:
                continue

            unique_types = []
            for type_name in types:
                if type_name not in unique_types:
                    unique_types.append(type_name)

            if len(unique_types) > 2:
                continue

            key = combo_key(unique_types)
            record = combos_by_key.setdefault(
                key,
                {
                    "types": unique_types,
                    "pokemon": [],
                },
            )

            record["pokemon"].append({
                "displayName": row.get("display_name", ""),
                "formApiName": row.get("form_api_name", ""),
                "spriteType": row.get("sprite_type", ""),
            })

    return list(combos_by_key.values())


def load_type_icon_paths(type_icon_dir):
    return {
        reference["type"]: Path(reference["path"])
        for reference in cv_service.load_type_icon_references(type_icon_dir)
    }


def replace_border_white_with_background(image, background_bgr=DEFAULT_BACKGROUND_BGR):
    cv2, np = cv_service.load_cv_dependencies()
    if cv_service.is_empty_image(image):
        return image

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    near_white = ((saturation < 45) & (value > 220)).astype("uint8")

    component_count, labels, _stats, _centroids = cv2.connectedComponentsWithStats(near_white)
    if component_count <= 1:
        return image

    edge_labels = set(labels[0, :])
    edge_labels.update(labels[-1, :])
    edge_labels.update(labels[:, 0])
    edge_labels.update(labels[:, -1])
    edge_labels.discard(0)

    if not edge_labels:
        return image

    output = image.copy()
    border_white = np.isin(labels, list(edge_labels))
    output[border_white] = background_bgr
    return output


def build_combo_image(
    type_names,
    type_icon_paths,
    output_path,
    gap=8,
    padding=0,
    background_bgr=DEFAULT_BACKGROUND_BGR,
):
    cv2, np = cv_service.load_cv_dependencies()
    images = []

    for type_name in type_names:
        icon_path = type_icon_paths.get(type_name)
        if icon_path is None:
            raise FileNotFoundError(f"Missing type icon reference for: {type_name}")

        image = cv2.imread(str(icon_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read type icon reference: {icon_path}")

        images.append(replace_border_white_with_background(image, background_bgr))

    if len(images) == 1 and padding == 0:
        cv2.imwrite(str(output_path), images[0])
        return

    target_height = max(image.shape[0] for image in images)
    resized_images = []

    for image in images:
        height, width = image.shape[:2]
        next_width = max(1, round(width * (target_height / height)))
        resized_images.append(
            cv2.resize(image, (next_width, target_height), interpolation=cv2.INTER_AREA)
        )

    output_height = target_height + (padding * 2)
    output_width = (
        sum(image.shape[1] for image in resized_images)
        + (gap * max(0, len(resized_images) - 1))
        + (padding * 2)
    )
    canvas = np.full((output_height, output_width, 3), background_bgr, dtype=np.uint8)

    x_offset = padding
    for image in resized_images:
        image_height, image_width = image.shape[:2]
        y_offset = padding + ((target_height - image_height) // 2)
        canvas[y_offset:y_offset + image_height, x_offset:x_offset + image_width] = image
        x_offset += image_width + gap

    cv2.imwrite(str(output_path), canvas)


def build_type_combo_references(
    type_database_path=DEFAULT_TYPE_DATABASE_PATH,
    type_icon_dir=DEFAULT_TYPE_ICON_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    metadata_path=DEFAULT_METADATA_PATH,
    include_single_types=True,
    gap=8,
    padding=8,
    background_bgr=DEFAULT_BACKGROUND_BGR,
):
    type_database_path = Path(type_database_path)
    type_icon_dir = Path(type_icon_dir)
    output_dir = Path(output_dir)
    metadata_path = Path(metadata_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    type_icon_paths = load_type_icon_paths(type_icon_dir)
    combos = load_real_type_combos(type_database_path)
    combos.sort(key=lambda combo: (len(combo["types"]), combo["types"]))

    metadata = []

    for combo in combos:
        type_names = combo["types"]

        if len(type_names) == 1 and not include_single_types:
            continue

        output_path = output_dir / f"{combo_slug(type_names)}.png"
        effective_gap = 0 if len(type_names) == 1 else gap
        effective_padding = 0 if len(type_names) == 1 else padding
        build_combo_image(
            type_names,
            type_icon_paths,
            output_path,
            gap=effective_gap,
            padding=effective_padding,
            background_bgr=background_bgr,
        )

        metadata.append({
            "types": type_names,
            "typeKey": combo_slug(type_names),
            "unorderedKey": combo_slug(sorted(type_names)),
            "imagePath": str(output_path.relative_to(BACKEND_DIR)).replace("\\", "/"),
            "pokemonCount": len(combo["pokemon"]),
            "examples": combo["pokemon"][:12],
        })

    metadata_path.write_text(
        json.dumps(
            {
                "sourceDatabase": str(type_database_path.relative_to(BACKEND_DIR)).replace("\\", "/"),
                "sourceTypeIconDir": str(type_icon_dir.relative_to(BACKEND_DIR)).replace("\\", "/"),
                "comboCount": len(metadata),
                "typeCombos": metadata,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate one reference image per real Pokemon type combo. "
            "Reversed duplicates are collapsed, so dragon/ground and ground/dragon "
            "produce one canonical reference."
        )
    )
    parser.add_argument(
        "--type-database",
        default=str(DEFAULT_TYPE_DATABASE_PATH),
        help="CSV with a pipe-separated 'types' column.",
    )
    parser.add_argument(
        "--type-icons",
        default=str(DEFAULT_TYPE_ICON_DIR),
        help="Directory containing one PNG per single type.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where generated combo PNGs will be written.",
    )
    parser.add_argument(
        "--metadata",
        default=str(DEFAULT_METADATA_PATH),
        help="Path for generated combo metadata JSON.",
    )
    parser.add_argument(
        "--dual-only",
        action="store_true",
        help="Skip one-type references and generate only two-type combo references.",
    )
    parser.add_argument(
        "--gap",
        type=int,
        default=8,
        help="Pixel gap between type icons in generated dual-type images.",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=8,
        help="Red card padding around generated dual-type images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = build_type_combo_references(
        type_database_path=args.type_database,
        type_icon_dir=args.type_icons,
        output_dir=args.output_dir,
        metadata_path=args.metadata,
        include_single_types=not args.dual_only,
        gap=args.gap,
        padding=args.padding,
    )

    print(f"Built {len(metadata)} type combo references.")
    print(f"Images:   {Path(args.output_dir)}")
    print(f"Metadata: {Path(args.metadata)}")


if __name__ == "__main__":
    main()

