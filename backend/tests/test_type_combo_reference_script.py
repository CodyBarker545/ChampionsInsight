"""Tests real type-combo reference generation."""

import csv
import json

from scripts.build_type_combo_references import (
    DEFAULT_BACKGROUND_BGR,
    build_type_combo_references,
    replace_border_white_with_background,
)
from services import cv_service


def write_type_icon(path, color):
    cv2, np = cv_service.load_cv_dependencies()
    image = np.full((24, 24, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), image)


def test_build_type_combo_references_collapses_reversed_dual_types(upload_dir):
    database_path = upload_dir / "champions_sprite_database.csv"
    icon_dir = upload_dir / "type_icons"
    output_dir = upload_dir / "type_combo_icons"
    metadata_path = output_dir / "type_combo_metadata.json"

    icon_dir.mkdir()
    write_type_icon(icon_dir / "dragon.png", (30, 30, 180))
    write_type_icon(icon_dir / "ground.png", (120, 90, 40))

    with database_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["display_name", "form_api_name", "sprite_type", "types"],
        )
        writer.writeheader()
        writer.writerow({
            "display_name": "Garchomp",
            "form_api_name": "garchomp",
            "sprite_type": "normal",
            "types": "dragon|ground",
        })
        writer.writerow({
            "display_name": "Reverse Test",
            "form_api_name": "reverse-test",
            "sprite_type": "normal",
            "types": "ground|dragon",
        })

    metadata = build_type_combo_references(
        type_database_path=database_path,
        type_icon_dir=icon_dir,
        output_dir=output_dir,
        metadata_path=metadata_path,
        include_single_types=False,
    )

    assert [record["types"] for record in metadata] == [["dragon", "ground"]]
    assert (output_dir / "dragon_ground.png").exists()
    assert not (output_dir / "ground_dragon.png").exists()

    saved_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert saved_metadata["comboCount"] == 1


def test_replace_border_white_keeps_internal_white_symbol():
    cv2, np = cv_service.load_cv_dependencies()
    image = np.full((24, 24, 3), (255, 255, 255), dtype=np.uint8)
    image[4:20, 4:20] = (40, 140, 80)
    image[10:14, 10:14] = (255, 255, 255)

    output = replace_border_white_with_background(image)

    assert output[0, 0].tolist() == list(DEFAULT_BACKGROUND_BGR)
    assert output[2, 12].tolist() == list(DEFAULT_BACKGROUND_BGR)
    assert output[12, 12].tolist() == [255, 255, 255]
    assert output[5, 5].tolist() == [40, 140, 80]
