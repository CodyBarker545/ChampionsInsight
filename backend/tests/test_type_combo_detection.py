"""Tests one-pass type-combo detection behavior."""

import numpy as np

from services import cv_service


def make_icon(color):
    image = np.full((32, 32, 3), color, dtype=np.uint8)
    image[10:22, 10:22] = (245, 245, 245)
    return image


def test_detect_type_method_results_uses_one_combo_prediction(monkeypatch):
    dragon_icon = make_icon((45, 35, 185))
    ground_icon = make_icon((130, 95, 45))
    combo_image = np.concatenate([dragon_icon, ground_icon], axis=1)
    slot_image = np.zeros((80, 220, 3), dtype=np.uint8)
    slot_image[5:37, 100:164] = combo_image

    monkeypatch.setattr(
        cv_service,
        "crop_adaptive_type_icons_from_slot",
        lambda _slot: [
            {
                "index": 1,
                "x": 100,
                "y": 5,
                "width": 32,
                "height": 32,
                "image": dragon_icon,
                "hasSymbol": True,
                "cropSource": "test",
                "cropQuality": 1.0,
            },
            {
                "index": 2,
                "x": 140,
                "y": 5,
                "width": 32,
                "height": 32,
                "image": ground_icon,
                "hasSymbol": True,
                "cropSource": "test",
                "cropQuality": 1.0,
            },
        ],
    )
    monkeypatch.setattr(
        cv_service,
        "extract_detected_type_icon_cluster_box",
        lambda _slot: {
            "x": 100,
            "y": 5,
            "width": 64,
            "height": 32,
        },
    )
    monkeypatch.setattr(
        cv_service,
        "load_type_combo_references",
        lambda: [
            {
                "types": ["dragon", "ground"],
                "typeKey": "dragon_ground",
                "image": combo_image,
                "path": "test/dragon_ground.png",
            }
        ],
    )
    monkeypatch.setattr(
        cv_service,
        "classify_type_by_template",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("single-icon classifier should not run")
        ),
    )
    monkeypatch.setattr(
        cv_service,
        "stitch_type_icon_crops",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("dual type prediction should use the cluster crop")
        ),
    )

    result = cv_service.detect_type_method_results(slot_image)

    assert result["selected"] == ["dragon", "ground"]
    assert result["typeComboDetails"]["typeKey"] == "dragon_ground"
    assert result["typeComboDetails"]["cropSource"] == "type_cluster_candidate"
    assert result["typeComboDetails"]["predictionSource"] == "type_combo_template"


def test_load_type_combo_references_parses_camera_reference_names(upload_dir):
    cv2, _np = cv_service.load_cv_dependencies()
    combo_dir = upload_dir / "type_combo_icons"
    combo_dir.mkdir()

    image = make_icon((45, 35, 185))
    cv2.imwrite(str(combo_dir / "dragon_flying__camera-1.jpg"), image)
    cv2.imwrite(str(combo_dir / "fighting_posion__camera-1.jpg"), image)
    cv2.imwrite(str(combo_dir / "grass-fairy__camera-1.jpg"), image)

    references = cv_service.load_type_combo_references(
        type_combo_reference_dir=combo_dir,
        metadata_path=combo_dir / "missing_metadata.json",
    )

    assert [reference["types"] for reference in references] == [
        ["dragon", "flying"],
        ["fighting", "poison"],
        ["grass", "fairy"],
    ]
