"""Tests guided image crop helpers for opponent detection."""

import numpy as np

from services.cv_service import (
    OPPONENT_GUIDED_TEAM_BOXES,
    RECTIFIED_CARD_HEIGHT,
    RECTIFIED_CARD_WIDTH,
    build_quality_result,
    box_to_pixels,
    detect_opponent_card_boxes,
    detect_slot_object_layer,
    detect_slot_objects_from_layer,
    detect_pokemon_from_slot,
    extract_opponent_pokemon_region,
    filter_references_by_types,
    generate_pokemon_sprite_candidates,
    generate_type_icon_crop_candidates,
    is_usable_type_icon_candidate,
    load_cv_dependencies,
    rectify_opponent_card_crop,
    select_detected_types_from_methods,
    select_type_confirmed_reference,
    should_accept_type_icon_crop,
)
from services.slot_object_detection_service import detect_slot_objects
from services.cv_card_service import OpponentCardService
from services.cv_detection_service import get_trusted_detected_types
from services.cv_spirit_service import PokemonSpiritDetectionService
from services.cv_type_service import (
    PokemonTypeDetectionService,
    should_trust_cv_pair_over_weak_embedding,
    select_first_single_type_result,
    select_strongest_single_type_result,
)


# Tests that guided percentage boxes become bounded pixel boxes.
def test_box_to_pixels_uses_image_dimensions():
    box = OPPONENT_GUIDED_TEAM_BOXES[0]

    result = box_to_pixels(box, image_width=1000, image_height=500)

    assert result == {"x": 220, "y": 53, "width": 580, "height": 66}


# Tests that card detection keeps the far-right type icon panel in phone captures.
def test_detect_opponent_card_boxes_keeps_right_type_panel():
    cv2, _np = load_cv_dependencies()
    image = np.zeros((900, 600, 3), dtype=np.uint8)

    for top in (30, 170, 310, 450, 590, 730):
        cv2.rectangle(image, (120, top), (540, top + 95), (10, 0, 210), -1)

    boxes = detect_opponent_card_boxes(image)

    assert len(boxes) == 6
    assert boxes[0]["x"] <= 120
    assert boxes[0]["x"] + boxes[0]["width"] >= 540


# Tests that sparse red background artifacts do not widen lower slot crops.
def test_detect_opponent_card_boxes_ignores_sparse_right_red_noise():
    cv2, _np = load_cv_dependencies()
    image = np.zeros((900, 700, 3), dtype=np.uint8)

    for top in (30, 170, 310, 450, 590, 730):
        cv2.rectangle(image, (120, top), (540, top + 95), (10, 0, 210), -1)

    for y in range(450, 825, 8):
        cv2.line(image, (560, y), (680, min(899, y + 34)), (10, 0, 190), 1)

    boxes = detect_opponent_card_boxes(image)

    assert len(boxes) == 6
    assert max(box["width"] for box in boxes) < 470


# Tests that missing references return an unknown Pokemon result.
def test_detect_pokemon_from_slot_without_references():
    result = detect_pokemon_from_slot(slot_image=None, references=[])

    assert result == {"pokemonName": "unknown", "confidence": 0.0, "referenceImage": ""}


# Tests that empty OpenCV crops do not crash detection.
def test_detect_pokemon_from_slot_with_empty_crop():
    result = detect_pokemon_from_slot(slot_image=np.array([]), references=[{"name": "Test"}])

    assert result == {"pokemonName": "unknown", "confidence": 0.0, "referenceImage": ""}


# Tests that two-type reads keep both types while still exposing the strongest type.
def test_select_strongest_single_type_result_keeps_dual_type_filter():
    result = select_strongest_single_type_result(
        ["water", "ghost"],
        [
            {"type": "water", "confidence": 0.81, "hasSymbol": True, "index": 1},
            {"type": "ghost", "confidence": 0.93, "hasSymbol": True, "index": 2},
        ],
        fallback_source="type_embedding",
    )

    assert result == {
        "selected": ["water", "ghost"],
        "allSelectedTypes": ["water", "ghost"],
        "strongestType": "ghost",
        "strongestTypeConfidence": 0.93,
        "typePredictionSource": "type_embedding_dual",
    }


# Tests that CV-only type reads can still guide dual-type downstream matching.
def test_select_first_single_type_result_keeps_dual_type_filter():
    result = select_first_single_type_result(["grass", "poison"], fallback_source="cv_template")

    assert result == {
        "selected": ["grass", "poison"],
        "allSelectedTypes": ["grass", "poison"],
        "strongestType": "grass",
        "strongestTypeConfidence": None,
        "typePredictionSource": "cv_template_dual",
    }


# Tests that weak fixed fallback embedding reads do not override a coherent CV pair.
def test_should_trust_cv_pair_over_weak_embedding_for_bad_fallback_icon():
    result = should_trust_cv_pair_over_weak_embedding(
        ["fire", "dark"],
        ["electric", "dark"],
        [
            {
                "type": "electric",
                "cropSource": "fixed_type_icon_fallback",
                "cropQuality": 0.0,
            },
            {
                "type": "dark",
                "cropSource": "type_icon_object_proposal",
                "cropQuality": 0.74,
            },
        ],
    )

    assert result is True


# Tests that uncertain type reads do not become hard Pokemon filters.
def test_get_trusted_detected_types_only_trusts_confident_combo():
    weak_combo_with_embedding = {
        "selected": ["ice"],
        "typePredictionSource": "type_embedding",
        "embeddingSelected": ["ice"],
        "typeComboDetails": {
            "predictionSource": "type_combo_template",
            "score": 0.70,
            "needsReview": True,
        },
    }
    confident_combo = {
        "selected": ["grass", "fire"],
        "typePredictionSource": "type_combo_template",
        "typeComboDetails": {
            "predictionSource": "type_combo_template",
            "score": 0.78,
            "needsReview": False,
        },
    }

    assert get_trusted_detected_types(weak_combo_with_embedding) == []
    assert get_trusted_detected_types(confident_combo) == ["grass", "fire"]


# Tests that strong object-layer type embeddings can now guide Pokemon matching.
def test_get_trusted_detected_types_accepts_confident_object_layer_embeddings():
    confident_embedding = {
        "selected": ["water", "flying"],
        "typePredictionSource": "type_embedding_object_layer_dual",
        "embeddingDetails": [
            {
                "type": "water",
                "confidence": 0.95,
                "hasSymbol": True,
                "cropAccepted": True,
            },
            {
                "type": "flying",
                "confidence": 0.91,
                "hasSymbol": True,
                "cropAccepted": True,
            },
        ],
    }
    weak_embedding = {
        "selected": ["water", "flying"],
        "typePredictionSource": "type_embedding_object_layer_dual",
        "embeddingDetails": [
            {
                "type": "water",
                "confidence": 0.95,
                "hasSymbol": True,
                "cropAccepted": True,
            },
            {
                "type": "flying",
                "confidence": 0.72,
                "hasSymbol": True,
                "cropAccepted": True,
            },
        ],
    }

    assert get_trusted_detected_types(confident_embedding) == ["water", "flying"]
    assert get_trusted_detected_types(weak_embedding) == []


# Tests that red-card-heavy fallback crops are not treated as real type icons.
def test_should_accept_type_icon_crop_rejects_red_card_background():
    red_crop = np.full((80, 80, 3), (0, 0, 190), dtype=np.uint8)

    assert should_accept_type_icon_crop(red_crop) is False


# Tests that a colored icon crop with a white symbol passes quality gating.
def test_should_accept_type_icon_crop_accepts_colored_icon_with_symbol():
    icon_crop = np.full((80, 80, 3), (190, 90, 20), dtype=np.uint8)
    icon_crop[30:50, 30:50] = (245, 245, 245)

    assert should_accept_type_icon_crop(icon_crop) is True


# Tests that type-icon candidate search keeps multiple usable crop strategies.
def test_generate_type_icon_crop_candidates_scores_icon_like_crops():
    slot = np.full((180, 420, 3), (0, 0, 190), dtype=np.uint8)
    slot[12:78, 250:316] = (190, 90, 20)
    slot[34:54, 273:293] = (245, 245, 245)
    slot[12:78, 318:384] = (20, 180, 90)
    slot[34:54, 341:361] = (245, 245, 245)

    candidates = generate_type_icon_crop_candidates(slot)

    assert len([candidate for candidate in candidates if candidate["cropQuality"] >= 0.48]) >= 2


# Tests that Pokemon sprite candidate search tightens object proposals.
def test_generate_pokemon_sprite_candidates_scores_multiple_crop_strategies():
    slot = np.full((180, 420, 3), (0, 0, 190), dtype=np.uint8)
    slot[35:145, 75:195] = (40, 170, 60)
    slot[70:95, 115:145] = (245, 245, 245)

    candidates = generate_pokemon_sprite_candidates(slot)
    sources = {candidate["source"] for candidate in candidates}
    best_candidate = max(candidates, key=lambda candidate: candidate["cropQuality"])

    assert {"color_object_proposal", "relaxed_color_object_proposal"}.intersection(sources)
    assert max(candidate["cropQuality"] for candidate in candidates) > 0.45
    assert best_candidate["box"]["width"] < int(420 * 0.40)
    assert best_candidate["box"]["height"] <= int(180 * 0.80)
    assert "rawBox" in best_candidate


# Tests that large fixed sprite fallbacks are tightened around the spirit foreground.
def test_extract_opponent_pokemon_region_trims_background_heavy_fallback():
    slot = np.full((180, 420, 3), (0, 0, 190), dtype=np.uint8)
    slot[42:132, 92:172] = (40, 170, 60)
    slot[68:92, 116:146] = (245, 245, 245)

    region = extract_opponent_pokemon_region(slot)

    assert region.shape[1] < int(420 * 0.40)
    assert region.shape[0] < int(180 * 0.80)


# Tests that one shared object layer selects the expected Pokemon and type roles.
def test_detect_slot_object_layer_selects_role_based_objects():
    slot = np.full((180, 420, 3), (0, 0, 190), dtype=np.uint8)
    slot[42:132, 92:172] = (40, 170, 60)
    slot[68:92, 116:146] = (245, 245, 245)
    slot[12:78, 250:316] = (190, 90, 20)
    slot[34:54, 273:293] = (245, 245, 245)
    slot[12:78, 318:384] = (20, 180, 90)
    slot[34:54, 341:361] = (245, 245, 245)

    object_layer = detect_slot_objects(slot)
    selected_objects = detect_slot_objects_from_layer(object_layer)

    assert object_layer["pokemon_sprite"]["role"] == "pokemon_sprite"
    assert object_layer["type_icon_1"]["role"] == "type_icon_1"
    assert object_layer["type_icon_2"]["role"] == "type_icon_2"
    assert [detected_object["role"] for detected_object in selected_objects] == [
        "pokemon_sprite",
        "type_icon_1",
        "type_icon_2",
    ]


# Tests that red fire icons on red cards are kept when fixed crops see the symbol.
def test_detect_slot_object_layer_keeps_red_type_icon_fallback():
    slot = np.full((180, 420, 3), (0, 0, 190), dtype=np.uint8)
    slot[42:132, 92:172] = (40, 170, 60)
    slot[23:91, 250:315] = (20, 20, 235)
    slot[38:70, 276:291] = (245, 245, 245)
    slot[23:91, 318:383] = (235, 210, 40)
    slot[38:70, 342:360] = (245, 245, 245)

    object_layer = detect_slot_objects(slot)
    type_icon_1 = object_layer["type_icon_1"]
    type_icon_2 = object_layer["type_icon_2"]

    assert type_icon_1 is not None
    assert type_icon_2 is not None
    assert is_usable_type_icon_candidate({
        "cropQuality": 0.0,
        "hasSymbol": True,
        "cropSource": "fixed_type_icon_fallback",
    })


# Tests that a clear red card crop is rectified to the standard card size.
def test_rectify_opponent_card_crop_returns_standard_size_for_red_card():
    cv2, _np = load_cv_dependencies()
    card_crop = np.zeros((120, 320, 3), dtype=np.uint8)
    points = np.array([[[60, 25], [285, 10], [300, 95], [45, 110]]], dtype=np.int32)
    cv2.fillPoly(card_crop, points, (30, 0, 190))

    result = rectify_opponent_card_crop(card_crop)

    assert result.shape[:2] == (RECTIFIED_CARD_HEIGHT, RECTIFIED_CARD_WIDTH)


# Tests that dual type detection prefers exact type matches before loose matches.
def test_filter_references_by_types_prefers_exact_dual_type_matches():
    references = [
        {"name": "Water Only", "types": ["water"]},
        {"name": "Ghost Only", "types": ["ghost"]},
        {"name": "Basculegion Male", "types": ["water", "ghost"]},
        {"name": "Basculegion Female", "types": ["water", "ghost"]},
    ]

    result = filter_references_by_types(references, ["water", "ghost"])

    assert [reference["name"] for reference in result] == ["Basculegion Male", "Basculegion Female"]


# Tests that type-confirmed fallbacks prefer the normal default form when scores tie.
def test_select_type_confirmed_reference_prefers_default_normal_form():
    references = [
        {"name": "Basculegion Female", "form": "Basculegion Female", "isShiny": False},
        {"name": "Basculegion Male", "form": "Basculegion Male", "isShiny": False},
        {"name": "Basculegion Male", "form": "Basculegion Male", "isShiny": True},
    ]

    result = select_type_confirmed_reference(references, best_reference=None)

    assert result["name"] == "Basculegion Male"


# Tests that the type selector trusts component detection first.
def test_select_detected_types_from_methods_prefers_component_types():
    result = select_detected_types_from_methods({
        "component": ["grass"],
        "template": [],
        "fixed": ["fire"],
        "symbolSquare": ["grass"],
    })

    assert result == ["grass"]


# Tests that fixed boxes only rescue clear single fire detections.
def test_select_detected_types_from_methods_uses_fixed_fire_fallback():
    result = select_detected_types_from_methods({
        "component": [],
        "template": [],
        "fixed": ["fire"],
        "symbolSquare": ["poison"],
    })

    assert result == ["fire"]


# Tests that symbol-square detections need agreement before they affect output.
def test_select_detected_types_from_methods_requires_symbol_square_agreement():
    result = select_detected_types_from_methods({
        "component": [],
        "template": [],
        "fixed": [],
        "symbolSquare": ["ghost", "dragon"],
    })

    assert result == []


# Tests that image quality rejects photos missing the six opponent cards.
def test_build_quality_result_rejects_missing_cards():
    result = build_quality_result(
        image_width=1000,
        image_height=1000,
        card_boxes=[{"x": 100, "y": 100, "width": 200, "height": 100}],
        sharpness_score=100.0,
        average_brightness=120.0,
        overexposed_ratio=0.0,
        underexposed_ratio=0.0,
    )

    assert result["canAnalyze"] is False
    assert result["qualityLevel"] == "bad"
    assert "Found 1 of 6" in result["issues"][0]


# Tests that image quality warns on dark photos without rejecting otherwise good card crops.
def test_build_quality_result_warns_for_dark_usable_photo():
    card_boxes = [
        {"x": 200, "y": 100 + index * 120, "width": 500, "height": 90}
        for index in range(6)
    ]

    result = build_quality_result(
        image_width=1000,
        image_height=900,
        card_boxes=card_boxes,
        sharpness_score=100.0,
        average_brightness=70.0,
        overexposed_ratio=0.0,
        underexposed_ratio=0.5,
    )

    assert result["canAnalyze"] is True
    assert result["qualityLevel"] == "warning"
    assert result["warnings"]


# Tests that the card service exposes card crop helpers.
def test_opponent_card_service_rectifies_card_crop():
    cv2, _np = load_cv_dependencies()
    card_crop = np.zeros((120, 320, 3), dtype=np.uint8)
    points = np.array([[[60, 25], [285, 10], [300, 95], [45, 110]]], dtype=np.int32)
    cv2.fillPoly(card_crop, points, (30, 0, 190))

    result = OpponentCardService().rectify_card_crop(card_crop)

    assert result.shape[:2] == (RECTIFIED_CARD_HEIGHT, RECTIFIED_CARD_WIDTH)


# Tests that the type service can be initialized without running detection.
def test_pokemon_type_detection_service_initializes():
    service = PokemonTypeDetectionService()

    assert service.type_references


# Tests that the spirit service accepts injected reference paths and loads references.
def test_pokemon_spirit_detection_service_initializes():
    service = PokemonSpiritDetectionService()

    assert service.references
