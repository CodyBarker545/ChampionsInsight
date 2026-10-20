"""Active opponent detector: YOLO slot crops and MobileNetV3 classification."""

from pathlib import Path
from typing import Any
from uuid import uuid4

from paths import DATA_DIR, OPPONENT_DEBUG_CROP_DIR
from services import cv_service, slot_object_detection_service
from services.pokemon_mobilenetv3_classifier_service import predict_pokemon_crops

REVIEW_ROOT = DATA_DIR / "training_dataset" / "review"
UNSORTED_SLOT_DIR = REVIEW_ROOT / "unsorted_slots"
UNSORTED_POKEMON_DIR = REVIEW_ROOT / "unsorted_pokemon"
MIN_IDENTITY_CONFIDENCE = 0.85


def format_mobilenet_top_hits(top_predictions, limit=5):
    formatted_hits = []
    for hit in (top_predictions or [])[:limit]:
        formatted_hits.append({
            "label": hit.get("name", ""),
            "confidence": round(float(hit.get("confidence", 0) or 0), 4),
            "similarity": round(float(hit.get("confidence", 0) or 0), 4),
            "path": "",
        })
    return formatted_hits

def safe_filename_part(value):
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(value or "unknown")
    ).strip("_") or "unknown"

def write_image(output_path, image):
    if image is None or cv_service.is_empty_image(image):
        return ""

    cv2, _np = cv_service.load_cv_dependencies()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise cv_service.ComputerVisionError(f"Could not write image: {output_path}")
    return str(output_path)

def write_review_crop(review_dir, prefix, image_stem, position, prediction, image):
    if image is None or cv_service.is_empty_image(image):
        return ""

    filename = (
        f"{safe_filename_part(prefix)}__{safe_filename_part(image_stem)}"
        f"__slot-{position}__{safe_filename_part(prediction)}__{uuid4().hex[:8]}.jpg"
    )
    return write_image(Path(review_dir) / filename, image)

def detect_opponent_team_mobilenetv3_from_image(
    image,
    image_name="in-memory-upload",
    save_debug=False,
    save_review_crops=False,
):
    """Detect opponent slots from an already-decoded image without saving uploads."""

    image_stem = Path(str(image_name or "in-memory-upload")).stem or "in-memory-upload"
    debug_original_path = ""

    if save_debug:
        image_debug_dir = OPPONENT_DEBUG_CROP_DIR / image_stem
        image_debug_dir.mkdir(parents=True, exist_ok=True)
        debug_original_path = write_image(image_debug_dir / "original.jpg", image)

    slot_crops = cv_service.crop_opponent_team_slots_from_image(
        image,
        image_stem=image_stem,
        save_debug=save_debug,
    )
    quality = cv_service.assess_opponent_image_quality_from_image(
        image, card_boxes=[crop['box'] for crop in slot_crops if crop.get('box')])
    detected_team = []
    pokemon_crops = []
    slot_records = []

    for card_crop in slot_crops:
        slot_image = card_crop["image"]
        object_layer = slot_object_detection_service.detect_slot_objects(slot_image)
        pokemon_object = object_layer.get("pokemon_sprite")

        if pokemon_object and not cv_service.is_empty_image(pokemon_object.get("image")):
            pokemon_crop = pokemon_object["image"]
        else:
            pokemon_crop = cv_service.extract_opponent_pokemon_region(slot_image)

        pokemon_crops.append(pokemon_crop)
        slot_records.append({
            "card_crop": card_crop,
            "object_layer": object_layer,
            "pokemon_crop": pokemon_crop,
        })

    predictions = predict_pokemon_crops(pokemon_crops, top_k=5)
    recover_uncertain_pokemon_crops(slot_records, predictions)

    for slot_record, prediction in zip(slot_records, predictions):
        card_crop = slot_record["card_crop"]
        object_layer = slot_record["object_layer"]
        pokemon_crop = slot_record["pokemon_crop"]
        position = card_crop["position"]
        slot_image = card_crop["image"]
        confidence = round(float(prediction.get("confidence", 0) or 0), 4)
        # A softmax winner is only a candidate until it clears acceptance.
        # Keep the raw prediction for review without enriching an uncertain
        # guess into a named roster member and its likely sets.
        accepted = bool(
            prediction.get("pokemon") and not prediction.get("error")
            and float(prediction.get("confidence", 0) or 0) >= MIN_IDENTITY_CONFIDENCE
        )
        pokemon_name = prediction["pokemon"] if accepted else "unknown"

        debug_pokemon_crop_path = ""
        review_paths: dict[str, Any] = {}
        if save_debug:
            image_debug_dir = OPPONENT_DEBUG_CROP_DIR / image_stem
            debug_pokemon_crop_path = write_image(
                image_debug_dir / f"opponent-pokemon-{position}.jpg",
                pokemon_crop,
            )

        if save_review_crops:
            review_paths = {
                "slot": write_review_crop(
                    UNSORTED_SLOT_DIR,
                    "slot",
                    image_stem,
                    position,
                    pokemon_name,
                    slot_image,
                ),
                "pokemon": write_review_crop(
                    UNSORTED_POKEMON_DIR,
                    "pokemon",
                    image_stem,
                    position,
                    pokemon_name,
                    pokemon_crop,
                ),
            }

        detected_team.append({
            "position": position,
            "pokemonName": pokemon_name,
            "name": pokemon_name,
            "confidence": confidence,
            "predictionSource": "mobilenetv3_pokemon",
            "matchReason": "accept_mobilenetv3" if accepted else "needs_review",
            "needsReview": not accepted,
            "pokemonTopCandidates": format_mobilenet_top_hits(prediction.get("top5", [])),
            "mobileNetV3Prediction": prediction,
            "box": card_crop.get("box"),
            "debugCropPath": card_crop.get("debugCropPath", ""),
            "debugPokemonCropPath": debug_pokemon_crop_path,
            "reviewCropPaths": review_paths,
            "referenceImage": "",
            "objectLayerSource": object_layer.get("source", "heuristic"),
            "pokemonCropSource": slot_record.get("crop_source", "detected_sprite"),
            "cropRecovery": slot_record.get("crop_recovery"),
            "cardConfidence": card_crop.get("cardConfidence"),
            "cardSource": card_crop.get("cardSource"),
        })

    card_detection = getattr(slot_crops, 'detection', {
        'unresolvedPositions': sorted(set(range(1, 7)) - {c['position'] for c in slot_crops}),
        'positionsAreRelative': len(slot_crops) < 6,
    })
    quality.setdefault('metrics', {})['detectedCardCount'] = len(slot_crops)
    return {
        "image": str(image_name),
        "quality": quality,
        "debugOriginalPath": debug_original_path,
        "detectedTeam": detected_team,
        "predictionSource": "mobilenetv3_yolo",
        "cardDetection": card_detection,
        "unresolvedPositions": card_detection['unresolvedPositions'],
        "positionsAreRelative": card_detection['positionsAreRelative'],
    }

def recover_uncertain_pokemon_crops(slot_records, predictions):
    """Retry uncertain tight crops with sprite context from the same card.

    Long tails and flowers can fall outside a YOLO sprite box. A wider view
    may recover them, but must clear the normal acceptance threshold and
    improve confidence substantially before replacing the original input.
    """
    retries = []
    for index, (record, prediction) in enumerate(zip(slot_records, predictions)):
        if prediction.get("error") or float(prediction.get("confidence", 0) or 0) >= MIN_IDENTITY_CONFIDENCE:
            continue
        card = record["card_crop"]["image"]
        if cv_service.is_empty_image(card):
            continue
        _height, width = card.shape[:2]
        # Retain sprite context while excluding the type icons on the right.
        wider_crop = card[:, int(width * 0.18):int(width * 0.61)]
        if not cv_service.is_empty_image(wider_crop):
            retries.append((index, wider_crop))

    if not retries:
        return
    alternatives = predict_pokemon_crops([crop for _index, crop in retries], top_k=5)
    for (index, crop), alternative in zip(retries, alternatives):
        original = predictions[index]
        original_confidence = float(original.get("confidence", 0) or 0)
        alternative_confidence = float(alternative.get("confidence", 0) or 0)
        accepted = bool(
            not alternative.get("error") and alternative.get("pokemon")
            and alternative_confidence >= MIN_IDENTITY_CONFIDENCE
            and alternative_confidence - original_confidence >= 0.15
        )
        record = slot_records[index]
        record["crop_recovery"] = {
            "accepted": accepted,
            "originalPrediction": original,
            "widerPrediction": alternative,
        }
        if accepted:
            record["pokemon_crop"] = crop
            record["crop_source"] = "wider_card_retry"
            predictions[index] = alternative

def detect_opponent_team_mobilenetv3(image_path, save_debug=False, save_review_crops=False):
    """Detect opponent slots using existing crop extraction and MobileNetV3 identity classification."""
    image_path = Path(image_path)
    image = cv_service.read_cv_image(image_path)
    return detect_opponent_team_mobilenetv3_from_image(
        image,
        image_name=image_path.name,
        save_debug=save_debug,
        save_review_crops=save_review_crops,
    )
