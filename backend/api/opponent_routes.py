"""Routes for opponent image upload and computer-vision detection."""

import json
import logging
from threading import Thread

from flask import Blueprint, jsonify, request

from api.common import (
    build_sprite_url_from_reference,
    clear_latest_opponent_prediction,
    get_latest_uploaded_image_path,
    get_uploaded_image_path_from_request,
    load_latest_opponent_prediction,
    make_json_safe,
    route_deps,
    save_latest_opponent_prediction,
)


opponent_bp = Blueprint("opponent_routes", __name__)
logger = logging.getLogger(__name__)


def run_background_opponent_detection(filename):
    deps = route_deps()
    image_path = deps.UPLOAD_DIR / filename

    try:
        logger.info("Starting background opponent detection for %s", filename)
        result = deps.detect_opponent_team(image_path)
        result["detectedTeam"] = enrich_detected_opponent_team(
            result.get("detectedTeam", [])
        )
        save_latest_opponent_prediction(result, filename)
        logger.info("Background opponent detection finished for %s", filename)
    except deps.ComputerVisionError as error:
        logger.warning("Background opponent detection failed for %s: %s", filename, error)


@opponent_bp.post("/opponent/image")
def upload_opponent_image():
    logger.info("Upload files received: %s", list(request.files.keys()))
    logger.info("Upload form received: %s", list(request.form.keys()))

    image = request.files.get("image")

    try:
        deps = route_deps()
        skip_detection = request.form.get("skipDetection") == "1"
        background_detection = request.form.get("backgroundDetection") == "1"
        result = deps.save_opponent_image(
            image,
            run_detection=not skip_detection and not background_detection,
        )
        clear_latest_opponent_prediction()
        if background_detection and result.get("status") == "received":
            Thread(
                target=run_background_opponent_detection,
                args=(result["filename"],),
                daemon=True,
            ).start()
            result = {
                **result,
                "status": "queued",
                "message": "Image received. Detection is running in the background.",
                "backgroundDetection": True,
            }
        elif not skip_detection and result.get("status") == "received":
            result = save_latest_opponent_prediction(result, result["filename"])
    except route_deps().ImageValidationError as error:
        logger.warning("Opponent image upload failed: %s", error)
        return jsonify({"error": str(error)}), 400

    return jsonify(make_json_safe(result))


@opponent_bp.get("/opponent/latest")
def get_latest_uploaded_opponent_image():
    latest_image = get_latest_uploaded_image_path()

    if latest_image is None:
        return jsonify({
            "filename": "",
            "modifiedTime": 0,
            "sizeBytes": 0,
        })

    stat = latest_image.stat()
    return jsonify({
        "filename": latest_image.name,
        "modifiedTime": stat.st_mtime,
        "sizeBytes": stat.st_size,
    })


@opponent_bp.get("/opponent/prediction/latest")
def get_latest_opponent_prediction():
    try:
        prediction = load_latest_opponent_prediction()
    except (OSError, json.JSONDecodeError) as error:
        logger.warning("Latest opponent prediction could not be loaded: %s", error)
        return jsonify({"error": "Latest opponent prediction could not be loaded."}), 500

    if prediction is None:
        return jsonify({
            "hasPrediction": False,
            "detectedTeam": [],
        })

    return jsonify({
        "hasPrediction": True,
        **prediction,
    })


@opponent_bp.post("/opponent/detect")
def detect_uploaded_opponent_image():
    deps = route_deps()
    logger.info("UPLOAD_DIR is: %s", deps.UPLOAD_DIR)

    image_path, error_response = get_uploaded_image_path_from_request()

    if error_response:
        return error_response

    try:
        logger.info("Starting opponent detection for %s", image_path.name)

        result = deps.detect_opponent_team(image_path)
        detected_team = result.get("detectedTeam", [])
        result["detectedTeam"] = enrich_detected_opponent_team(detected_team)
        result = save_latest_opponent_prediction(result, image_path.name)

    except deps.ComputerVisionError as error:
        logger.warning("Opponent detection failed for %s: %s", image_path.name, error)
        return jsonify({"error": str(error)}), 400

    return jsonify(make_json_safe(result))


def enrich_detected_opponent_team(detected_team):
    deps = route_deps()
    enriched_team = []

    for pokemon in detected_team:
        pokemon_name = pokemon.get("name") or pokemon.get("pokemonName")
        stats = deps.get_level_50_stats(pokemon_name, nature="hardy")
        type_evidence = pokemon.get("typeEvidence") or {}
        detected_type_evidence = pokemon.get("detectedTypes", [])

        if stats:
            reference_types = stats.get("types", [])
            enriched_team.append({
                **pokemon,
                "name": pokemon_name,
                "pokemonName": pokemon.get("pokemonName") or pokemon_name,
                "types": reference_types,
                "referenceTypes": reference_types,
                "detectedTypes": detected_type_evidence,
                "typeEvidence": type_evidence,
                "baseStats": stats.get("baseStats"),
                "finalStats": stats.get("finalStats"),
                "stats": stats,
                "image": pokemon.get("image") or stats.get("image"),
                "spriteUrl": build_sprite_url_from_reference(pokemon.get("referenceImage")),
                "referenceImage": pokemon.get("referenceImage"),
            })
        else:
            enriched_team.append({
                **pokemon,
                "name": pokemon_name,
                "pokemonName": pokemon.get("pokemonName") or pokemon_name,
                "types": pokemon.get("types") or pokemon.get("detectedTypes", []),
                "referenceTypes": pokemon.get("referenceTypes", []),
                "detectedTypes": detected_type_evidence,
                "typeEvidence": type_evidence,
                "stats": None,
                "baseStats": None,
                "finalStats": None,
                "image": pokemon.get("image"),
                "spriteUrl": build_sprite_url_from_reference(pokemon.get("referenceImage")),
                "referenceImage": pokemon.get("referenceImage"),
            })

    return enriched_team


@opponent_bp.post("/opponent/quality")
def check_uploaded_opponent_image_quality():
    image_path, error_response = get_uploaded_image_path_from_request()

    if error_response:
        return error_response

    try:
        return jsonify(make_json_safe(route_deps().assess_opponent_image_quality(image_path)))
    except route_deps().ComputerVisionError as error:
        logger.warning("Opponent image quality check failed for %s: %s", image_path.name, error)
        return jsonify({"error": str(error)}), 400


@opponent_bp.post("/opponent/types")
def detect_uploaded_opponent_types():
    image_path, error_response = get_uploaded_image_path_from_request()

    if error_response:
        return error_response

    try:
        return jsonify(make_json_safe(route_deps().detect_opponent_team_types(image_path)))
    except route_deps().ComputerVisionError as error:
        logger.warning("Opponent type detection failed for %s: %s", image_path.name, error)
        return jsonify({"error": str(error)}), 400
