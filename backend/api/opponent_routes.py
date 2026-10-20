"""Routes for opponent image upload and computer-vision detection."""

import json
import logging
import time
from pathlib import Path
from threading import Thread

from PIL import Image
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


def remove_transient_image(image_path, attempts=5):
    if not image_path:
        return

    path = Path(image_path)
    for attempt in range(attempts):
        try:
            if path.exists() and path.is_file():
                path.unlink()
            return
        except OSError as error:
            if attempt == attempts - 1:
                logger.warning("Could not remove transient uploaded image %s: %s", image_path, error)
                return
            time.sleep(0.2)


def remove_legacy_prediction_file(attempts=5):
    prediction_path = route_deps().UPLOAD_DIR / "latest_opponent_prediction.json"
    for attempt in range(attempts):
        try:
            if prediction_path.exists():
                prediction_path.unlink()
            return
        except OSError as error:
            if attempt == attempts - 1:
                logger.warning("Could not remove legacy prediction file: %s", error)
                return
            time.sleep(0.2)


def run_background_opponent_detection(filename, image_path=None):
    deps = route_deps()
    image_source = image_path

    try:
        logger.info("Starting background MobileNetV3 opponent detection for %s", filename)
        started = time.perf_counter()
        if isinstance(image_source, Path) or isinstance(image_source, str):
            result = deps.detect_opponent_team_mobilenetv3(image_source)
        else:
            result = deps.detect_opponent_team_mobilenetv3_from_image(
                image_source,
                image_name=filename,
            )
        detected_at = time.perf_counter()
        result["detectedTeam"] = enrich_detected_opponent_team(
            result.get("detectedTeam", [])
        )
        result["timingsMs"] = {
            "detection": round((detected_at - started) * 1000),
            "enrichment": round((time.perf_counter() - detected_at) * 1000),
            "totalProcessing": round((time.perf_counter() - started) * 1000),
        }
        save_latest_opponent_prediction(result, filename)
        logger.info("Background MobileNetV3 opponent detection finished for %s", filename)
    except deps.ComputerVisionError as error:
        logger.warning("Background MobileNetV3 opponent detection failed for %s: %s", filename, error)
    finally:
        if isinstance(image_source, Path) or isinstance(image_source, str):
            remove_transient_image(image_source)


@opponent_bp.post("/opponent/image")
def upload_opponent_image():
    logger.info("Upload files received: %s", list(request.files.keys()))
    logger.info("Upload form received: %s", list(request.form.keys()))

    image = request.files.get("image")

    saved_image_path = None
    try:
        deps = route_deps()
        skip_detection = request.form.get("skipDetection") == "1"
        background_detection = request.form.get("backgroundDetection") == "1"
        remove_legacy_prediction_file()
        upload = deps.prepare_opponent_image_for_prediction(image)
        image_array = upload.pop("image")
        result = {
            **upload,
            "message": "Image received. Detection skipped." if skip_detection else "Image received.",
            "quality": deps.assess_opponent_image_quality_from_image(image_array)
            if skip_detection and not background_detection and hasattr(deps, "assess_opponent_image_quality_from_image")
            else {},
            "detectedTeam": [],
            "detectionError": "",
        }
        clear_latest_opponent_prediction()
        if background_detection and result.get("status") == "received":
            Thread(
                target=run_background_opponent_detection,
                args=(result["filename"], image_array),
                daemon=True,
            ).start()
            result = {
                **result,
                "status": "queued",
                "message": "Image received. Detection is running in the background.",
                "backgroundDetection": True,
            }
            saved_image_path = None
        elif not skip_detection and result.get("status") == "received":
            detection = deps.detect_opponent_team_mobilenetv3_from_image(
                image_array,
                image_name=result["filename"],
            )
            detection["detectedTeam"] = enrich_detected_opponent_team(
                detection.get("detectedTeam", [])
            )
            result = save_latest_opponent_prediction(detection, result["filename"])
    except route_deps().ImageValidationError as error:
        logger.warning("Opponent image upload failed: %s", error)
        return jsonify({"error": str(error)}), 400
    finally:
        if image is not None:
            image.close()
        remove_transient_image(saved_image_path)

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
    # Temporarily disabled while MobileNetV3 is the sole active Pokemon
    # predictor. The embedding/FAISS implementation remains in the repo for
    # fallback experiments, but should not run from the app right now.
    return jsonify({
        "error": "Embedding/FAISS opponent detection is disabled. Use /api/opponent/detect-mobilenetv3.",
        "activeRoute": "/api/opponent/detect-mobilenetv3",
    }), 410


@opponent_bp.post("/opponent/detect-mobilenetv3")
def detect_uploaded_opponent_image_with_mobilenetv3():
    deps = route_deps()
    image_path, error_response = get_uploaded_image_path_from_request()

    if error_response:
        return error_response

    try:
        logger.info("Starting MobileNetV3 opponent detection for %s", image_path.name)
        debug_enabled = request.args.get("debug", "").lower() in {"1", "true", "yes"}
        result = deps.detect_opponent_team_mobilenetv3(
            image_path,
            save_debug=debug_enabled,
            save_review_crops=debug_enabled,
        )
        detected_team = result.get("detectedTeam", [])
        result["detectedTeam"] = enrich_detected_opponent_team(detected_team)
        result = save_latest_opponent_prediction(result, image_path.name)
    except deps.ComputerVisionError as error:
        logger.warning("MobileNetV3 opponent detection failed for %s: %s", image_path.name, error)
        return jsonify({"error": str(error)}), 400

    return jsonify(make_json_safe(result))


def enrich_detected_opponent_team(detected_team):
    deps = route_deps()
    enriched_team = []

    for pokemon in detected_team:
        pokemon_name = pokemon.get("name") or pokemon.get("pokemonName")
        summary = deps.build_pokemon_summary(pokemon_name)
        # MobileNet supplies identity, not a reference sprite. Include display
        # data now so phone clients need no extra request for each roster image.
        pokemon = {
            **pokemon,
            "pokemon": {key: value for key, value in summary.items() if key != "moves"},
            "image": pokemon.get("image") or summary.get("image", ""),
            "abilities": summary.get("abilities", []),
            "types": pokemon.get("types") or summary.get("types", []),
        }
        stats = deps.get_level_50_stats(pokemon_name, nature="hardy")

        if stats:
            enriched_team.append({
                **pokemon,
                "name": pokemon_name,
                "pokemonName": pokemon.get("pokemonName") or pokemon_name,
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
    # Temporarily disabled with the other non-MobileNet opponent CV routes.
    # MobileNetV3 crop classification is the active predictor while we validate
    # the supervised model end-to-end.
    return jsonify({
        "error": "Opponent type detection is disabled while MobileNetV3 is the active predictor.",
        "activeRoute": "/api/opponent/detect-mobilenetv3",
    }), 410


@opponent_bp.post("/opponent/predict-pokemon-crop")
def predict_uploaded_pokemon_crop():
    image_file = request.files.get("image")
    if image_file is None:
        return jsonify({"error": "Image file is required."}), 400

    try:
        with Image.open(image_file.stream) as image:
            prediction = route_deps().predict_pokemon_crop(image.convert("RGB"))
    except OSError as error:
        return jsonify({"error": f"Could not read uploaded crop: {error}"}), 400

    status_code = 200 if not prediction.get("error") else 503
    return jsonify(make_json_safe(prediction)), status_code


def choose_pokemon_identity_strategy(mobilenet_prediction: dict, faiss_prediction: dict | None = None):
    """
    Future production rule for blending supervised classification with the current
    embedding/type-based detector. Keep MobileNetV3 as an additive path for now.
    """
    confidence = float(mobilenet_prediction.get("confidence", 0.0) or 0.0)
    if confidence >= 0.85:
        return "accept_mobilenetv3"
    if confidence >= 0.60:
        return "compare_with_faiss_and_type"
    return "needs_review"
