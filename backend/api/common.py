"""Shared API helpers for JSON safety, uploads, predictions, and sprites."""

import copy
import time

import numpy as np
from flask import jsonify, request
from werkzeug.utils import secure_filename


def route_deps():
    """Returns the public routes module so tests can monkeypatch dependencies."""
    from api import routes

    return routes


def make_json_safe(value):
    """Convert NumPy/OpenCV values into normal JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}

    if isinstance(value, list):
        return [make_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value


_LATEST_OPPONENT_PREDICTION = None


def get_latest_prediction_path():
    deps = route_deps()
    return deps.UPLOAD_DIR / "latest_opponent_prediction.json"


def delete_legacy_latest_prediction_file():
    try:
        prediction_path = get_latest_prediction_path()
        if prediction_path.exists():
            prediction_path.unlink()
    except OSError:
        pass


def clear_latest_opponent_prediction():
    global _LATEST_OPPONENT_PREDICTION
    delete_legacy_latest_prediction_file()
    _LATEST_OPPONENT_PREDICTION = {
        "hasPrediction": False,
        "detectedTeam": [],
        "savedAt": time.time(),
    }


def save_latest_opponent_prediction(result, filename):
    global _LATEST_OPPONENT_PREDICTION
    delete_legacy_latest_prediction_file()
    prediction = make_json_safe({
        **result,
        "filename": filename,
        "savedAt": time.time(),
    })
    _LATEST_OPPONENT_PREDICTION = prediction

    return copy.deepcopy(prediction)


def load_latest_opponent_prediction():
    if _LATEST_OPPONENT_PREDICTION is None:
        return None

    return copy.deepcopy(_LATEST_OPPONENT_PREDICTION)


def is_valid_uploaded_filename(filename):
    if not isinstance(filename, str) or filename.strip() == "":
        return False

    return filename == secure_filename(filename)


def get_uploaded_image_path_from_request():
    deps = route_deps()
    payload = request.get_json(silent=True) or {}
    filename = payload.get("filename", "")

    if filename:
        if not is_valid_uploaded_filename(filename):
            return None, (jsonify({"error": "Valid uploaded image filename is required."}), 400)

        image_path = deps.UPLOAD_DIR / filename

        if not image_path.exists():
            return None, (jsonify({"error": "Uploaded image was not found."}), 404)

        return image_path, None

    latest_image = get_latest_uploaded_image_path()

    if latest_image is None:
        return None, (jsonify({
            "error": f"No uploaded opponent image was found in {deps.UPLOAD_DIR}."
        }), 404)

    return latest_image, None


def get_latest_uploaded_image_path():
    deps = route_deps()
    allowed_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    if not deps.UPLOAD_DIR.exists():
        return None

    uploaded_images = [
        path
        for path in deps.UPLOAD_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in allowed_extensions
    ]

    if not uploaded_images:
        return None

    return max(uploaded_images, key=lambda path: path.stat().st_mtime)


def build_sprite_url_from_reference(reference_image):
    if not reference_image:
        return None

    reference_path = str(reference_image).replace("\\", "/")

    if "/champions_sprites/normal/" in reference_path:
        filename = reference_path.split("/champions_sprites/normal/")[-1]
        return f"/api/pokemon/sprite/normal/{filename}"

    if "/champions_sprites/shiny/" in reference_path:
        filename = reference_path.split("/champions_sprites/shiny/")[-1]
        return f"/api/pokemon/sprite/shiny/{filename}"

    return None
