"""Validates and stores uploaded opponent team images."""

import logging
from pathlib import Path
from uuid import uuid4

from werkzeug.utils import secure_filename

from paths import UPLOAD_DIR
from services.cv_detection_service import assess_opponent_image_quality, detect_opponent_team
from services.cv_service import ComputerVisionError
from services.pokemon_data_service import enrich_detected_team


MIN_IMAGE_BYTES = 1024
ALLOWED_IMAGE_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
}
logger = logging.getLogger(__name__)


class ImageValidationError(ValueError):
    """Raised when an uploaded image cannot be accepted."""


# Checks whether an uploaded file starts with a supported image header.
def has_valid_image_signature(file_storage):
    header = file_storage.stream.read(16)
    file_storage.stream.seek(0)

    return (
        header.startswith(b"\xff\xd8\xff")
        or header.startswith(b"\x89PNG\r\n\x1a\n")
        or header.startswith(b"RIFF") and header[8:12] == b"WEBP"
    )


# Returns the byte size for an uploaded file.
def get_file_size(file_storage):
    file_storage.stream.seek(0, 2)
    file_size = file_storage.stream.tell()
    file_storage.stream.seek(0)
    return file_size


# Builds a safe unique filename for an uploaded image.
def build_unique_filename(original_filename, mimetype):
    original_name = secure_filename(original_filename)
    extension = ALLOWED_IMAGE_TYPES[mimetype]
    return original_name, f"{Path(original_name).stem or 'opponent'}-{uuid4().hex[:8]}{extension}"


# Validates and stores an opponent team image.
def save_opponent_image(image, run_detection=True):
    if image is None or image.filename == "":
        raise ImageValidationError("Image file is required.")

    if image.mimetype not in ALLOWED_IMAGE_TYPES:
        raise ImageValidationError("Only JPEG, PNG, and WebP images are supported.")

    if not has_valid_image_signature(image):
        raise ImageValidationError("The uploaded file is not a valid image.")

    image_size = get_file_size(image)
    if image_size < MIN_IMAGE_BYTES:
        raise ImageValidationError("The uploaded image is too small to be a phone photo.")

    UPLOAD_DIR.mkdir(exist_ok=True)
    original_name, filename = build_unique_filename(image.filename, image.mimetype)
    saved_path = UPLOAD_DIR / filename
    print("UPLOAD_DIR =", UPLOAD_DIR.resolve())
    print("SAVED ORIGINAL IMAGE =", saved_path.resolve())

    image.save(saved_path)

    print("SAVED EXISTS =", saved_path.exists())

    try:
        quality = assess_opponent_image_quality(saved_path)
    except ComputerVisionError as error:
        logger.warning("Image quality check failed for uploaded image %s: %s", filename, error)
        quality = {
            "canAnalyze": False,
            "qualityLevel": "bad",
            "issues": ["Image could not be read. Upload a clear JPEG, PNG, or WebP photo."],
            "warnings": [],
            "metrics": {},
        }

    if not quality["canAnalyze"]:
        return {
            "status": "needs_retake",
            "filename": filename,
            "originalFilename": original_name,
            "contentType": image.mimetype,
            "sizeBytes": image_size,
            "message": "Image received, but the photo needs to be retaken before detection.",
            "quality": quality,
            "detectedTeam": [],
            "detectionError": "Photo quality is too low for reliable detection.",
        }

    detected_team = []
    detection_error = ""
    if run_detection:
        try:
            logger.info("Running opponent detection for uploaded image %s", filename)
            detection = detect_opponent_team(saved_path)
            detected_team = enrich_detected_team(detection["detectedTeam"])
        except ComputerVisionError as error:
            logger.warning("Opponent detection failed for uploaded image %s: %s", filename, error)
            detection_error = str(error)

    return {
        "status": "received",
        "filename": filename,
        "originalFilename": original_name,
        "contentType": image.mimetype,
        "sizeBytes": image_size,
        "message": (
            "Image received."
            if not run_detection
            else "Image received. Computer vision detection completed."
        ),
        "quality": quality,
        "detectedTeam": detected_team,
        "detectionError": detection_error,
    }
