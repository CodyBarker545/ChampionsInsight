"""Validates and stores uploaded opponent team images."""

import logging
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np
from PIL import Image, ImageOps
from werkzeug.utils import secure_filename

from paths import UPLOAD_DIR
from services.cv_service import ComputerVisionError, assess_opponent_image_quality
from services.opponent_detection_service import detect_opponent_team_mobilenetv3 as detect_opponent_team
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


def validate_opponent_image_upload(image):
    if image is None or image.filename == "":
        raise ImageValidationError("Image file is required.")

    if image.mimetype not in ALLOWED_IMAGE_TYPES:
        raise ImageValidationError("Only JPEG, PNG, and WebP images are supported.")

    if not has_valid_image_signature(image):
        raise ImageValidationError("The uploaded file is not a valid image.")

    image_size = get_file_size(image)
    if image_size < MIN_IMAGE_BYTES:
        raise ImageValidationError("The uploaded image is too small to be a phone photo.")

    original_name, filename = build_unique_filename(image.filename, image.mimetype)
    return original_name, filename, image_size


def prepare_opponent_image_for_prediction(image):
    original_name, filename, image_size = validate_opponent_image_upload(image)

    try:
        with Image.open(image.stream) as pil_image:
            rgb_image = ImageOps.exif_transpose(pil_image).convert("RGB")
            image_array = np.array(rgb_image)[:, :, ::-1].copy()
    except OSError as error:
        raise ImageValidationError(f"Could not read uploaded image: {error}") from error

    return {
        "status": "received",
        "filename": filename,
        "originalFilename": original_name,
        "contentType": image.mimetype,
        "sizeBytes": image_size,
        "image": image_array,
    }


# Validates and stores an opponent team image.
def save_opponent_image(image, run_detection=True, persist=True):
    original_name, filename, image_size = validate_opponent_image_upload(image)
    if persist:
        UPLOAD_DIR.mkdir(exist_ok=True)
        saved_path = UPLOAD_DIR / filename
    else:
        temp_dir = Path(tempfile.gettempdir()) / "champions-insight-uploads"
        temp_dir.mkdir(exist_ok=True)
        saved_path = temp_dir / filename

    image.save(saved_path)

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

    quality_warning = not quality.get("canAnalyze", True)
    debug_crops_generated = False
    detection_error = ""
    detected_team = []
    if run_detection:
        try:
            logger.info(
                "Running opponent detection for uploaded image %s",
                filename,
            )
            detection = detect_opponent_team(saved_path)
            debug_crops_generated = True
            detected_team = enrich_detected_team(detection.get("detectedTeam", []))
        except ComputerVisionError as error:
            logger.warning("Opponent detection failed for uploaded image %s: %s", filename, error)
            detection_error = str(error)

    if quality_warning:
        message = (
            "Image received. Detection was attempted despite photo quality warnings."
            if run_detection
            else "Image received. Detection will run despite photo quality warnings."
        )
    elif not run_detection:
        message = "Image received. Detection skipped."
    else:
        message = "Image received. Computer vision detection completed."

    return {
        "status": "received",
        "filename": filename,
        "savedPath": str(saved_path),
        "originalFilename": original_name,
        "contentType": image.mimetype,
        "sizeBytes": image_size,
        "message": message,
        "quality": quality,
        "detectedTeam": detected_team,
        "detectionError": detection_error,
        "debugCropsGenerated": debug_crops_generated,
    }
