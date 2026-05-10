"""Tests image upload validation and storage helpers."""

from io import BytesIO

import pytest
from werkzeug.datastructures import FileStorage

from services import image_service
from services.image_service import ImageValidationError, save_opponent_image


VALID_JPEG = b"\xff\xd8\xff\xe0" + (b"0" * 2048)
VALID_PNG = b"\x89PNG\r\n\x1a\n" + (b"0" * 2048)


# Builds a FileStorage object for image service tests.
def make_file_storage(content=VALID_JPEG, filename="team.jpg", content_type="image/jpeg"):
    return FileStorage(
        stream=BytesIO(content),
        filename=filename,
        content_type=content_type,
    )


# Tests that a valid image is saved with metadata.
def test_save_opponent_image_stores_valid_file(upload_dir, monkeypatch):
    monkeypatch.setattr(image_service, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(
        image_service,
        "assess_opponent_image_quality",
        lambda _path: {"canAnalyze": True, "qualityLevel": "good", "issues": [], "warnings": [], "metrics": {}},
    )
    monkeypatch.setattr(
        image_service,
        "detect_opponent_team",
        lambda _path: {"detectedTeam": []},
    )

    result = save_opponent_image(make_file_storage())

    assert result["status"] == "received"
    assert result["contentType"] == "image/jpeg"
    assert result["sizeBytes"] == len(VALID_JPEG)
    assert result["filename"].endswith(".jpg")
    assert (upload_dir / result["filename"]).exists()
    assert result["debugCropsGenerated"] is True


# Tests that uploads still create debug crops when prediction is skipped.
def test_save_opponent_image_generates_debug_crops_when_detection_skipped(upload_dir, monkeypatch):
    calls = []

    monkeypatch.setattr(image_service, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(
        image_service,
        "assess_opponent_image_quality",
        lambda _path: {"canAnalyze": True, "qualityLevel": "good", "issues": [], "warnings": [], "metrics": {}},
    )

    def fake_detect_opponent_team(path):
        calls.append(path)
        return {"detectedTeam": [{"pokemonName": "Debugmon"}]}

    monkeypatch.setattr(image_service, "detect_opponent_team", fake_detect_opponent_team)

    result = save_opponent_image(make_file_storage(), run_detection=False)

    assert len(calls) == 1
    assert calls[0] == upload_dir / result["filename"]
    assert result["status"] == "received"
    assert result["debugCropsGenerated"] is True
    assert result["detectedTeam"] == []
    assert result["message"] == "Image received. Debug crops generated."


# Tests that rejected photos still run debug crop generation.
def test_save_opponent_image_generates_debug_crops_for_low_quality_photo(upload_dir, monkeypatch):
    calls = []

    monkeypatch.setattr(image_service, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(
        image_service,
        "assess_opponent_image_quality",
        lambda _path: {
            "canAnalyze": False,
            "qualityLevel": "bad",
            "issues": ["Found 4 of 6 opponent cards."],
            "warnings": [],
            "metrics": {},
        },
    )

    def fake_detect_opponent_team(path):
        calls.append(path)
        return {
            "detectedTeam": [],
            "skippedReason": "bad_quality",
            "debugCropPaths": [str(upload_dir / "debug" / "opponent-slot-1.jpg")],
        }

    monkeypatch.setattr(image_service, "detect_opponent_team", fake_detect_opponent_team)

    result = save_opponent_image(make_file_storage(), run_detection=False)

    assert len(calls) == 1
    assert result["status"] == "needs_retake"
    assert result["debugCropsGenerated"] is True
    assert result["detectedTeam"] == []


# Tests that PNG signatures are accepted.
def test_save_opponent_image_accepts_png(upload_dir, monkeypatch):
    monkeypatch.setattr(image_service, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(
        image_service,
        "assess_opponent_image_quality",
        lambda _path: {"canAnalyze": True, "qualityLevel": "good", "issues": [], "warnings": [], "metrics": {}},
    )
    monkeypatch.setattr(
        image_service,
        "detect_opponent_team",
        lambda _path: {"detectedTeam": []},
    )

    result = save_opponent_image(make_file_storage(VALID_PNG, "team.png", "image/png"))

    assert result["filename"].endswith(".png")


# Tests that missing image uploads are rejected.
def test_save_opponent_image_requires_file():
    with pytest.raises(ImageValidationError, match="Image file is required."):
        save_opponent_image(None)


# Tests that unsupported image types are rejected.
def test_save_opponent_image_rejects_unsupported_type():
    file_storage = make_file_storage(filename="team.gif", content_type="image/gif")

    with pytest.raises(ImageValidationError, match="Only JPEG, PNG, and WebP images are supported."):
        save_opponent_image(file_storage)


# Tests that corrupt image data is rejected.
def test_save_opponent_image_rejects_bad_signature():
    file_storage = make_file_storage(b"not an image" * 200, "team.jpg", "image/jpeg")

    with pytest.raises(ImageValidationError, match="not a valid image"):
        save_opponent_image(file_storage)


# Tests that tiny placeholder images are rejected.
def test_save_opponent_image_rejects_tiny_image():
    file_storage = make_file_storage(b"\xff\xd8\xff", "team.jpg", "image/jpeg")

    with pytest.raises(ImageValidationError, match="too small"):
        save_opponent_image(file_storage)
