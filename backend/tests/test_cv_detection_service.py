"""Tests orchestration behavior for opponent computer-vision detection."""

from services.cv_detection_service import OpponentDetectionService


class FakeCardService:
    def __init__(self):
        self.crop_calls = []

    def assess_quality(self, _image_path):
        return {
            "canAnalyze": False,
            "qualityLevel": "bad",
            "issues": ["Found 4 of 6 opponent cards."],
            "warnings": [],
            "metrics": {},
        }

    def crop_team_slots(self, image_path, save_debug=False):
        self.crop_calls.append((image_path, save_debug))
        return [
            {
                "position": 1,
                "box": {"x": 1, "y": 2, "width": 3, "height": 4},
                "image": None,
                "debugCropPath": "debug/opponent-slot-1.jpg",
            }
        ]


class FakeSpiritService:
    references = ["one-reference"]


# Tests that low-quality images still produce best-effort debug crops.
def test_bad_quality_detection_still_writes_debug_crops(upload_dir):
    image_path = upload_dir / "upload.jpg"
    image_path.write_bytes(b"fake image")
    card_service = FakeCardService()
    service = OpponentDetectionService(
        card_service=card_service,
        type_service=None,
        spirit_service=FakeSpiritService(),
        debug_dir=upload_dir / "debug",
    )
    service.prepare_detection_image = lambda path, save_debug=True: (path, {"imagePath": str(path)})
    service.write_debug_original_image = lambda path: str(upload_dir / "debug" / path.name / "original.jpg")

    result = service.detect_team(image_path, save_debug=True)

    assert result["skippedReason"] == "bad_quality"
    assert result["debugOriginalPath"].endswith("original.jpg")
    assert result["debugCropPaths"] == ["debug/opponent-slot-1.jpg"]
    assert card_service.crop_calls == [(image_path, True)]
