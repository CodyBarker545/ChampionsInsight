"""Runs opponent team detection on one uploaded image and writes JSON output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from paths import OPPONENT_DEBUG_CROP_DIR
from services.cv_service import make_json_safe_cv_value
from services.cv_detection_service import detect_opponent_team


# Builds a compact JSON result for position-based team detection.
def build_position_output(detection_result: dict) -> dict:
    return {
        "image": detection_result["image"],
        "referenceCount": detection_result["referenceCount"],
        "team": [
            {
                "position": slot["position"],
                "pokemonName": slot["pokemonName"],
                "confidence": slot["confidence"],
                "matchReason": slot.get("matchReason", ""),
                "detectedTypes": slot.get("detectedTypes", []),
                "typeMethodResults": slot.get("typeMethodResults", {}),
                "pokemonCropPath": slot.get("debugPokemonCropPath", ""),
                "typeIconCropPath": slot.get("debugTypeIconCropPath", ""),
                "typeIconCropPaths": slot.get("debugTypeIconCropPaths", []),
            }
            for slot in detection_result["detectedTeam"]
        ],
    }


# Runs detection and saves a JSON file.
def run_detection(image_path: Path, output_path: Path) -> dict:
    detection_result = detect_opponent_team(image_path, save_debug=True)
    position_output = make_json_safe_cv_value(build_position_output(detection_result))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(position_output, indent=2), encoding="utf-8")
    return position_output


# Parses command line options for the detector script.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect Pokemon positions from an uploaded team image.")
    parser.add_argument(
        "--image",
        required=True,
        help="Image path relative to backend, or an absolute image path.",
    )
    parser.add_argument(
        "--output",
        default=str(OPPONENT_DEBUG_CROP_DIR / "detected_opponent_team_positions.json"),
        help="JSON output path relative to backend, or an absolute output path.",
    )
    return parser.parse_args()


# Runs the script from the command line.
def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    output_path = Path(args.output)

    if not image_path.is_absolute():
        image_path = BACKEND_DIR / image_path
    if not output_path.is_absolute():
        output_path = BACKEND_DIR / output_path

    result = run_detection(image_path, output_path)
    print(f"Saved detection JSON to {output_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

