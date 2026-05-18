"""Runs opponent type detection on uploaded images and writes JSON output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from paths import OPPONENT_DEBUG_CROP_DIR
from services.cv_type_service import detect_opponent_team_types


# Builds a compact JSON result for position-based type detection.
def build_type_output(detection_result: dict) -> dict:
    return {
        "image": detection_result["image"],
        "mode": detection_result["mode"],
        "quality": detection_result["quality"],
        "teamTypes": [
            {
                "position": slot["position"],
                "types": slot.get("types", []),
                "typeMethodResults": slot.get("typeMethodResults", {}),
                "typeIconCropPath": slot.get("typeIconCropPath", ""),
                "typeIconCropPaths": slot.get("typeIconCropPaths", []),
            }
            for slot in detection_result["teamTypes"]
        ],
    }


# Runs type detection and saves a JSON file.
def run_type_detection(image_path: Path, output_path: Path) -> dict:
    detection_result = detect_opponent_team_types(image_path, save_debug=True)
    position_output = build_type_output(detection_result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(position_output, indent=2), encoding="utf-8")
    return position_output


# Parses command line options for the type detector script.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect opponent Pokemon type positions from team images.")
    parser.add_argument(
        "--image",
        required=True,
        help="Image path relative to backend, or an absolute image path.",
    )
    parser.add_argument(
        "--output",
        default=str(OPPONENT_DEBUG_CROP_DIR / "detected_opponent_team_types.json"),
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

    result = run_type_detection(image_path, output_path)
    print(f"Saved type detection JSON to {output_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

