"""Detects opponent Pokemon from guided-camera opponent team images."""

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

from paths import (
    BACKEND_DIR,
    OPPONENT_DEBUG_CROP_DIR,
    POKEMON_REFERENCE_DIR,
    SPRITE_METADATA_PATH,
    SPRITE_ROOT,
    TYPE_COMBO_REFERENCE_DIR,
    TYPE_COMBO_REFERENCE_METADATA_PATH,
    TYPE_ICON_REFERENCE_DIR,
)

REFERENCE_IMAGE_DIR = SPRITE_ROOT / "normal"
SHINY_REFERENCE_IMAGE_DIR = SPRITE_ROOT / "shiny"
REFERENCE_METADATA_PATH = SPRITE_METADATA_PATH
EXTRA_REFERENCE_IMAGE_DIR = POKEMON_REFERENCE_DIR
TYPE_REFERENCE_IMAGE_DIR = TYPE_ICON_REFERENCE_DIR
TYPE_COMBO_REFERENCE_IMAGE_DIR = TYPE_COMBO_REFERENCE_DIR

OPPONENT_GUIDED_TEAM_BOXES = [
    {"left": 0.220, "top": 0.106, "width": 0.580, "height": 0.132},
    {"left": 0.220, "top": 0.244, "width": 0.580, "height": 0.132},
    {"left": 0.220, "top": 0.382, "width": 0.580, "height": 0.132},
    {"left": 0.220, "top": 0.520, "width": 0.580, "height": 0.132},
    {"left": 0.220, "top": 0.659, "width": 0.580, "height": 0.133},
    {"left": 0.220, "top": 0.799, "width": 0.580, "height": 0.135},
]

SUPPORTED_REFERENCE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
MATCH_REGION_WIDTH = 420
EXPECTED_OPPONENT_TEAM_SIZE = 6
RECTIFIED_CARD_WIDTH = 640
RECTIFIED_CARD_HEIGHT = 220
MIN_CARD_RECTIFICATION_SKEW = 3.0
MIN_ACCEPTABLE_SHARPNESS = 45.0
MIN_ACCEPTABLE_CARD_AREA_RATIO = 0.08
MAX_ACCEPTABLE_CARD_CENTER_DRIFT_RATIO = 0.22
MAX_ACCEPTABLE_SPACING_VARIATION_RATIO = 0.38
MAX_ACCEPTABLE_OVEREXPOSED_RATIO = 0.18
WARNING_UNDEREXPOSED_RATIO = 0.35
REFERENCE_CARD_BACKGROUND_BGR = (55, 0, 145)
MATCH_SCALES = (0.80, 1.00, 1.20)
PARTIAL_MATCH_SCALES = (0.42, 0.56, 0.70)
MIN_DETECTION_CONFIDENCE = 0.40
TYPE_CONFIRMED_CONFIDENCE = 0.75
TYPE_SUPPORTED_CONFIDENCE = 0.55
TYPE_COMBO_TRUST_THRESHOLD = 0.74
COLOR_RERANK_LIMIT = 24
FEATURE_RERANK_LIMIT = 16
FEATURE_SCORE_WEIGHT = 1.15
MIN_TYPE_ICON_REGION_HEIGHT = 180
YOLO_SLOT_OBJECT_MODEL_PATH = (
    BACKEND_DIR
    / "data"
    / "cv"
    / "models"
    / "slot_object_detector"
    / "yolov8n_slot_objects_padded_clean_types"
    / "weights"
    / "best.pt"
)
YOLO_SLOT_OBJECT_CONFIDENCE = 0.30
YOLO_SLOT_OBJECT_IMAGE_SIZE = 640


logger = logging.getLogger(__name__)
_DEFAULT_DETECTOR = None
_YOLO_SLOT_OBJECT_MODEL = None
_YOLO_SLOT_OBJECT_MODEL_LOAD_FAILED = False


class ComputerVisionError(RuntimeError):
    """Raised when opponent image detection cannot run."""


class OpponentTeamDetector:
    """Detects opponent team Pokemon using cached reference templates."""

    # Stores detector paths and preloads reference images for reuse.
    def __init__(
        self,
        reference_dir=REFERENCE_IMAGE_DIR,
        metadata_path=REFERENCE_METADATA_PATH,
        debug_dir=OPPONENT_DEBUG_CROP_DIR,
        extra_reference_dir=EXTRA_REFERENCE_IMAGE_DIR,
        type_reference_dir=TYPE_REFERENCE_IMAGE_DIR,
    ):
        self.reference_dir = Path(reference_dir)
        self.metadata_path = Path(metadata_path)
        self.debug_dir = Path(debug_dir)
        self.extra_reference_dir = Path(extra_reference_dir)
        self.type_reference_dir = Path(type_reference_dir)
        self.references = self.load_reference_images()
        self.type_references = self.load_type_references()

    # Loads and prepares all reference images once for repeated detections.
    def load_reference_images(self):
        return load_reference_images(
            reference_dir=self.reference_dir,
            metadata_path=self.metadata_path,
            extra_reference_dir=self.extra_reference_dir,
        )

    # Loads type icon references once for repeated opponent card reads.
    def load_type_references(self):
        return load_type_icon_references(self.type_reference_dir)

    # Detects all opponent slots from one uploaded image.
    def detect_team(self, image_path, save_debug=True):
        return detect_opponent_team_with_references(
            image_path=Path(image_path),
            references=self.references,
            type_references=self.type_references,
            debug_dir=self.debug_dir,
            save_debug=save_debug,
        )

    # Detects only opponent type icons from one uploaded image.
    def detect_team_types(self, image_path, save_debug=True):
        return detect_opponent_team_types_with_references(
            image_path=Path(image_path),
            type_references=self.type_references,
            debug_dir=self.debug_dir,
            save_debug=save_debug,
        )


# Imports OpenCV and NumPy only when detection is requested.
def load_cv_dependencies():
    try:
        import cv2
        import numpy as np
    except ImportError as error:
        raise ComputerVisionError("OpenCV dependencies are not installed.") from error

    return cv2, np


# Checks whether an OpenCV image has usable pixel data.
def is_empty_image(image):
    return image is None or not hasattr(image, "size") or image.size == 0


# Reads an image from disk for computer vision processing.
def read_cv_image(image_path):
    cv2, _np = load_cv_dependencies()
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ComputerVisionError(f"Could not read image: {image_path}")

    return image


# Converts a guided percentage box to pixel coordinates.
def box_to_pixels(box, image_width, image_height):
    x = int(round(box["left"] * image_width))
    y = int(round(box["top"] * image_height))
    width = int(round(box["width"] * image_width))
    height = int(round(box["height"] * image_height))

    x = max(0, min(x, image_width - 1))
    y = max(0, min(y, image_height - 1))
    width = max(1, min(width, image_width - x))
    height = max(1, min(height, image_height - y))

    return {"x": x, "y": y, "width": width, "height": height}


def clamp_box_to_image(box, image_width, image_height):
    x = max(0, min(int(round(box["x"])), image_width - 1))
    y = max(0, min(int(round(box["y"])), image_height - 1))
    width = max(1, min(int(round(box["width"])), image_width - x))
    height = max(1, min(int(round(box["height"])), image_height - y))
    return {"x": x, "y": y, "width": width, "height": height}


def crop_image_box(image, box):
    if is_empty_image(image):
        return image

    height, width = image.shape[:2]
    clamped = clamp_box_to_image(box, width, height)
    x = clamped["x"]
    y = clamped["y"]
    return image[y:y + clamped["height"], x:x + clamped["width"]]


def expand_box(box, padding, image_width, image_height):
    return clamp_box_to_image(
        {
            "x": box["x"] - padding,
            "y": box["y"] - padding,
            "width": box["width"] + padding * 2,
            "height": box["height"] + padding * 2,
        },
        image_width,
        image_height,
    )
# Normalized team-column size expected by the backend slot detector.
NORMALIZED_TEAM_COLUMN_WIDTH = 1215
NORMALIZED_TEAM_COLUMN_HEIGHT = 2160


def prepare_opponent_team_column_image(
    image_path,
    debug_dir=OPPONENT_DEBUG_CROP_DIR,
    save_debug=False,
):
    """
    Converts a raw full phone photo into a normalized opponent team-column image.

    This lets users upload/take a normal full-screen photo instead of perfectly
    matching the frontend guide box.

    Flow:
    1. Read raw photo.
    2. Find red opponent card regions anywhere in the image.
    3. Build one bounding box around the best vertical group of red cards.
    4. Crop that team column with padding.
    5. Resize to 1215x2160 so the existing slot detector can run normally.
    """
    cv2, np = load_cv_dependencies()
    image_path = Path(image_path)
    image = read_cv_image(image_path)

    if is_empty_image(image):
        return {
            "imagePath": str(image_path),
            "usedRawPhotoColumnDetection": False,
            "reason": "empty-image",
            "teamColumnBox": None,
            "rawCardBoxes": [],
        }

    image_height, image_width = image.shape[:2]

    guided_boxes = detect_opponent_card_boxes(image)
    if image_already_matches_guided_team_column(image_width, image_height, guided_boxes):
        return {
            "imagePath": str(image_path),
            "usedRawPhotoColumnDetection": False,
            "reason": "already-guided-team-column",
            "teamColumnBox": None,
            "rawCardBoxes": guided_boxes,
            "normalizedSize": {
                "width": image_width,
                "height": image_height,
            },
        }

    raw_card_boxes = detect_raw_photo_red_card_boxes(image)

    if len(raw_card_boxes) >= 3:
        team_column_box = build_team_column_box_from_card_boxes(
            raw_card_boxes,
            image_width=image_width,
            image_height=image_height,
        )
        reason = "red-card-column"
    else:
        team_column_box = None
        reason = "not-enough-red-card-rows"

    if not team_column_box:
        return {
            "imagePath": str(image_path),
            "usedRawPhotoColumnDetection": False,
            "reason": "no-team-column-found",
            "teamColumnBox": None,
            "rawCardBoxes": raw_card_boxes,
        }

    team_column_crop = crop_image_box(image, team_column_box)

    if is_empty_image(team_column_crop):
        return {
            "imagePath": str(image_path),
            "usedRawPhotoColumnDetection": False,
            "reason": "empty-team-column-crop",
            "teamColumnBox": team_column_box,
            "rawCardBoxes": raw_card_boxes,
        }

    normalized = cv2.resize(
        team_column_crop,
        (NORMALIZED_TEAM_COLUMN_WIDTH, NORMALIZED_TEAM_COLUMN_HEIGHT),
        interpolation=cv2.INTER_CUBIC,
    )

    image_debug_dir = Path(debug_dir) / image_path.stem
    normalized_path = image_debug_dir / "normalized-team-column.jpg"

    if save_debug:
        image_debug_dir.mkdir(parents=True, exist_ok=True)
        overlay = image.copy()
        draw_debug_raw_team_column_overlay(
            overlay,
            team_column_box=team_column_box,
            raw_card_boxes=raw_card_boxes,
        )

        overlay_path = image_debug_dir / "raw-team-column-detection.jpg"

        cv2.imwrite(str(overlay_path), overlay)
        cv2.imwrite(str(normalized_path), normalized)

        output_path = normalized_path
    else:
        image_debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(normalized_path), normalized)
        output_path = normalized_path

    return {
        "imagePath": str(output_path),
        "usedRawPhotoColumnDetection": True,
        "reason": reason,
        "teamColumnBox": team_column_box,
        "rawCardBoxes": raw_card_boxes,
        "normalizedSize": {
            "width": NORMALIZED_TEAM_COLUMN_WIDTH,
            "height": NORMALIZED_TEAM_COLUMN_HEIGHT,
        },
    }


def image_already_matches_guided_team_column(image_width, image_height, card_boxes):
    if image_width <= 0 or image_height <= 0:
        return False

    if len(card_boxes) != EXPECTED_OPPONENT_TEAM_SIZE:
        return False

    aspect = image_width / max(1, image_height)
    expected_aspect = NORMALIZED_TEAM_COLUMN_WIDTH / NORMALIZED_TEAM_COLUMN_HEIGHT
    if abs(aspect - expected_aspect) > 0.10:
        return False

    center_drift_ratio = calculate_card_center_drift_ratio(card_boxes, image_width)
    spacing_variation_ratio = calculate_card_spacing_variation_ratio(card_boxes)
    if center_drift_ratio > MAX_ACCEPTABLE_CARD_CENTER_DRIFT_RATIO:
        return False
    if spacing_variation_ratio > MAX_ACCEPTABLE_SPACING_VARIATION_RATIO:
        return False

    return True


def detect_raw_photo_red_card_boxes(image):
    """
    Finds the six horizontal red opponent cards in a raw full phone photo.

    This version uses row projection instead of relying mainly on contours.
    It is better for phone photos where glare/background connects red areas.
    """
    if is_empty_image(image):
        return []

    cv2, np = load_cv_dependencies()
    image_height, image_width = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red_1 = cv2.inRange(
        hsv,
        np.array([0, 55, 45]),
        np.array([14, 255, 255]),
    )
    lower_red_2 = cv2.inRange(
        hsv,
        np.array([145, 55, 45]),
        np.array([179, 255, 255]),
    )

    red_mask = lower_red_1 | lower_red_2

    # Remove tiny noise, but do not connect separate cards too aggressively.
    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )
    red_mask = cv2.morphologyEx(
        red_mask,
        cv2.MORPH_CLOSE,
        np.ones((9, 5), np.uint8),
    )

    # Focus on the middle/right part where the opponent cards normally sit.
    # This avoids left-side screen background/glare influencing the row detector.
    search_left = int(image_width * 0.12)
    search_right = int(image_width * 0.98)
    search_mask = red_mask[:, search_left:search_right]

    search_width = search_right - search_left

    # Count red pixels per row.
    row_counts = (search_mask > 0).sum(axis=1)

    # A red card row usually has a strong horizontal red band.
    # Use a relaxed threshold because glare and sprites break the red area.
    row_threshold = max(12, int(search_width * 0.16))

    raw_runs = build_mask_runs(
        row_counts,
        row_threshold,
        min_length=max(8, int(image_height * 0.018)),
    )

    # Merge split pieces of the same card row.
    merged_runs = merge_raw_photo_card_row_runs(
        raw_runs,
        image_height=image_height,
    )

    boxes = []

    for run_start, run_end in merged_runs:
        run_height = run_end - run_start + 1

        # Filter out tiny red UI fragments.
        if run_height < image_height * 0.035:
            continue

        if run_height > image_height * 0.22:
            continue

        band_mask = search_mask[run_start:run_end + 1]
        column_counts = (band_mask > 0).sum(axis=0)

        # Find x range for this row.
        column_threshold = max(3, int(run_height * 0.12))
        column_runs = build_mask_runs(
            column_counts,
            column_threshold,
            min_length=max(12, int(search_width * 0.08)),
        )

        if not column_runs:
            continue

        # Use the widest red region in that row.
        best_column_run = max(
            column_runs,
            key=lambda run: run[1] - run[0] + 1,
        )

        x1 = search_left + best_column_run[0]
        x2 = search_left + best_column_run[1]

        width = x2 - x1 + 1
        height = run_height

        if width < image_width * 0.22:
            continue

        aspect = width / max(1, height)

        # Cards are horizontal bars.
        if aspect < 1.4:
            continue

        # Add small padding around each detected card.
        x_padding = int(width * 0.04)
        y_padding = int(height * 0.08)

        box = clamp_box_to_image(
            {
                "x": x1 - x_padding,
                "y": run_start - y_padding,
                "width": width + x_padding * 2,
                "height": height + y_padding * 2,
            },
            image_width,
            image_height,
        )

        boxes.append({
            **box,
            "area": int(box["width"] * box["height"]),
            "aspect": round(float(box["width"] / max(1, box["height"])), 4),
        })

    boxes = remove_duplicate_raw_card_boxes(boxes)
    boxes = filter_best_vertical_card_group(
        boxes,
        image_width=image_width,
        image_height=image_height,
    )

    return sorted(boxes, key=lambda box: box["y"])[:EXPECTED_OPPONENT_TEAM_SIZE]



def merge_nearby_raw_red_boxes(boxes, image_width, image_height):
    """
    Merges red fragments that belong to the same card.
    Glare/text/sprite overlays can split one red card into several contours.
    """
    if not boxes:
        return []

    sorted_boxes = sorted(boxes, key=lambda box: (box["y"], box["x"]))
    merged = []

    for box in sorted_boxes:
        did_merge = False

        for index, existing in enumerate(merged):
            if raw_boxes_should_merge(existing, box, image_width, image_height):
                x1 = min(existing["x"], box["x"])
                y1 = min(existing["y"], box["y"])
                x2 = max(existing["x"] + existing["width"], box["x"] + box["width"])
                y2 = max(existing["y"] + existing["height"], box["y"] + box["height"])

                merged[index] = {
                    "x": int(x1),
                    "y": int(y1),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                    "area": int((x2 - x1) * (y2 - y1)),
                    "aspect": round(float((x2 - x1) / max(1, y2 - y1)), 4),
                }
                did_merge = True
                break

        if not did_merge:
            merged.append(box)

    return merged


def raw_boxes_should_merge(first, second, image_width, image_height):
    first_center_y = first["y"] + first["height"] / 2
    second_center_y = second["y"] + second["height"] / 2

    vertical_distance = abs(first_center_y - second_center_y)

    first_x2 = first["x"] + first["width"]
    second_x2 = second["x"] + second["width"]

    horizontal_overlap = max(
        0,
        min(first_x2, second_x2) - max(first["x"], second["x"]),
    )

    smaller_width = max(1, min(first["width"], second["width"]))
    overlap_ratio = horizontal_overlap / smaller_width

    same_row = vertical_distance <= max(first["height"], second["height"]) * 0.65
    strong_x_overlap = overlap_ratio >= 0.35

    return same_row and strong_x_overlap


def filter_best_vertical_card_group(boxes, image_width, image_height):
    """
    Keeps the most likely vertical team column group.

    Raw photos may contain other red UI elements. This keeps the group with
    similar x-position, similar width, and vertical stacking.
    """
    if len(boxes) <= 1:
        return boxes

    candidates = sorted(boxes, key=lambda box: box["area"], reverse=True)[:18]
    best_group = []

    for seed in candidates:
        seed_center_x = seed["x"] + seed["width"] / 2
        seed_width = seed["width"]

        group = []

        for box in candidates:
            center_x = box["x"] + box["width"] / 2
            width_ratio = box["width"] / max(1, seed_width)
            center_drift = abs(center_x - seed_center_x) / max(1, image_width)

            if center_drift <= 0.18 and 0.45 <= width_ratio <= 1.85:
                group.append(box)

        group = sorted(group, key=lambda box: box["y"])

        if len(group) > len(best_group):
            best_group = group
        elif len(group) == len(best_group):
            old_area = sum(box["area"] for box in best_group)
            new_area = sum(box["area"] for box in group)
            if new_area > old_area:
                best_group = group

    return best_group[:EXPECTED_OPPONENT_TEAM_SIZE]


def build_team_column_box_from_card_boxes(card_boxes, image_width, image_height):
    if not card_boxes:
        return None

    sorted_boxes = sorted(card_boxes, key=lambda box: box["y"])

    x1 = min(box["x"] for box in sorted_boxes)
    y1 = min(box["y"] for box in sorted_boxes)
    x2 = max(box["x"] + box["width"] for box in sorted_boxes)
    y2 = max(box["y"] + box["height"] for box in sorted_boxes)

    column_width = x2 - x1
    column_height = y2 - y1

    x_padding = int(column_width * 0.18)
    y_padding = int(column_height * 0.08)

    return clamp_box_to_image(
        {
            "x": x1 - x_padding,
            "y": y1 - y_padding,
            "width": column_width + x_padding * 2,
            "height": column_height + y_padding * 2,
        },
        image_width,
        image_height,
    )


def detect_raw_photo_red_column_box(image):
    """
    Fallback when individual card boxes cannot be separated.
    Finds the largest red-heavy area and crops around it.
    """
    if is_empty_image(image):
        return None

    cv2, np = load_cv_dependencies()
    image_height, image_width = image.shape[:2]
    image_area = max(1, image_width * image_height)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_red_1 = cv2.inRange(hsv, np.array([0, 55, 45]), np.array([12, 255, 255]))
    lower_red_2 = cv2.inRange(hsv, np.array([145, 55, 45]), np.array([179, 255, 255]))
    red_mask = lower_red_1 | lower_red_2

    kernel = np.ones((25, 25), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    contours, _hierarchy = cv2.findContours(
        red_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    valid_boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.002:
            continue

        x, y, width, height = cv2.boundingRect(contour)

        if width <= 0 or height <= 0:
            continue

        aspect = width / max(1, height)

        # A full six-slot column is usually taller than it is wide.
        # But allow some flexibility for angled photos.
        if aspect > 1.7:
            continue

        valid_boxes.append({
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "area": width * height,
        })

    if not valid_boxes:
        return None

    best_box = max(valid_boxes, key=lambda box: box["area"])

    return expand_box_xy(
        best_box,
        x_padding=int(best_box["width"] * 0.22),
        y_padding=int(best_box["height"] * 0.10),
        image_width=image_width,
        image_height=image_height,
    )


def draw_debug_raw_team_column_overlay(image, team_column_box, raw_card_boxes):
    if is_empty_image(image):
        return image

    cv2, _np = load_cv_dependencies()

    for index, box in enumerate(raw_card_boxes, start=1):
        x1 = int(box["x"])
        y1 = int(box["y"])
        x2 = x1 + int(box["width"])
        y2 = y1 + int(box["height"])

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(
            image,
            f"card {index}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    if team_column_box:
        x1 = int(team_column_box["x"])
        y1 = int(team_column_box["y"])
        x2 = x1 + int(team_column_box["width"])
        y2 = y1 + int(team_column_box["height"])

        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 255), 5)
        cv2.putText(
            image,
            "team column",
            (x1, max(32, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.95,
            (255, 0, 255),
            3,
            cv2.LINE_AA,
        )

    return image

def merge_raw_photo_card_row_runs(runs, image_height):
    """
    Merges row runs that are probably pieces of the same red card.
    Avoids merging separate team slots.
    """
    if not runs:
        return []

    merged = []
    index = 0

    max_same_card_gap = max(4, int(image_height * 0.018))
    max_card_height = int(image_height * 0.18)

    while index < len(runs):
        start, end = runs[index]

        while index + 1 < len(runs):
            next_start, next_end = runs[index + 1]
            gap = next_start - end
            combined_height = next_end - start + 1

            if gap <= max_same_card_gap and combined_height <= max_card_height:
                end = next_end
                index += 1
            else:
                break

        merged.append([start, end])
        index += 1

    return merged


def remove_duplicate_raw_card_boxes(boxes):
    """
    Removes duplicate boxes caused by glare splitting/rejoining the same card.
    """
    if not boxes:
        return []

    boxes = sorted(boxes, key=lambda box: (box["y"], -box["area"]))
    kept = []

    for box in boxes:
        box_center_y = box["y"] + box["height"] / 2

        duplicate = False
        for kept_box in kept:
            kept_center_y = kept_box["y"] + kept_box["height"] / 2
            center_gap = abs(box_center_y - kept_center_y)

            if center_gap < max(box["height"], kept_box["height"]) * 0.55:
                duplicate = True
                break

        if not duplicate:
            kept.append(box)

    return kept

def expand_box_xy(box, x_padding, y_padding, image_width, image_height):
    return clamp_box_to_image(
        {
            "x": box["x"] - x_padding,
            "y": box["y"] - y_padding,
            "width": box["width"] + x_padding * 2,
            "height": box["height"] + y_padding * 2,
        },
        image_width,
        image_height,
    )


# Crops all guided opponent team slots from an image.
def crop_opponent_team_slots(image_path, save_debug=False):
    cv2, _np = load_cv_dependencies()
    image = read_cv_image(image_path)

    image_height, image_width = image.shape[:2]
    crops = []

    if save_debug:
        OPPONENT_DEBUG_CROP_DIR.mkdir(parents=True, exist_ok=True)

    detected_boxes = detect_opponent_card_boxes(image)
    crop_boxes = detected_boxes if len(detected_boxes) >= EXPECTED_OPPONENT_TEAM_SIZE else []
    if not crop_boxes:
        crop_boxes = [
            box_to_pixels(box, image_width, image_height)
            for box in OPPONENT_GUIDED_TEAM_BOXES
        ]

    for index, pixel_box in enumerate(crop_boxes[:EXPECTED_OPPONENT_TEAM_SIZE], start=1):
        x = pixel_box["x"]
        y = pixel_box["y"]
        width = pixel_box["width"]
        height = pixel_box["height"]
        crop = image[y:y + height, x:x + width]
        if is_empty_image(crop):
            raise ComputerVisionError(f"Opponent crop {index} was empty.")

        crop = rectify_opponent_card_crop(crop)

        debug_path = ""
        if save_debug:
            image_debug_dir = OPPONENT_DEBUG_CROP_DIR / Path(image_path).stem
            image_debug_dir.mkdir(parents=True, exist_ok=True)

            debug_path = str(image_debug_dir / f"opponent-slot-{index}.jpg")
            if not cv2.imwrite(debug_path, crop):
                raise ComputerVisionError(f"Could not write debug crop: {debug_path}")

        crops.append({
            "position": index,
            "box": pixel_box,
            "image": crop,
            "debugCropPath": debug_path,
        })

    return crops


# Straightens one red opponent card crop into a consistent rectangle.
def rectify_opponent_card_crop(card_crop):
    if is_empty_image(card_crop):
        return card_crop

    cv2, np = load_cv_dependencies()
    hsv_crop = cv2.cvtColor(card_crop, cv2.COLOR_BGR2HSV)
    lower_red = cv2.inRange(hsv_crop, np.array([0, 70, 60]), np.array([10, 255, 255]))
    upper_red = cv2.inRange(hsv_crop, np.array([150, 70, 60]), np.array([179, 255, 255]))
    red_mask = lower_red | upper_red
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _hierarchy = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return card_crop

    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < card_crop.shape[0] * card_crop.shape[1] * 0.20:
        return card_crop

    rect = cv2.minAreaRect(largest_contour)
    if calculate_rect_skew_degrees(rect) < MIN_CARD_RECTIFICATION_SKEW:
        return card_crop

    box_points = cv2.boxPoints(rect)
    ordered_points = order_perspective_points(box_points)
    target_points = np.array(
        [
            [0, 0],
            [RECTIFIED_CARD_WIDTH - 1, 0],
            [RECTIFIED_CARD_WIDTH - 1, RECTIFIED_CARD_HEIGHT - 1],
            [0, RECTIFIED_CARD_HEIGHT - 1],
        ],
        dtype="float32",
    )
    transform = cv2.getPerspectiveTransform(ordered_points, target_points)
    return cv2.warpPerspective(card_crop, transform, (RECTIFIED_CARD_WIDTH, RECTIFIED_CARD_HEIGHT))


# Returns how far a rotated rectangle is from horizontal alignment.
def calculate_rect_skew_degrees(rect):
    angle = abs(float(rect[2]))
    if angle > 45:
        angle = abs(90 - angle)

    return angle


# Orders four points as top-left, top-right, bottom-right, and bottom-left.
def order_perspective_points(points):
    _cv2, np = load_cv_dependencies()
    points = np.asarray(points, dtype="float32")
    ordered = np.zeros((4, 2), dtype="float32")

    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(4)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


# Scores whether an opponent photo is good enough for reliable detection.
def assess_opponent_image_quality(image_path):
    image = read_cv_image(image_path)
    return assess_opponent_image_quality_from_image(image)


# Builds image quality metrics and user-facing capture guidance.
def assess_opponent_image_quality_from_image(image):
    if is_empty_image(image):
        return build_quality_result(
            image_width=0,
            image_height=0,
            card_boxes=[],
            sharpness_score=0.0,
            average_brightness=0.0,
            overexposed_ratio=0.0,
            underexposed_ratio=0.0,
        )

    cv2, _np = load_cv_dependencies()
    image_height, image_width = image.shape[:2]
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value_channel = hsv_image[:, :, 2]

    card_boxes = detect_opponent_card_boxes(image)
    sharpness_score = float(cv2.Laplacian(gray_image, cv2.CV_64F).var())
    average_brightness = float(value_channel.mean())
    overexposed_ratio = float((value_channel > 245).sum() / value_channel.size)
    underexposed_ratio = float((value_channel < 35).sum() / value_channel.size)

    return build_quality_result(
        image_width=image_width,
        image_height=image_height,
        card_boxes=card_boxes,
        sharpness_score=sharpness_score,
        average_brightness=average_brightness,
        overexposed_ratio=overexposed_ratio,
        underexposed_ratio=underexposed_ratio,
    )


# Converts quality metrics into pass or retake guidance.
def build_quality_result(
    image_width,
    image_height,
    card_boxes,
    sharpness_score,
    average_brightness,
    overexposed_ratio,
    underexposed_ratio,
):
    image_area = max(1, image_width * image_height)
    card_area_ratio = sum(box["width"] * box["height"] for box in card_boxes) / image_area
    center_drift_ratio = calculate_card_center_drift_ratio(card_boxes, image_width)
    spacing_variation_ratio = calculate_card_spacing_variation_ratio(card_boxes)
    issues = []
    warnings = []

    if len(card_boxes) != EXPECTED_OPPONENT_TEAM_SIZE:
        issues.append(
            f"Found {len(card_boxes)} of {EXPECTED_OPPONENT_TEAM_SIZE} opponent cards. Line up all six red cards in the guide."
        )

    if sharpness_score < MIN_ACCEPTABLE_SHARPNESS:
        issues.append("Image is too blurry. Hold the camera still and retake the photo.")

    if card_area_ratio < MIN_ACCEPTABLE_CARD_AREA_RATIO:
        issues.append("Opponent cards are too small. Move closer so the red cards fill more of the guide.")

    if center_drift_ratio > MAX_ACCEPTABLE_CARD_CENTER_DRIFT_RATIO:
        issues.append("Team list is too angled. Hold the phone more square to the screen.")

    if spacing_variation_ratio > MAX_ACCEPTABLE_SPACING_VARIATION_RATIO:
        issues.append("Card spacing looks uneven. Keep the full team list inside the guide boxes.")

    if overexposed_ratio > MAX_ACCEPTABLE_OVEREXPOSED_RATIO:
        issues.append("Image has too much glare. Reduce reflections or change the camera angle.")

    if underexposed_ratio > WARNING_UNDEREXPOSED_RATIO:
        warnings.append("Image is dark. Detection may work better with more light or a brighter screen.")

    quality_level = "bad" if issues else "warning" if warnings else "good"
    return {
        "canAnalyze": len(issues) == 0,
        "qualityLevel": quality_level,
        "issues": issues,
        "warnings": warnings,
        "metrics": {
            "imageWidth": int(image_width),
            "imageHeight": int(image_height),
            "detectedCardCount": len(card_boxes),
            "sharpnessScore": round(sharpness_score, 2),
            "averageBrightness": round(average_brightness, 2),
            "overexposedRatio": round(overexposed_ratio, 4),
            "underexposedRatio": round(underexposed_ratio, 4),
            "cardAreaRatio": round(card_area_ratio, 4),
            "centerDriftRatio": round(center_drift_ratio, 4),
            "spacingVariationRatio": round(spacing_variation_ratio, 4),
        },
    }


# Measures how far card centers drift horizontally across the team list.
def calculate_card_center_drift_ratio(card_boxes, image_width):
    if len(card_boxes) < 2 or image_width <= 0:
        return 0.0

    centers = [box["x"] + box["width"] / 2 for box in card_boxes]
    return (max(centers) - min(centers)) / image_width


# Measures whether the six cards are spaced like one consistent team list.
def calculate_card_spacing_variation_ratio(card_boxes):
    if len(card_boxes) < 3:
        return 0.0

    sorted_boxes = sorted(card_boxes, key=lambda box: box["y"])
    centers = [box["y"] + box["height"] / 2 for box in sorted_boxes]
    gaps = [centers[index + 1] - centers[index] for index in range(len(centers) - 1)]
    average_gap = sum(gaps) / len(gaps)
    if average_gap <= 0:
        return 0.0

    return max(abs(gap - average_gap) for gap in gaps) / average_gap


# Finds opponent card boxes by looking for the red opponent team panels.
def detect_opponent_card_boxes(image):
    if is_empty_image(image):
        return []

    cv2, np = load_cv_dependencies()
    image_height, image_width = image.shape[:2]
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = cv2.inRange(hsv_image, np.array([0, 80, 80]), np.array([8, 255, 255]))
    upper_red = cv2.inRange(hsv_image, np.array([150, 80, 80]), np.array([179, 255, 255]))
    red_mask = lower_red | upper_red

    search_left = int(image_width * 0.20)
    search_right = int(image_width * 0.94)
    search_mask = red_mask[:, search_left:search_right]
    row_counts = (search_mask > 0).sum(axis=1)
    row_threshold = int((search_right - search_left) * 0.20)
    raw_runs = build_mask_runs(row_counts, row_threshold, min_length=int(image_height * 0.015))
    merged_runs = [
        run
        for run in raw_runs
        if not (run[0] < image_height * 0.14 and run[1] - run[0] + 1 < image_height * 0.055)
        and not (run[0] > image_height * 0.92)
    ]
    merged_runs = merge_partial_card_runs(
        merged_runs,
        min_height=int(image_height * 0.08),
        max_gap=int(image_height * 0.016),
        max_height=int(image_height * 0.18),
    )

    boxes = []
    for run_start, run_end in merged_runs:
        height = run_end - run_start + 1
        if height < image_height * 0.07:
            continue

        run_mask = search_mask[run_start:run_end + 1]
        points = np.argwhere(run_mask > 0)
        if points.size == 0:
            continue

        _rows, cols = points[:, 0], points[:, 1]
        column_counts = (run_mask > 0).sum(axis=0)
        strong_column_threshold = max(4, int(height * 0.22))
        strong_cols = np.where(column_counts >= strong_column_threshold)[0]
        edge_cols = strong_cols if strong_cols.size else cols
        x_min = max(0, search_left + int(edge_cols.min()) - 20)
        x_max = min(image_width - 1, search_left + int(edge_cols.max()) + 20)

        boxes.append({
            "x": int(x_min),
            "y": int(run_start),
            "width": int(x_max - x_min + 1),
            "height": int(height),
        })

    boxes = split_merged_opponent_card_boxes(
        sorted(boxes, key=lambda box: box["y"]),
        image_height=image_height,
    )
    return boxes[:EXPECTED_OPPONENT_TEAM_SIZE]


def split_merged_opponent_card_boxes(boxes, image_height):
    if len(boxes) >= EXPECTED_OPPONENT_TEAM_SIZE or len(boxes) < 2:
        return boxes

    normal_heights = sorted(box["height"] for box in boxes)
    median_height = normal_heights[len(normal_heights) // 2]
    if median_height <= 0:
        return boxes

    repaired_boxes = []
    for box in boxes:
        if box["height"] < median_height * 1.65:
            repaired_boxes.append(box)
            continue

        split_height = int(round(median_height))
        gap = max(4, int(round(split_height * 0.08)))
        first = {
            **box,
            "height": min(split_height, box["height"]),
        }
        second_y = box["y"] + split_height + gap
        second = {
            **box,
            "y": second_y,
            "height": min(split_height, max(1, box["y"] + box["height"] - second_y)),
        }
        if second["y"] + second["height"] <= image_height:
            repaired_boxes.extend([first, second])
        else:
            repaired_boxes.append(box)

    return sorted(repaired_boxes, key=lambda box: box["y"])


# Builds row or column runs where a mask projection exceeds a threshold.
def build_mask_runs(values, threshold, min_length=1):
    runs = []
    run_start = None

    for index, value in enumerate(values):
        if value > threshold and run_start is None:
            run_start = index
        if (value <= threshold or index == len(values) - 1) and run_start is not None:
            run_end = index - 1 if value <= threshold else index
            if run_end - run_start + 1 >= min_length:
                runs.append([run_start, run_end])
            run_start = None

    return runs


# Merges split pieces of one red card without merging neighboring full cards.
def merge_partial_card_runs(runs, min_height, max_gap, max_height):
    merged = []
    index = 0
    while index < len(runs):
        run_start, run_end = runs[index]
        if index + 1 < len(runs):
            next_start, next_end = runs[index + 1]
            height = run_end - run_start + 1
            next_height = next_end - next_start + 1
            combined_height = next_end - run_start + 1
            gap = next_start - run_end
            if (
                gap <= max_gap
                and combined_height <= max_height
                and (height < min_height or next_height < min_height)
            ):
                merged.append([run_start, next_end])
                index += 2
                continue

        merged.append([run_start, run_end])
        index += 1

    return merged


# Loads reference images used for template matching.
def load_reference_images(
    reference_dir=REFERENCE_IMAGE_DIR,
    metadata_path=REFERENCE_METADATA_PATH,
    extra_reference_dir=EXTRA_REFERENCE_IMAGE_DIR,
):
    cv2, _np = load_cv_dependencies()
    metadata_by_filename = load_reference_metadata(metadata_path)
    references = []

    for current_reference_dir in (reference_dir, SHINY_REFERENCE_IMAGE_DIR, extra_reference_dir):
        if not current_reference_dir.exists():
            continue

        for image_path in sorted(current_reference_dir.rglob("*")):
            if image_path.suffix.lower() not in SUPPORTED_REFERENCE_EXTENSIONS:
                continue

            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                continue

            metadata = metadata_by_filename.get(image_path.name, {})
            if not metadata and Path(current_reference_dir) == Path(extra_reference_dir):
                metadata = infer_extra_reference_metadata(image_path, metadata_by_filename)
            prepared_image = prepare_reference_image(image)
            grayscale_match_image = preprocess_for_matching(prepared_image)
            color_match_image = preprocess_color_for_matching(prepared_image)
            feature_keypoints, feature_descriptors = build_feature_descriptors(prepared_image)
            references.append({
                "name": metadata.get("display_name") or image_path.stem.replace("_", " ").replace("-", " ").title(),
                "species": metadata.get("species_display_name") or metadata.get("display_name") or image_path.stem,
                "form": metadata.get("form_display_name") or metadata.get("display_name") or image_path.stem,
                "isShiny": bool(metadata.get("is_shiny", False)),
                "types": metadata.get("types", []),
                "path": str(image_path),
                "image": prepared_image,
                "matchImage": grayscale_match_image,
                "colorMatchImage": color_match_image,
                "grayscaleTemplates": build_scaled_templates(grayscale_match_image),
                "colorTemplates": build_scaled_templates(color_match_image, normalize=False),
                "partialGrayscaleTemplates": build_scaled_templates(
                    grayscale_match_image,
                    scales=PARTIAL_MATCH_SCALES,
                ),
                "partialColorTemplates": build_scaled_templates(
                    color_match_image,
                    normalize=False,
                    scales=PARTIAL_MATCH_SCALES,
                ),
                "featureKeypointCount": len(feature_keypoints),
                "featureDescriptors": feature_descriptors,
            })

    logger.info("Loaded %s opponent reference images.", len(references))
    return references


def infer_extra_reference_metadata(image_path, metadata_by_filename):
    normalized_stem = normalize_reference_name(image_path.stem)
    display_matches = []
    species_matches = []

    for record in metadata_by_filename.values():
        display_name = normalize_reference_name(record.get("display_name", ""))
        form_name = normalize_reference_name(record.get("form_display_name", ""))
        species_name = normalize_reference_name(record.get("species_display_name", ""))

        for normalized_name in (display_name, form_name):
            if normalized_name and normalized_name in normalized_stem:
                display_matches.append((len(normalized_name), record))

        if species_name and species_name in normalized_stem:
            species_matches.append((len(species_name), record))

    if display_matches:
        return max(display_matches, key=lambda match: match[0])[1]

    if species_matches:
        return min(species_matches, key=lambda match: match[0])[1]

    return {}


def normalize_reference_name(name):
    return "".join(character for character in str(name).lower() if character.isalnum())


SUPPORTED_TYPE_REFERENCE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

# Camera crops that are visibly mislabeled, too washed out, or dominated by
# background/glare. Keep them on disk for audit, but do not use them as type
# references.
BLOCKED_TYPE_REFERENCE_FILENAMES = {
    "normal__camera-1.jpg",
    "normal__camera-2.jpg",
    "normal__camera-4.jpg",
    "steel__camera-5.jpg",
    "steel__camera-7.jpg",
}


def infer_type_name_from_icon_filename(image_path):
    """
    Allows:
    fire.png -> fire
    fire__camera_1.jpg -> fire
    water__glare_2.jpg -> water
    """
    stem = image_path.stem.lower().strip()

    if "__" in stem:
        return stem.split("__", 1)[0]

    return stem


def is_blocked_type_reference_path(image_path):
    return Path(str(image_path)).name.lower() in BLOCKED_TYPE_REFERENCE_FILENAMES


# Loads reference type icons for template and embedding matching.
def load_type_icon_references(type_reference_dir=TYPE_REFERENCE_IMAGE_DIR):
    cv2, _np = load_cv_dependencies()
    type_reference_dir = Path(type_reference_dir)
    references = []

    if not type_reference_dir.exists():
        logger.warning("Type reference icon directory was not found: %s", type_reference_dir)
        return references

    for image_path in sorted(type_reference_dir.iterdir()):
        if image_path.suffix.lower() not in SUPPORTED_TYPE_REFERENCE_EXTENSIONS:
            continue

        if is_blocked_type_reference_path(image_path):
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        icon_type = infer_type_name_from_icon_filename(image_path)

        references.append({
            "type": icon_type,
            "image": image,
            "whiteMask": extract_white_symbol_mask(image),
            "colorSignature": extract_type_color_signature(image),
            "path": str(image_path),
        })

    logger.info("Loaded %s type reference icons.", len(references))
    return references


@lru_cache(maxsize=8)
def load_type_combo_references(
    type_combo_reference_dir=TYPE_COMBO_REFERENCE_IMAGE_DIR,
    metadata_path=TYPE_COMBO_REFERENCE_METADATA_PATH,
):
    cv2, _np = load_cv_dependencies()
    type_combo_reference_dir = Path(type_combo_reference_dir)
    metadata_path = Path(metadata_path)
    references = []
    metadata_by_path = {}

    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata_by_path = {
                Path(record.get("imagePath", "")).name: record
                for record in metadata.get("typeCombos", [])
            }
        except (OSError, json.JSONDecodeError) as error:
            logger.warning("Type combo reference metadata could not be loaded: %s", error)

    if not type_combo_reference_dir.exists():
        logger.warning("Type combo reference directory was not found: %s", type_combo_reference_dir)
        return references

    for image_path in sorted(type_combo_reference_dir.iterdir()):
        if image_path.suffix.lower() not in SUPPORTED_TYPE_REFERENCE_EXTENSIONS:
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        record = metadata_by_path.get(image_path.name, {})
        types = record.get("types") or parse_type_combo_reference_types(image_path.stem)

        references.append({
            "types": [str(type_name).lower() for type_name in types],
            "typeKey": record.get("typeKey") or image_path.stem,
            "image": image,
            "path": str(image_path),
        })

    logger.info("Loaded %s type combo references.", len(references))
    return references


def parse_type_combo_reference_types(stem):
    type_slug = str(stem or "").split("__", 1)[0].replace("-", "_")
    aliases = {
        "posion": "poison",
    }

    return [
        aliases.get(type_name, type_name)
        for type_name in type_slug.split("_")
        if type_name
    ]


# Crops empty background from a reference sprite.
def prepare_reference_image(image):
    cv2, np = load_cv_dependencies()
    if len(image.shape) == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        points = np.argwhere(alpha > 0)
        if points.size == 0:
            return composite_transparent_sprite(image)

        y_min, x_min = points.min(axis=0)
        y_max, x_max = points.max(axis=0)
        cropped = image[y_min:y_max + 1, x_min:x_max + 1]
        return composite_transparent_sprite(cropped)

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = grayscale < 245
    points = np.argwhere(mask)
    if points.size == 0:
        return image

    y_min, x_min = points.min(axis=0)
    y_max, x_max = points.max(axis=0)
    return image[y_min:y_max + 1, x_min:x_max + 1]


# Places a transparent reference sprite on a card-colored background.
def composite_transparent_sprite(image):
    cv2, np = load_cv_dependencies()
    if len(image.shape) != 3 or image.shape[2] != 4:
        return image

    alpha = image[:, :, 3].astype("float32") / 255.0
    sprite = image[:, :, :3].astype("float32")
    background = np.full(sprite.shape, REFERENCE_CARD_BACKGROUND_BGR, dtype="float32")
    composited = sprite * alpha[:, :, None] + background * (1.0 - alpha[:, :, None])
    return composited.astype("uint8")


# Crops the opponent Pokemon sprite area from a guided opponent team slot.
def extract_opponent_pokemon_region(slot_image):
    if is_empty_image(slot_image):
        return slot_image

    sprite_object = detect_pokemon_sprite_object(slot_image)
    if sprite_object and not is_empty_image(sprite_object.get("image")):
        return sprite_object["image"]

    height, width = slot_image.shape[:2]
    x = int(width * 0.02)
    y = 0
    crop_width = int(width * 0.58)
    crop_height = int(height * 0.96)
    return slot_image[y:y + crop_height, x:x + crop_width]


def extract_fixed_opponent_pokemon_region(slot_image):
    if is_empty_image(slot_image):
        return slot_image

    height, width = slot_image.shape[:2]
    x = int(width * 0.02)
    y = 0
    crop_width = int(width * 0.58)
    crop_height = int(height * 0.96)
    return slot_image[y:y + crop_height, x:x + crop_width]


def detect_slot_objects(slot_image):
    object_layer = detect_slot_object_layer(slot_image)
    return detect_slot_objects_from_layer(object_layer)


def detect_slot_objects_from_layer(object_layer):
    return [
        detected_object
        for detected_object in (
            object_layer.get("pokemon_sprite"),
            object_layer.get("type_icon_1"),
            object_layer.get("type_icon_2"),
        )
        if detected_object
    ]


def detect_slot_object_layer(slot_image):
    if is_empty_image(slot_image):
        return {
            "pokemon_sprite": None,
            "type_icon_1": None,
            "type_icon_2": None,
            "candidates": {
                "pokemon_sprite": [],
                "type_icons": [],
            },
        }

    yolo_object_layer = detect_yolo_slot_object_layer(slot_image)
    if yolo_object_layer:
        return yolo_object_layer

    pokemon_candidates = generate_pokemon_sprite_candidates(slot_image)
    pokemon_object = select_pokemon_sprite_object(pokemon_candidates, slot_image)

    type_icon_candidates = generate_type_icon_crop_candidates(slot_image)
    type_icon_objects = select_type_icon_objects(type_icon_candidates, slot_image)

    return {
        "pokemon_sprite": pokemon_object,
        "type_icon_1": type_icon_objects[0] if len(type_icon_objects) >= 1 else None,
        "type_icon_2": type_icon_objects[1] if len(type_icon_objects) >= 2 else None,
        "candidates": {
            "pokemon_sprite": pokemon_candidates,
            "type_icons": type_icon_candidates,
        },
    }


def load_yolo_slot_object_model():
    global _YOLO_SLOT_OBJECT_MODEL
    global _YOLO_SLOT_OBJECT_MODEL_LOAD_FAILED

    if _YOLO_SLOT_OBJECT_MODEL is not None:
        return _YOLO_SLOT_OBJECT_MODEL

    if _YOLO_SLOT_OBJECT_MODEL_LOAD_FAILED or not YOLO_SLOT_OBJECT_MODEL_PATH.exists():
        return None

    try:
        os.environ.setdefault("YOLO_CONFIG_DIR", str(BACKEND_DIR / "Ultralytics"))
        from ultralytics import YOLO

        _YOLO_SLOT_OBJECT_MODEL = YOLO(str(YOLO_SLOT_OBJECT_MODEL_PATH))
        return _YOLO_SLOT_OBJECT_MODEL
    except Exception as error:
        _YOLO_SLOT_OBJECT_MODEL_LOAD_FAILED = True
        logger.warning("YOLO slot object model could not be loaded: %s", error)
        return None


def get_yolo_inference_device():
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def detect_yolo_slot_object_layer(slot_image):
    if is_empty_image(slot_image):
        return None

    model = load_yolo_slot_object_model()
    if model is None:
        return None

    try:
        results = model.predict(
            source=slot_image,
            conf=YOLO_SLOT_OBJECT_CONFIDENCE,
            imgsz=YOLO_SLOT_OBJECT_IMAGE_SIZE,
            device=get_yolo_inference_device(),
            verbose=False,
        )
    except Exception as error:
        logger.warning("YOLO slot object detection failed: %s", error)
        return None

    detections = yolo_slot_object_detections_from_results(results, slot_image)
    if not detections:
        return None

    pokemon_candidates = [
        detection for detection in detections
        if detection.get("label") == "pokemon_sprite"
    ]
    type_icon_candidates = [
        detection for detection in detections
        if detection.get("label") == "type_icon"
    ]

    pokemon_object = select_pokemon_sprite_object(pokemon_candidates, slot_image)
    if pokemon_object is None:
        pokemon_candidates = generate_pokemon_sprite_candidates(slot_image)
        pokemon_object = select_pokemon_sprite_object(pokemon_candidates, slot_image)

    type_icon_objects = select_type_icon_objects(type_icon_candidates, slot_image)
    if len(type_icon_objects) < 2:
        fallback_type_icon_candidates = generate_type_icon_crop_candidates(slot_image)
        merged_type_icon_candidates = type_icon_candidates + [
            candidate for candidate in fallback_type_icon_candidates
            if not any(
                type_icon_boxes_overlap(candidate, old)
                and effective_type_icon_crop_quality(old) >= 0.48
                for old in type_icon_candidates
            )
        ]
        type_icon_candidates = merged_type_icon_candidates
        type_icon_objects = select_type_icon_objects(type_icon_candidates, slot_image)

    if not pokemon_object and len(type_icon_objects) < 1:
        return None

    return {
        "pokemon_sprite": pokemon_object,
        "type_icon_1": type_icon_objects[0] if len(type_icon_objects) >= 1 else None,
        "type_icon_2": type_icon_objects[1] if len(type_icon_objects) >= 2 else None,
        "candidates": {
            "pokemon_sprite": pokemon_candidates,
            "type_icons": type_icon_candidates,
        },
        "source": "yolo_slot_object_detector",
    }


def yolo_slot_object_detections_from_results(results, slot_image):
    detections = []
    if not results:
        return detections

    result = results[0]
    names = getattr(result, "names", {}) or {}
    image_height, image_width = slot_image.shape[:2]

    for yolo_box in getattr(result, "boxes", []) or []:
        class_id = int(yolo_box.cls[0])
        class_name = names.get(class_id, str(class_id))
        confidence = float(yolo_box.conf[0])
        x1, y1, x2, y2 = [
            int(round(value))
            for value in yolo_box.xyxy[0].tolist()
        ]
        box = clamp_box_to_image(
            {
                "x": x1,
                "y": y1,
                "width": max(1, x2 - x1),
                "height": max(1, y2 - y1),
            },
            image_width,
            image_height,
        )

        if class_name == "pokemon_sprite":
            detections.append(
                make_pokemon_sprite_candidate(
                    slot_image,
                    box,
                    "yolo_slot_object_detector",
                    confidence,
                )
            )
            continue

        if class_name == "type_icon":
            crop = crop_image_box(slot_image, box)
            detections.append({
                "label": "type_icon",
                "role": "type_icon",
                "confidence": round(confidence, 4),
                "box": box,
                "rawBox": box,
                "x": box["x"],
                "y": box["y"],
                "width": box["width"],
                "height": box["height"],
                "area": box["width"] * box["height"],
                "image": crop,
                "source": "yolo_slot_object_detector",
                "cropSource": "yolo_slot_object_detector",
                "hasSymbol": has_type_icon_symbol(crop),
                "cropQuality": round(score_type_icon_crop_quality(crop), 4),
            })

    return detections


def select_pokemon_sprite_object(candidates, slot_image):
    """
    Selects the best Pokemon sprite crop.

    Important change:
    - prefer tight object/foreground proposals
    - avoid wide fallback crops that include most of the card/type area
    - only use wide fallback crops when nothing better exists
    """
    if is_empty_image(slot_image):
        return None

    slot_height, slot_width = slot_image.shape[:2]

    def is_focused_candidate(candidate):
        if is_empty_image(candidate.get("image")):
            return False
        if not is_reasonable_sprite_crop(candidate.get("image"), slot_image):
            return False

        box = candidate.get("box") or {}
        width_ratio = box.get("width", 0) / max(1, slot_width)
        height_ratio = box.get("height", 0) / max(1, slot_height)

        # The previous issue: fallback crops could span most of the card.
        # Keep those as emergency fallback only.
        if width_ratio > 0.52:
            return False
        if height_ratio > 0.93:
            return False

        return candidate.get("cropQuality", 0) >= 0.38

    focused_candidates = [candidate for candidate in candidates if is_focused_candidate(candidate)]

    source_priority = {
        "foreground_object_proposal": 0.16,
        "color_object_proposal": 0.13,
        "relaxed_color_object_proposal": 0.10,
        "fixed_sprite_fallback": -0.08,
        "shifted_fixed_sprite_fallback": -0.10,
        "expanded_fixed_sprite_fallback": -0.16,
    }

    if focused_candidates:
        best_candidate = max(
            focused_candidates,
            key=lambda candidate: (
                float(candidate.get("cropQuality", 0) or 0)
                + float(candidate.get("confidence", 0) or 0) * 0.20
                + source_priority.get(candidate.get("source", ""), 0.0),
                -((candidate.get("box") or {}).get("width", 0) / max(1, slot_width)),
            ),
        )
    elif candidates:
        # Emergency fallback, but still prefer the least-wide usable crop.
        usable_candidates = [
            candidate
            for candidate in candidates
            if not is_empty_image(candidate.get("image"))
        ]
        if not usable_candidates:
            return None

        best_candidate = max(
            usable_candidates,
            key=lambda candidate: (
                float(candidate.get("cropQuality", 0) or 0)
                - ((candidate.get("box") or {}).get("width", 0) / max(1, slot_width)) * 0.20,
                float(candidate.get("confidence", 0) or 0),
            ),
        )
    else:
        return None

    return {
        **best_candidate,
        "label": "pokemon_sprite",
        "role": "pokemon_sprite",
    }

def normalize_type_box(box, image_width, image_height, padding_ratio=0.08):
    """
    Converts a detected type icon box into a conservative padded square crop.

    Keep this conservative. Large padding makes two adjacent type icons overlap.
    """
    x = int(round(box.get("x", 0)))
    y = int(round(box.get("y", 0)))
    width = int(round(box.get("width", 1)))
    height = int(round(box.get("height", 1)))

    center_x = x + width / 2
    center_y = y + height / 2

    size = max(width, height)
    size = int(round(size * (1 + padding_ratio)))

    new_x = int(round(center_x - size / 2))
    new_y = int(round(center_y - size / 2))

    return clamp_box_to_image(
        {
            "x": new_x,
            "y": new_y,
            "width": size,
            "height": size,
        },
        image_width,
        image_height,
    )


def build_type_icon_object_from_box(source_object, slot_image, box, index):
    """
    Builds a final type icon object from a clean square box.
    """
    crop = crop_image_box(slot_image, box)

    crop_quality = (
        score_type_icon_crop_quality(crop)
        if not is_empty_image(crop)
        else float(source_object.get("cropQuality", 0) or 0)
    )

    has_symbol = (
        has_type_icon_symbol(crop)
        if not is_empty_image(crop)
        else bool(source_object.get("hasSymbol"))
    )

    return {
        **source_object,
        "label": f"type_icon_{index}",
        "role": f"type_icon_{index}",
        "index": index,
        "box": box,
        "x": box["x"],
        "y": box["y"],
        "width": box["width"],
        "height": box["height"],
        "area": box["width"] * box["height"],
        "image": crop,
        "hasSymbol": bool(has_symbol),
        "cropQuality": round(float(crop_quality or 0), 4),
    }


def normalize_type_icon_object(type_object, slot_image, padding_ratio=0.08):
    """
    Re-crops a selected type icon using a conservative padded square box.
    """
    if not type_object or is_empty_image(slot_image):
        return type_object

    image_height, image_width = slot_image.shape[:2]

    original_box = type_object.get("box") or {
        "x": type_object.get("x", 0),
        "y": type_object.get("y", 0),
        "width": type_object.get("width", 1),
        "height": type_object.get("height", 1),
    }

    normalized_box = normalize_type_box(
        original_box,
        image_width,
        image_height,
        padding_ratio=padding_ratio,
    )

    return build_type_icon_object_from_box(
        {
            **type_object,
            "rawBox": original_box,
        },
        slot_image,
        normalized_box,
        type_object.get("index", 1),
    )


def type_icon_pair_needs_separation(first_object, second_object, slot_image):
    """
    Detects cases where the two selected type icon crops overlap or one box
    swallowed part of the other.
    """
    if not first_object or not second_object or is_empty_image(slot_image):
        return False

    first_box = first_object.get("box") or first_object
    second_box = second_object.get("box") or second_object

    if type_icon_boxes_overlap(first_object, second_object):
        return True

    first_x2 = first_box["x"] + first_box["width"]
    second_x1 = second_box["x"]
    second_center_x = second_box["x"] + second_box["width"] / 2

    if first_x2 > second_center_x:
        return True

    if second_x1 - first_x2 < 2:
        return True

    first_area = max(1, first_box["width"] * first_box["height"])
    second_area = max(1, second_box["width"] * second_box["height"])

    return max(first_area, second_area) / min(first_area, second_area) > 1.55


def separate_type_icon_pair(first_object, second_object, slot_image):
    """
    Rebuilds two selected type icons as clean left/right square crops.

    This keeps object detection involved because the cluster is built from
    detected candidates, but the final boxes are forced to be separated.
    """
    if not first_object or not second_object or is_empty_image(slot_image):
        return first_object, second_object

    image_height, image_width = slot_image.shape[:2]

    first_box = first_object.get("rawBox") or first_object.get("box") or {
        "x": first_object.get("x", 0),
        "y": first_object.get("y", 0),
        "width": first_object.get("width", 1),
        "height": first_object.get("height", 1),
    }

    second_box = second_object.get("rawBox") or second_object.get("box") or {
        "x": second_object.get("x", 0),
        "y": second_object.get("y", 0),
        "width": second_object.get("width", 1),
        "height": second_object.get("height", 1),
    }

    boxes_with_objects = sorted(
        [(first_box, first_object), (second_box, second_object)],
        key=lambda pair: pair[0]["x"] + pair[0]["width"] / 2,
    )

    left_box, left_object = boxes_with_objects[0]
    right_box, right_object = boxes_with_objects[1]

    cluster_box = combine_boxes([left_box, right_box], image_width, image_height)
    split_objects = split_type_icon_cluster_box(slot_image, cluster_box)

    if len(split_objects) >= 2:
        return split_objects[0], split_objects[1]

    # Fallback: build two boxes anchored from the right object.
    cluster_x = cluster_box["x"]
    cluster_y = cluster_box["y"]
    cluster_width = cluster_box["width"]
    cluster_height = cluster_box["height"]

    icon_size = int(round(cluster_height * 1.05))
    icon_size = max(int(image_height * 0.28), min(icon_size, int(image_height * 0.50)))

    gap = max(2, int(round(icon_size * 0.08)))
    center_y = cluster_y + cluster_height / 2

    right_center_x = right_box["x"] + right_box["width"] / 2
    left_center_x = right_center_x - icon_size - gap

    if left_center_x - icon_size / 2 < 0:
        left_center_x = cluster_x + cluster_width * 0.30
        right_center_x = cluster_x + cluster_width * 0.70

    left_clean_box = clamp_box_to_image(
        {
            "x": int(round(left_center_x - icon_size / 2)),
            "y": int(round(center_y - icon_size / 2)),
            "width": icon_size,
            "height": icon_size,
        },
        image_width,
        image_height,
    )

    right_clean_box = clamp_box_to_image(
        {
            "x": int(round(right_center_x - icon_size / 2)),
            "y": int(round(center_y - icon_size / 2)),
            "width": icon_size,
            "height": icon_size,
        },
        image_width,
        image_height,
    )

    return (
        build_type_icon_object_from_box(
            {
                **left_object,
                "rawBox": left_box,
                "source": "type_icon_pair_separated",
            },
            slot_image,
            left_clean_box,
            1,
        ),
        build_type_icon_object_from_box(
            {
                **right_object,
                "rawBox": right_box,
                "source": "type_icon_pair_separated",
            },
            slot_image,
            right_clean_box,
            2,
        ),
    )


def select_type_icon_objects(candidates, slot_image):
    if is_empty_image(slot_image):
        return []

    _height, slot_width = slot_image.shape[:2]

    accepted_candidates = [
        candidate
        for candidate in candidates
        if candidate.get("hasSymbol")
        and is_usable_type_icon_candidate(candidate)
        and not is_empty_image(candidate.get("image"))
    ]

    if not accepted_candidates:
        accepted_candidates = [
            candidate
            for candidate in candidates
            if not is_empty_image(candidate.get("image"))
        ]

    selected = []

    role_specs = [
        ("type_icon_1", 1, 0.675, 0.50, 0.78),
        ("type_icon_2", 2, 0.825, 0.66, 0.96),
    ]

    for _role, _index, target_center_ratio, min_center_ratio, max_center_ratio in role_specs:
        role_candidates = [
            candidate
            for candidate in accepted_candidates
            if type_icon_candidate_in_role_band(
                candidate,
                slot_width,
                min_center_ratio,
                max_center_ratio,
            )
            and not any(type_icon_boxes_overlap(candidate, old) for old in selected)
        ]

        if not role_candidates:
            role_candidates = [
                candidate
                for candidate in accepted_candidates
                if not any(type_icon_boxes_overlap(candidate, old) for old in selected)
            ]

        if not role_candidates:
            continue

        selected.append(
            max(
                role_candidates,
                key=lambda candidate: score_type_icon_role_candidate(
                    candidate,
                    slot_width,
                    target_center_ratio,
                ),
            )
        )

    selected.sort(key=lambda icon_crop: icon_crop.get("x", 0))

    objects = []

    for index, icon_crop in enumerate(selected[:2], start=1):
        original_box = icon_crop.get("box") or {
            "x": icon_crop.get("x", 0),
            "y": icon_crop.get("y", 0),
            "width": icon_crop.get("width", 1),
            "height": icon_crop.get("height", 1),
        }

        raw_object = {
            "label": f"type_icon_{index}",
            "role": f"type_icon_{index}",
            "index": index,
            "confidence": icon_crop.get("confidence", 0.0),
            "box": original_box,
            "rawBox": original_box,
            "image": icon_crop.get("image"),
            "hasSymbol": bool(icon_crop.get("hasSymbol")),
            "area": icon_crop.get(
                "area",
                original_box.get("width", 0) * original_box.get("height", 0),
            ),
            "source": icon_crop.get("cropSource", icon_crop.get("source", "")),
            "cropQuality": icon_crop.get("cropQuality", 0.0),
        }

        objects.append(
            normalize_type_icon_object(
                raw_object,
                slot_image,
                padding_ratio=0.08,
            )
        )

    if len(objects) >= 2 and type_icon_pair_needs_separation(objects[0], objects[1], slot_image):
        objects[0], objects[1] = separate_type_icon_pair(
            objects[0],
            objects[1],
            slot_image,
        )

    return objects


def type_icon_candidate_in_role_band(candidate, slot_width, min_center_ratio, max_center_ratio):
    if slot_width <= 0:
        return False

    center_ratio = (
        candidate.get("x", 0) + (candidate.get("width", 0) / 2)
    ) / slot_width

    return min_center_ratio <= center_ratio <= max_center_ratio


def score_type_icon_role_candidate(candidate, slot_width, target_center_ratio):
    if slot_width <= 0:
        position_score = 0.0
    else:
        center_ratio = (
            candidate.get("x", 0) + (candidate.get("width", 0) / 2)
        ) / slot_width
        position_score = max(
            0.0,
            1.0 - (abs(center_ratio - target_center_ratio) / 0.20),
        )

    return (
        effective_type_icon_crop_quality(candidate) * 0.55
        + float(candidate.get("confidence", 0) or 0) * 0.25
        + position_score * 0.20
    )


def is_usable_type_icon_candidate(candidate):
    if float(candidate.get("cropQuality", 0) or 0) >= 0.48:
        return True

    source = candidate.get("cropSource", candidate.get("source", ""))
    confidence = float(candidate.get("confidence", 0) or 0)
    if (
        candidate.get("hasSymbol")
        and (
            source == "fixed_type_icon_fallback"
            or source.startswith("shifted_fixed_type_icon_")
            or source == "type_icon_cluster_split"
            or source == "type_icon_pair_separated"
            or (source == "yolo_slot_object_detector" and confidence >= 0.55)
        )
    ):
        return True

    return False


def effective_type_icon_crop_quality(candidate):
    crop_quality = float(candidate.get("cropQuality", 0) or 0)
    if crop_quality > 0:
        return crop_quality

    return 0.56 if is_usable_type_icon_candidate(candidate) else 0.0


def strip_object_image(detected_object):
    return {
        key: make_json_safe_cv_value(value)
        for key, value in detected_object.items()
        if key != "image"
    }


def make_json_safe_cv_value(value):
    if isinstance(value, dict):
        return {
            key: make_json_safe_cv_value(inner_value)
            for key, inner_value in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_safe_cv_value(inner_value)
            for inner_value in value
        ]

    if hasattr(value, "item"):
        return value.item()

    return value


def detect_pokemon_sprite_object(slot_image):
    if is_empty_image(slot_image):
        return None

    candidates = generate_pokemon_sprite_candidates(slot_image)
    return select_pokemon_sprite_object(candidates, slot_image)


def generate_pokemon_sprite_candidates(slot_image):
    if is_empty_image(slot_image):
        return []

    height, width = slot_image.shape[:2]
    candidates = []

    foreground_box = detect_pokemon_sprite_box_by_foreground(slot_image)
    if foreground_box:
        candidates.append(make_pokemon_sprite_candidate(
            slot_image,
            foreground_box,
            "foreground_object_proposal",
            score_pokemon_sprite_box(foreground_box, width, height) + 0.06,
        ))

    color_box = detect_pokemon_sprite_box_by_color(slot_image)
    if color_box:
        candidates.append(make_pokemon_sprite_candidate(
            slot_image,
            color_box,
            "color_object_proposal",
            score_pokemon_sprite_box(color_box, width, height),
        ))

    relaxed_color_box = detect_pokemon_sprite_box_by_color(
        slot_image,
        saturation_min=22,
        red_saturation_min=68,
        source_padding=14,
    )
    if relaxed_color_box:
        candidates.append(make_pokemon_sprite_candidate(
            slot_image,
            relaxed_color_box,
            "relaxed_color_object_proposal",
            score_pokemon_sprite_box(relaxed_color_box, width, height) - 0.03,
        ))

    # Fallbacks are still local to the left-side sprite area, but are no longer
    # allowed to cover most of the full card.
    fixed_boxes = [
        (
            "fixed_sprite_fallback",
            {
                "x": int(width * 0.08),
                "y": int(height * 0.04),
                "width": int(width * 0.46),
                "height": int(height * 0.86),
            },
            0.30,
        ),
        (
            "shifted_fixed_sprite_fallback",
            {
                "x": int(width * 0.14),
                "y": int(height * 0.04),
                "width": int(width * 0.42),
                "height": int(height * 0.86),
            },
            0.28,
        ),
        (
            "expanded_fixed_sprite_fallback",
            {
                "x": int(width * 0.04),
                "y": int(height * 0.02),
                "width": int(width * 0.52),
                "height": int(height * 0.90),
            },
            0.24,
        ),
    ]

    for source, box, confidence in fixed_boxes:
        candidates.append(make_pokemon_sprite_candidate(slot_image, box, source, confidence))

    return dedupe_box_candidates(candidates)

def make_pokemon_sprite_candidate(slot_image, box, source, confidence):
    height, width = slot_image.shape[:2]
    original_box = clamp_box_to_image(box, width, height)
    crop = crop_image_box(slot_image, original_box)
    box, crop = trim_sprite_candidate_to_foreground(
        slot_image,
        original_box,
        crop,
    )
    quality = score_pokemon_sprite_crop_quality(crop, slot_image, box)

    return {
        "confidence": round(max(0.0, min(0.98, confidence)), 4),
        "box": box,
        "rawBox": original_box,
        "image": crop,
        "source": source,
        "cropQuality": round(quality, 4),
    }


def trim_sprite_candidate_to_foreground(slot_image, box, sprite_region):
    if is_empty_image(sprite_region) or is_empty_image(slot_image):
        return box, sprite_region

    cv2, np = load_cv_dependencies()
    crop_height, crop_width = sprite_region.shape[:2]
    hsv = cv2.cvtColor(sprite_region, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    red_card = ((hue <= 12) | (hue >= 155)) & (saturation > 48) & (value > 45)
    dark_background = value < 28
    yellow_name_text = (hue >= 20) & (hue <= 45) & (saturation > 80) & (value > 125)

    foreground = (
        (~red_card)
        & (~dark_background)
        & (~yellow_name_text)
        & (value > 34)
        & ((saturation > 14) | (value > 112))
    )

    foreground_mask = foreground.astype("uint8") * 255
    foreground_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )
    foreground_mask = cv2.morphologyEx(
        foreground_mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
    )

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground_mask)
    component_indexes = []
    min_component_area = max(50, int(crop_height * crop_width * 0.003))

    for component_index in range(1, component_count):
        x, y, component_width, component_height, area = stats[component_index]
        if area < min_component_area:
            continue
        if component_width < max(5, crop_width * 0.025):
            continue
        if component_height < max(5, crop_height * 0.035):
            continue

        center_x, center_y = centroids[component_index]
        if center_y < crop_height * 0.08 and component_height < crop_height * 0.18:
            continue

        component_indexes.append(component_index)

    if not component_indexes:
        return box, sprite_region

    # Keep components close to the largest component to avoid including card noise.
    largest_index = max(
        component_indexes,
        key=lambda component_index: stats[component_index][4],
    )
    largest_center_x, largest_center_y = centroids[largest_index]

    close_indexes = []
    for component_index in component_indexes:
        center_x, center_y = centroids[component_index]
        if abs(center_x - largest_center_x) <= crop_width * 0.38:
            if abs(center_y - largest_center_y) <= crop_height * 0.55:
                close_indexes.append(component_index)

    selected_points = np.argwhere(np.isin(labels, close_indexes or [largest_index]))
    if selected_points.size == 0:
        return box, sprite_region

    y_min, x_min = selected_points.min(axis=0)
    y_max, x_max = selected_points.max(axis=0)

    content_width = x_max - x_min + 1
    content_height = y_max - y_min + 1
    if content_width < crop_width * 0.08 or content_height < crop_height * 0.14:
        return box, sprite_region

    pad_x = max(10, int(content_width * 0.18))
    pad_y = max(10, int(content_height * 0.15))
    trim_x1 = max(0, int(x_min) - pad_x)
    trim_y1 = max(0, int(y_min) - pad_y)
    trim_x2 = min(crop_width, int(x_max) + 1 + pad_x)
    trim_y2 = min(crop_height, int(y_max) + 1 + pad_y)

    trimmed = sprite_region[trim_y1:trim_y2, trim_x1:trim_x2]
    if is_empty_image(trimmed):
        return box, sprite_region

    trimmed_height, trimmed_width = trimmed.shape[:2]
    slot_height, slot_width = slot_image.shape[:2]
    if trimmed_width < slot_width * 0.08 or trimmed_height < slot_height * 0.20:
        return box, sprite_region

    # Avoid returning huge card-like crops.
    if trimmed_width > slot_width * 0.54 or trimmed_height > slot_height * 0.94:
        return box, sprite_region

    trimmed_box = clamp_box_to_image(
        {
            "x": box["x"] + trim_x1,
            "y": box["y"] + trim_y1,
            "width": trimmed_width,
            "height": trimmed_height,
        },
        slot_width,
        slot_height,
    )
    return trimmed_box, trimmed

def score_pokemon_sprite_crop_quality(sprite_region, slot_image, box):
    if is_empty_image(sprite_region) or is_empty_image(slot_image):
        return 0.0

    cv2, _np = load_cv_dependencies()
    crop_height, crop_width = sprite_region.shape[:2]
    slot_height, slot_width = slot_image.shape[:2]

    width_ratio = crop_width / max(1, slot_width)
    height_ratio = crop_height / max(1, slot_height)

    # Good sprite crops are focused, not almost the full card.
    size_score = min(0.28, width_ratio * 0.72) + min(0.28, height_ratio * 0.44)

    hsv = cv2.cvtColor(sprite_region, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    red_card = ((hue <= 12) | (hue >= 155)) & (saturation > 48) & (value > 45)
    dark_bg = value < 32
    yellow_name_text = (hue >= 20) & (hue <= 45) & (saturation > 80) & (value > 125)
    foreground = (~red_card) & (~dark_bg) & (~yellow_name_text) & (value > 38) & ((saturation > 14) | (value > 112))
    foreground_ratio = foreground.sum() / max(1, crop_height * crop_width)
    foreground_score = min(0.30, foreground_ratio * 0.95)

    edge_penalty = 0.0
    if box["x"] <= 1 or box["x"] + box["width"] >= slot_width - 1:
        edge_penalty += 0.05
    if box["y"] <= 1 and box["y"] + box["height"] >= slot_height - 1:
        edge_penalty += 0.05

    # Penalize crops that are still basically the whole left card.
    if width_ratio > 0.52:
        edge_penalty += (width_ratio - 0.52) * 0.90
    if height_ratio > 0.94:
        edge_penalty += (height_ratio - 0.94) * 0.50

    return max(0.0, min(1.0, 0.16 + size_score + foreground_score - edge_penalty))

def dedupe_box_candidates(candidates):
    selected = []
    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: candidate.get("cropQuality", 0),
        reverse=True,
    )

    for candidate in sorted_candidates:
        if any(
            box_iou(candidate["box"], old["box"]) > 0.82
            and candidate.get("source") == old.get("source")
            for old in selected
        ):
            continue
        selected.append(candidate)

    return selected


def box_iou(first_box, second_box):
    first_x2 = first_box["x"] + first_box["width"]
    first_y2 = first_box["y"] + first_box["height"]
    second_x2 = second_box["x"] + second_box["width"]
    second_y2 = second_box["y"] + second_box["height"]

    overlap_width = max(0, min(first_x2, second_x2) - max(first_box["x"], second_box["x"]))
    overlap_height = max(0, min(first_y2, second_y2) - max(first_box["y"], second_box["y"]))
    overlap_area = overlap_width * overlap_height
    first_area = first_box["width"] * first_box["height"]
    second_area = second_box["width"] * second_box["height"]
    union_area = max(1, first_area + second_area - overlap_area)
    return overlap_area / union_area


def detect_pokemon_sprite_box_by_foreground(slot_image):
    """
    Detects a tighter Pokemon sprite box by removing the red opponent-card
    background and keeping the connected foreground components.

    This is meant to fix debug crops where pokemon_sprite spans the whole card.
    """
    if is_empty_image(slot_image):
        return None

    cv2, np = load_cv_dependencies()
    height, width = slot_image.shape[:2]

    search_x1 = int(width * 0.02)
    search_x2 = int(width * 0.62)
    search_y1 = int(height * 0.00)
    search_y2 = int(height * 0.98)

    search_region = slot_image[search_y1:search_y2, search_x1:search_x2]
    if is_empty_image(search_region):
        return None

    hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    red_card = ((hue <= 12) | (hue >= 155)) & (saturation > 48) & (value > 45)
    dark_background = value < 26
    yellow_name_text = (hue >= 20) & (hue <= 45) & (saturation > 80) & (value > 125)

    # Include colorful sprites plus pale/white sprite parts.
    foreground = (
        (~red_card)
        & (~dark_background)
        & (~yellow_name_text)
        & (value > 36)
        & ((saturation > 14) | (value > 112))
    )

    mask = foreground.astype("uint8") * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)

    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    region_height, region_width = search_region.shape[:2]
    min_area = max(70, int(region_height * region_width * 0.0035))

    components = []
    for component_index in range(1, component_count):
        x, y, component_width, component_height, area = stats[component_index]
        if area < min_area:
            continue
        if component_width < max(5, region_width * 0.025):
            continue
        if component_height < max(5, region_height * 0.045):
            continue

        center_x, center_y = centroids[component_index]

        # Ignore leftover name text/top artifacts and anything too close to type icons.
        if center_y < region_height * 0.08 and component_height < region_height * 0.18:
            continue
        if center_x > region_width * 0.92:
            continue

        components.append({
            "label": component_index,
            "x": int(x),
            "y": int(y),
            "width": int(component_width),
            "height": int(component_height),
            "area": int(area),
            "center_x": float(center_x),
            "center_y": float(center_y),
        })

    if not components:
        return None

    # Start from the largest component, then include nearby components that are
    # likely parts of the same sprite.
    main = max(components, key=lambda item: item["area"])
    selected = []
    for component in components:
        if abs(component["center_x"] - main["center_x"]) <= region_width * 0.32:
            if abs(component["center_y"] - main["center_y"]) <= region_height * 0.48:
                selected.append(component)

    if not selected:
        selected = [main]

    x1 = min(component["x"] for component in selected)
    y1 = min(component["y"] for component in selected)
    x2 = max(component["x"] + component["width"] for component in selected)
    y2 = max(component["y"] + component["height"] for component in selected)

    content_width = x2 - x1
    content_height = y2 - y1

    if content_width < width * 0.06 or content_height < height * 0.18:
        return None

    pad_x = max(10, int(content_width * 0.18))
    pad_y = max(10, int(content_height * 0.15))

    return clamp_box_to_image(
        {
            "x": search_x1 + x1 - pad_x,
            "y": search_y1 + y1 - pad_y,
            "width": content_width + pad_x * 2,
            "height": content_height + pad_y * 2,
        },
        width,
        height,
    )

def detect_pokemon_sprite_box_by_color(
    slot_image,
    saturation_min=35,
    red_saturation_min=40,
    source_padding=14,
):
    if is_empty_image(slot_image):
        return None

    cv2, np = load_cv_dependencies()
    height, width = slot_image.shape[:2]
    search_x2 = int(width * 0.64)
    search_region = slot_image[
        0:int(height * 0.98),
        :search_x2,
    ]
    if is_empty_image(search_region):
        return None

    y_offset = 0
    hsv_region = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
    hue = hsv_region[:, :, 0]
    saturation = hsv_region[:, :, 1]
    value = hsv_region[:, :, 2]

    red_background = ((hue >= 155) | (hue <= 12)) & (saturation > red_saturation_min) & (value > 45)
    colorful_foreground = (~red_background) & (saturation > saturation_min) & (value > 45)
    foreground_mask = colorful_foreground.astype("uint8") * 255

    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    foreground_mask = cv2.dilate(foreground_mask, np.ones((9, 9), np.uint8), iterations=1)

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(foreground_mask)
    candidates = []
    for component_index in range(1, component_count):
        x, y, component_width, component_height, area = stats[component_index]
        if area < 800:
            continue
        if component_width < 25 or component_height < 25:
            continue
        if component_width < search_region.shape[1] * 0.08 and component_height > search_region.shape[0] * 0.55:
            continue
        if component_width > component_height * 2.4 and component_height < search_region.shape[0] * 0.34:
            continue
        center_x = x + component_width / 2
        center_y = y + component_height / 2
        if center_x < search_region.shape[1] * 0.08:
            continue
        if center_x > search_region.shape[1] * 0.82:
            continue
        if center_y < search_region.shape[0] * 0.05:
            continue
        candidates.append((component_index, x, y, component_width, component_height, area))

    if not candidates:
        return None

    main_candidate = max(candidates, key=lambda candidate: candidate[5])
    main_label = main_candidate[0]
    close_labels = [main_label]
    main_x, main_y, main_width, main_height = main_candidate[1:5]
    main_center_x = main_x + main_width / 2
    main_center_y = main_y + main_height / 2

    for candidate in candidates:
        label, x, y, component_width, component_height, _area = candidate
        center_x = x + component_width / 2
        center_y = y + component_height / 2
        if abs(center_x - main_center_x) < search_region.shape[1] * 0.26:
            if abs(center_y - main_center_y) < search_region.shape[0] * 0.55:
                close_labels.append(label)

    selected_components = [
        candidate
        for candidate in candidates
        if candidate[0] in close_labels
    ]

    x_min = min(candidate[1] for candidate in selected_components)
    y_min = min(candidate[2] for candidate in selected_components)
    x_max = max(candidate[1] + candidate[3] for candidate in selected_components)
    y_max = max(candidate[2] + candidate[4] for candidate in selected_components)

    crop_width = x_max - x_min
    crop_height = y_max - y_min
    touches_search_edge = x_max >= search_region.shape[1] - 3
    if touches_search_edge and crop_width < search_region.shape[1] * 0.72:
        return None

    if crop_width < width * 0.13 and crop_height > height * 0.58:
        return None

    local_box = {
        "x": x_min,
        "y": y_min,
        "width": x_max - x_min,
        "height": y_max - y_min,
    }
    local_box = expand_box_xy(
        local_box,
        source_padding,
        source_padding,
        search_region.shape[1],
        search_region.shape[0],
    )
    return clamp_box_to_image(
        {
            "x": local_box["x"],
            "y": local_box["y"] + y_offset,
            "width": local_box["width"],
            "height": local_box["height"],
        },
        width,
        height,
    )


def score_pokemon_sprite_box(box, slot_width, slot_height):
    width_ratio = box["width"] / max(1, slot_width)
    height_ratio = box["height"] / max(1, slot_height)
    area_ratio = (box["width"] * box["height"]) / max(1, slot_width * slot_height)
    score = 0.45 + min(0.25, width_ratio) + min(0.20, height_ratio * 0.45) + min(0.10, area_ratio)
    return min(0.98, score)


# Checks that a sprite crop is large enough to be useful for matching.
def is_reasonable_sprite_crop(sprite_region, slot_image):
    if is_empty_image(sprite_region) or is_empty_image(slot_image):
        return False

    sprite_height, sprite_width = sprite_region.shape[:2]
    slot_height, slot_width = slot_image.shape[:2]

    if sprite_width < slot_width * 0.08:
        return False
    if sprite_height < slot_height * 0.20:
        return False

    # Main debug issue: reject crops that are basically the full card.
    if sprite_width > slot_width * 0.56:
        return False
    if sprite_height > slot_height * 0.96:
        return False

    return True

# Extracts the sprite by removing the red card background and type icons.
def extract_sprite_by_color(slot_image):
    if is_empty_image(slot_image):
        return slot_image

    sprite_object = detect_pokemon_sprite_object(slot_image)
    if not sprite_object:
        return None

    return sprite_object["image"]


# Detects likely Pokemon types from the colored icons on an opponent card.
def detect_types_from_opponent_slot(slot_image, type_references=None):
    return detect_type_method_results(slot_image, type_references=type_references or [])["selected"]


def detect_type_method_results(slot_image, type_references=None, type_icon_crops=None):
    if is_empty_image(slot_image):
        return {
            "fixedSlot": [],
            "comboTemplate": [],
            "selected": [],
        }

    icon_crops = (
        type_icon_crops
        if type_icon_crops is not None
        else crop_adaptive_type_icons_from_slot(slot_image)
    )
    combo_crop = build_type_combo_candidate_crop(slot_image, icon_crops)
    combo_result = classify_type_crop(
        combo_crop,
        type_references=type_references or [],
    )
    detected_types = combo_result.get("types", [])

    return {
        "fixedSlot": detected_types,
        "comboTemplate": detected_types,
        "selected": normalize_detected_type_pair(detected_types),
        "typeComboDetails": {
            **combo_result,
            "cropSource": combo_crop.get("cropSource", "") if combo_crop else "",
            "typeCount": combo_crop.get("typeCount", 0) if combo_crop else 0,
        },
    }


def classify_type_crop(combo_crop, type_references=None):
    if not combo_crop:
        return unknown_type_combo_result("empty-image")

    type_count = combo_crop.get("typeCount")
    crop_image = combo_crop.get("image")

    if type_count == 1:
        detected_type = classify_type_by_template(
            crop_image,
            type_references or load_type_icon_references(),
        )
        if not detected_type:
            return unknown_type_combo_result("single-type-no-match")

        return {
            "types": [detected_type],
            "typeKey": detected_type,
            "referenceImage": "",
            "score": 1.0,
            "pixelScore": 1.0,
            "colorScore": 1.0,
            "orbScore": 1.0,
            "predictionSource": "single_type_template",
            "needsReview": False,
        }

    icon_crops = combo_crop.get("iconCrops") or []
    if icon_crops:
        references = type_references or load_type_icon_references()
        icon_types = []
        for icon_crop in icon_crops[:2]:
            detected_type = classify_type_by_template(icon_crop.get("image"), references)
            if detected_type and detected_type not in icon_types:
                icon_types.append(detected_type)

        if icon_types:
            return {
                "types": normalize_detected_type_pair(icon_types),
                "typeKey": "_".join(icon_types),
                "referenceImage": "",
                "score": 1.0,
                "pixelScore": 1.0,
                "colorScore": 1.0,
                "orbScore": 1.0,
                "predictionSource": "individual_type_icon_template",
                "needsReview": len(icon_types) < int(type_count or 0),
            }

    return classify_type_combo_by_template(
        crop_image,
        load_type_combo_references(),
        expected_type_count=type_count,
    )


def build_type_combo_candidate_crop(slot_image, type_icon_crops=None):
    if is_empty_image(slot_image):
        return None

    icon_crops = (
        type_icon_crops
        if type_icon_crops is not None
        else crop_adaptive_type_icons_from_slot(slot_image)
    )
    usable_crops = [
        icon_crop
        for icon_crop in icon_crops
        if is_usable_type_icon_candidate(icon_crop)
        and not is_empty_image(icon_crop.get("image"))
    ]
    usable_crops.sort(key=lambda icon_crop: icon_crop.get("x", 0))
    usable_crops = usable_crops[:2]

    if not usable_crops:
        return None

    if len(usable_crops) >= 2:
        height, width = slot_image.shape[:2]
        source_boxes = [
            {
                "x": icon_crop.get("x", 0),
                "y": icon_crop.get("y", 0),
                "width": icon_crop.get("width", 1),
                "height": icon_crop.get("height", 1),
            }
            for icon_crop in usable_crops
        ]
        cluster_box = combine_boxes(source_boxes, width, height)
        if cluster_box:
            cluster_box = expand_box_xy(
                cluster_box,
                max(2, int(width * 0.004)),
                max(2, int(height * 0.008)),
                width,
                height,
            )
        if not cluster_box:
            cluster_box = extract_detected_type_icon_cluster_box(slot_image)
        cluster_image = crop_image_box(slot_image, cluster_box) if cluster_box else None

        if not is_empty_image(cluster_image):
            return {
                "image": cluster_image,
                "typeCount": 2,
                "cropSource": "type_cluster_candidate",
                "sourceBoxes": [
                    {
                        "index": icon_crop.get("index"),
                        "x": icon_crop.get("x"),
                        "y": icon_crop.get("y"),
                        "width": icon_crop.get("width"),
                        "height": icon_crop.get("height"),
                        "cropSource": icon_crop.get("cropSource", ""),
                        "cropQuality": icon_crop.get("cropQuality"),
                    }
                    for icon_crop in usable_crops
                ],
                "iconCrops": usable_crops,
                "clusterBox": cluster_box,
            }

        return None

    single_crop = usable_crops[0]

    return {
        "image": single_crop["image"],
        "typeCount": 1,
        "cropSource": "single_type_icon_candidate",
        "sourceBoxes": [
            {
                "index": icon_crop.get("index"),
                "x": icon_crop.get("x"),
                "y": icon_crop.get("y"),
                "width": icon_crop.get("width"),
                "height": icon_crop.get("height"),
                "cropSource": icon_crop.get("cropSource", ""),
                "cropQuality": icon_crop.get("cropQuality"),
            }
            for icon_crop in usable_crops
        ],
        "iconCrops": usable_crops,
    }


def stitch_type_icon_crops(icon_images, gap=8, padding=0):
    cv2, np = load_cv_dependencies()
    images = [image for image in icon_images if not is_empty_image(image)]

    if not images:
        return None

    if len(images) == 1 and padding == 0:
        return images[0]

    target_height = max(image.shape[0] for image in images)
    resized_images = []

    for image in images:
        height, width = image.shape[:2]
        next_width = max(1, round(width * (target_height / max(1, height))))
        resized_images.append(
            cv2.resize(image, (next_width, target_height), interpolation=cv2.INTER_AREA)
        )

    output_height = target_height + (padding * 2)
    output_width = (
        sum(image.shape[1] for image in resized_images)
        + (gap * max(0, len(resized_images) - 1))
        + (padding * 2)
    )
    canvas = np.full((output_height, output_width, 3), 255, dtype=np.uint8)

    x_offset = padding
    for image in resized_images:
        image_height, image_width = image.shape[:2]
        y_offset = padding + ((target_height - image_height) // 2)
        canvas[y_offset:y_offset + image_height, x_offset:x_offset + image_width] = image
        x_offset += image_width + gap

    return canvas


def classify_type_combo_by_template(combo_crop, type_combo_references, expected_type_count=None):
    if is_empty_image(combo_crop):
        return unknown_type_combo_result("empty-image")

    cv2, _np = load_cv_dependencies()
    candidates = [
        reference
        for reference in type_combo_references or []
        if expected_type_count is None
        or len(reference.get("types", [])) == expected_type_count
    ]

    if not candidates:
        return unknown_type_combo_result("no-references")

    crop_color = extract_type_color_signature(combo_crop)
    scored_results = []

    for reference in candidates:
        reference_image = reference.get("image")

        if is_empty_image(reference_image):
            continue

        resized_reference = cv2.resize(
            reference_image,
            (combo_crop.shape[1], combo_crop.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        resized_crop = cv2.resize(
            combo_crop,
            (resized_reference.shape[1], resized_reference.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

        pixel_score = 1.0 - (
            cv2.absdiff(resized_crop, resized_reference).mean() / 255.0
        )
        color_score = compare_type_color_signatures(
            crop_color,
            extract_type_color_signature(reference_image),
        )
        orb_score = compute_orb_similarity(combo_crop, reference_image)
        final_score = (
            pixel_score * 0.45
            + color_score * 0.35
            + orb_score * 0.20
        )

        scored_results.append({
            "types": reference.get("types", []),
            "typeKey": reference.get("typeKey", ""),
            "referenceImage": reference.get("path", ""),
            "score": round(float(final_score), 4),
            "pixelScore": round(float(pixel_score), 4),
            "colorScore": round(float(color_score), 4),
            "orbScore": round(float(orb_score), 4),
        })

    scored_results.sort(key=lambda item: item["score"], reverse=True)

    if not scored_results:
        return unknown_type_combo_result("no-match")

    best_result = scored_results[0]

    if best_result["score"] < 0.42:
        return {
            **best_result,
            "types": [],
            "predictionSource": "type_combo_template_low_confidence",
            "needsReview": True,
        }

    return {
        **best_result,
        "predictionSource": "type_combo_template",
        "needsReview": best_result["score"] < TYPE_COMBO_TRUST_THRESHOLD,
    }


def unknown_type_combo_result(reason):
    return {
        "types": [],
        "typeKey": "",
        "referenceImage": "",
        "score": 0.0,
        "pixelScore": 0.0,
        "colorScore": 0.0,
        "orbScore": 0.0,
        "predictionSource": f"type_combo_template_{reason}",
        "needsReview": True,
    }


def has_poison_icon_hint(slot_image):
    if is_empty_image(slot_image):
        return False

    cv2, np = load_cv_dependencies()
    region = extract_type_icon_region(slot_image)
    if is_empty_image(region):
        return False

    height, width = region.shape[:2]
    right_region = region[:, int(width * 0.40):]
    if is_empty_image(right_region):
        return False

    hsv = cv2.cvtColor(right_region, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    purple_pixels = (
        (hue >= 118)
        & (hue <= 158)
        & (saturation > 45)
        & (value > 55)
    )
    white_symbol = (saturation < 120) & (value > 95)

    area = max(1, right_region.shape[0] * right_region.shape[1])
    return purple_pixels.sum() / area >= 0.08 and white_symbol.sum() / area >= 0.015


def has_rock_flying_icon_pair_hint(slot_image):
    if is_empty_image(slot_image):
        return False

    cv2, _np = load_cv_dependencies()
    region = extract_type_icon_region(slot_image)
    if is_empty_image(region):
        return False

    height, width = region.shape[:2]
    upper_region = region[:int(height * 0.82), :]
    left_icon = upper_region[:, int(width * 0.04):int(width * 0.48)]
    right_icon = upper_region[:, int(width * 0.46):int(width * 0.92)]
    if is_empty_image(left_icon) or is_empty_image(right_icon):
        return False

    left_hsv = cv2.cvtColor(left_icon, cv2.COLOR_BGR2HSV)
    right_hsv = cv2.cvtColor(right_icon, cv2.COLOR_BGR2HSV)

    left_saturation = left_hsv[:, :, 1]
    left_value = left_hsv[:, :, 2]
    right_hue = right_hsv[:, :, 0]
    right_saturation = right_hsv[:, :, 1]
    right_value = right_hsv[:, :, 2]

    left_glared_rock = (left_saturation < 95) & (left_value > 175)
    right_blue_flying = (
        (right_hue >= 86)
        & (right_hue <= 132)
        & (right_saturation > 45)
        & (right_value > 80)
    )

    left_area = max(1, left_icon.shape[0] * left_icon.shape[1])
    right_area = max(1, right_icon.shape[0] * right_icon.shape[1])
    return left_glared_rock.sum() / left_area >= 0.18 and right_blue_flying.sum() / right_area >= 0.10


def has_dark_icon_hint(slot_image):
    if is_empty_image(slot_image):
        return False

    cv2, np = load_cv_dependencies()
    region = extract_type_icon_region(slot_image)
    if is_empty_image(region):
        return False

    height, width = region.shape[:2]
    search_regions = [
        region[:, :int(width * 0.55)],
        region[:, int(width * 0.35):],
    ]

    for search_region in search_regions:
        if is_empty_image(search_region):
            continue

        if has_dark_icon_region_hint(search_region):
            return True

    return False


def has_dark_icon_region_hint(icon_region):
    if is_empty_image(icon_region):
        return False

    cv2, np = load_cv_dependencies()
    hsv = cv2.cvtColor(icon_region, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    gray_icon = (saturation < 90) & (value > 70) & (value < 225)
    white_symbol = (saturation < 115) & (value > 95)
    area = max(1, icon_region.shape[0] * icon_region.shape[1])
    return gray_icon.sum() / area >= 0.08 and white_symbol.sum() / area >= 0.020


def has_normal_psychic_icon_hint(slot_image):
    if is_empty_image(slot_image):
        return False

    cv2, np = load_cv_dependencies()
    region = extract_type_icon_region(slot_image)
    if is_empty_image(region):
        return False

    height, width = region.shape[:2]
    left_region = region[:, :width // 2]
    right_region = region[:, width // 2:]
    if is_empty_image(left_region) or is_empty_image(right_region):
        return False

    left_hsv = cv2.cvtColor(left_region, cv2.COLOR_BGR2HSV)
    right_hsv = cv2.cvtColor(right_region, cv2.COLOR_BGR2HSV)

    left_pale = (left_hsv[:, :, 1] < 65) & (left_hsv[:, :, 2] > 160)
    right_pink = (
        (right_hsv[:, :, 0] >= 135)
        & (right_hsv[:, :, 0] <= 165)
        & (right_hsv[:, :, 1] > 45)
        & (right_hsv[:, :, 2] > 95)
    )

    return (
        left_pale.sum() / max(1, left_region.shape[0] * left_region.shape[1]) >= 0.32
        and right_pink.sum() / max(1, right_region.shape[0] * right_region.shape[1]) >= 0.20
    )

# Crops the part of an opponent card that contains both type icons.
def extract_type_icon_region(slot_image):
    if is_empty_image(slot_image):
        return slot_image

    height, width = slot_image.shape[:2]

    return slot_image[
        int(height * 0.01):int(height * 0.56),
        int(width * 0.53):int(width * 0.98),
    ]


# Returns a wider right-side region used only when normal type splitting fails.
def extract_fallback_type_icon_region(slot_image):
    if is_empty_image(slot_image):
        return slot_image

    height, width = slot_image.shape[:2]

    return slot_image[
        int(height * 0.00):int(height * 0.62),
        int(width * 0.45):int(width * 1.00),
    ]

# Checks whether two detected icon boxes overlap too much.
def type_icon_boxes_overlap(first_match, second_match):
    first_x2 = first_match["x"] + first_match["width"]
    first_y2 = first_match["y"] + first_match["height"]

    second_x2 = second_match["x"] + second_match["width"]
    second_y2 = second_match["y"] + second_match["height"]

    overlap_width = max(0, min(first_x2, second_x2) - max(first_match["x"], second_match["x"]))
    overlap_height = max(0, min(first_y2, second_y2) - max(first_match["y"], second_match["y"]))

    overlap_area = overlap_width * overlap_height
    smaller_area = min(
        first_match["width"] * first_match["height"],
        second_match["width"] * second_match["height"],
    )

    return smaller_area > 0 and overlap_area / smaller_area > 0.20


def detect_type_icon_objects_from_slot(slot_image):
    """
    Detects type icon objects from one opponent card slot.

    Main improvement:
    - component detection still runs first
    - if crops overlap or look contaminated, rebuild from a local type cluster
    - final output is one clean square per type icon
    """
    if is_empty_image(slot_image):
        return []

    objects = detect_type_icon_objects_by_components(slot_image)

    if len(objects) >= 2:
        objects = sorted(objects[:2], key=lambda item: item["box"]["x"])
        first = object_box_to_top_level(objects[0])
        second = object_box_to_top_level(objects[1])

        if not type_icon_pair_needs_cluster_split(first, second, slot_image):
            for index, item in enumerate(objects, start=1):
                item["label"] = f"type_icon_{index}"
                item["role"] = f"type_icon_{index}"
                item["index"] = index
            return objects[:2]

    cluster_objects = detect_type_icon_objects_by_cluster_split(slot_image)
    if cluster_objects:
        return cluster_objects

    if objects:
        objects = sorted(objects[:2], key=lambda item: item["box"]["x"])
        for index, item in enumerate(objects, start=1):
            item["label"] = f"type_icon_{index}"
            item["role"] = f"type_icon_{index}"
            item["index"] = index
        return objects[:2]

    return []


def detect_type_icon_objects_by_components(slot_image):
    """
    Original component-based type-icon detector, kept as the first attempt.
    """
    if is_empty_image(slot_image):
        return []

    region = extract_type_icon_region(slot_image)
    if is_empty_image(region):
        return []

    height, width = slot_image.shape[:2]
    region_y = int(height * 0.01)
    region_x = int(width * 0.53)
    region_icons = crop_two_type_icons_from_region(region)

    objects = []

    for icon in region_icons:
        box = clamp_box_to_image(
            {
                "x": region_x + icon["x"],
                "y": region_y + icon["y"],
                "width": icon["width"],
                "height": icon["height"],
            },
            width,
            height,
        )

        box = expand_type_icon_object_box(box, width, height)
        crop = crop_image_box(slot_image, box)

        if is_empty_image(crop):
            continue

        objects.append({
            "label": f"type_icon_{len(objects) + 1}",
            "role": f"type_icon_{len(objects) + 1}",
            "index": len(objects) + 1,
            "confidence": 0.82 if icon.get("hasSymbol") else 0.55,
            "box": box,
            "image": crop,
            "hasSymbol": bool(icon.get("hasSymbol")) or has_type_icon_symbol(crop),
            "area": box["width"] * box["height"],
            "source": "type_icon_object_proposal",
            "cropQuality": round(score_type_icon_crop_quality(crop), 4),
        })

    if len(objects) < 2:
        fixed_objects = detect_fixed_type_icon_objects_from_slot(slot_image)

        for fixed_object in fixed_objects:
            fixed_center = fixed_object["box"]["x"] + fixed_object["box"]["width"] / 2

            if any(
                abs(fixed_center - (old["box"]["x"] + old["box"]["width"] / 2)) < width * 0.08
                for old in objects
            ):
                continue

            objects.append(fixed_object)

            if len(objects) == 2:
                break

    objects.sort(key=lambda item: item["box"]["x"])

    for index, item in enumerate(objects[:2], start=1):
        item["label"] = f"type_icon_{index}"
        item["role"] = f"type_icon_{index}"
        item["index"] = index

    return objects[:2]


def extract_detected_type_icon_cluster_box(slot_image):
    """
    Finds the local cluster containing one or two type icons.

    This is used both for actual cluster splitting and for debug output.
    """
    if is_empty_image(slot_image):
        return None

    cv2, np = load_cv_dependencies()

    height, width = slot_image.shape[:2]

    search_x1 = int(width * 0.52)
    search_x2 = int(width * 0.98)
    search_y1 = int(height * 0.00)
    search_y2 = int(height * 0.58)

    search_region = slot_image[search_y1:search_y2, search_x1:search_x2]

    if is_empty_image(search_region):
        return None

    hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    red_card = (
        ((hue <= 12) | (hue >= 155))
        & (saturation > 55)
        & (value > 45)
    )

    colored_tile = (
        (value > 70)
        & (
            (saturation > 35)
            | (value > 170)
        )
        & (~red_card)
    )

    white_or_pale = (
        (saturation < 120)
        & (value > 135)
    )

    mask = (colored_tile | white_or_pale).astype("uint8") * 255

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
    )

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask)

    component_boxes = []

    region_height, region_width = search_region.shape[:2]
    min_area = max(80, int(region_height * region_width * 0.010))

    for component_index in range(1, component_count):
        x, y, component_width, component_height, area = stats[component_index]

        if area < min_area:
            continue

        if component_width < region_width * 0.05:
            continue

        if component_height < region_height * 0.15:
            continue

        center_y = y + component_height / 2

        # Avoid gender icon below the type icons.
        if center_y > region_height * 0.72:
            continue

        fill_ratio = area / max(1, component_width * component_height)
        if component_width > region_width * 0.55 and fill_ratio < 0.20:
            continue

        component_boxes.append({
            "x": search_x1 + int(x),
            "y": search_y1 + int(y),
            "width": int(component_width),
            "height": int(component_height),
            "area": int(area),
        })

    if not component_boxes:
        return None

    cluster_box = combine_boxes(component_boxes, width, height)

    cluster_box = expand_box_xy(
        cluster_box,
        max(4, int(width * 0.010)),
        max(4, int(height * 0.025)),
        width,
        height,
    )

    return cluster_box


def detect_type_icon_objects_by_cluster_split(slot_image):
    """
    Builds a full local type-icon cluster and splits it into left/right squares.
    """
    if is_empty_image(slot_image):
        return []

    cluster_box = extract_detected_type_icon_cluster_box(slot_image)

    if not cluster_box:
        return []

    return split_type_icon_cluster_box(slot_image, cluster_box)


def split_type_icon_cluster_box(slot_image, cluster_box):
    """
    Splits one local type-icon cluster into one or two clean square boxes.
    """
    if is_empty_image(slot_image) or not cluster_box:
        return []

    height, width = slot_image.shape[:2]

    cluster_x = cluster_box["x"]
    cluster_y = cluster_box["y"]
    cluster_width = cluster_box["width"]
    cluster_height = cluster_box["height"]

    icon_size = int(round(cluster_height * 1.05))

    min_size = int(round(height * 0.28))
    max_size = int(round(height * 0.50))
    icon_size = max(min_size, min(icon_size, max_size))

    two_icons = cluster_width >= icon_size * 1.14
    if two_icons:
        gap = max(4, int(round(icon_size * 0.06)))
        fitted_size = int((cluster_width - gap) / 2)
        icon_size = max(min_size, min(icon_size, fitted_size))

    center_y = cluster_y + cluster_height / 2

    if two_icons:
        candidate_centers = [
            (1, cluster_x + (icon_size / 2)),
            (2, cluster_x + cluster_width - (icon_size / 2)),
        ]
    else:
        candidate_centers = [
            (1, cluster_x + cluster_width / 2),
        ]

    boxes = []

    for index, center_x in candidate_centers:
        box = clamp_box_to_image(
            {
                "x": int(round(center_x - icon_size / 2)),
                "y": int(round(center_y - icon_size / 2)),
                "width": icon_size,
                "height": icon_size,
            },
            width,
            height,
        )

        crop = crop_image_box(slot_image, box)

        if is_empty_image(crop):
            continue

        boxes.append({
            "label": f"type_icon_{index}",
            "role": f"type_icon_{index}",
            "index": index,
            "confidence": 0.70 if has_type_icon_symbol(crop) else 0.50,
            "box": box,
            "image": crop,
            "hasSymbol": has_type_icon_symbol(crop),
            "area": box["width"] * box["height"],
            "source": "type_icon_cluster_split",
            "cropQuality": round(score_type_icon_crop_quality(crop), 4),
        })

    boxes.sort(key=lambda item: item["box"]["x"])

    for index, item in enumerate(boxes, start=1):
        item["label"] = f"type_icon_{index}"
        item["role"] = f"type_icon_{index}"
        item["index"] = index

    return boxes[:2]


def type_icon_pair_needs_cluster_split(first_object, second_object, slot_image):
    """
    Decides whether component crops are poor enough to rebuild from the local cluster.
    """
    if not first_object or not second_object or is_empty_image(slot_image):
        return True

    first_box = first_object.get("box") or first_object
    second_box = second_object.get("box") or second_object

    if type_icon_boxes_overlap(first_object, second_object):
        return True

    first_x2 = first_box["x"] + first_box["width"]
    second_x1 = second_box["x"]
    gap = second_x1 - first_x2

    if gap < 2:
        return True

    first_area = max(1, first_box["width"] * first_box["height"])
    second_area = max(1, second_box["width"] * second_box["height"])

    if max(first_area, second_area) / min(first_area, second_area) > 1.55:
        return True

    return False


def object_box_to_top_level(detected_object):
    """
    Converts an object with object['box'] into the same shape expected by
    type_icon_boxes_overlap().
    """
    box = detected_object.get("box") or {}

    return {
        **detected_object,
        "x": box.get("x", detected_object.get("x", 0)),
        "y": box.get("y", detected_object.get("y", 0)),
        "width": box.get("width", detected_object.get("width", 1)),
        "height": box.get("height", detected_object.get("height", 1)),
    }


def combine_boxes(boxes, image_width, image_height):
    """
    Combines multiple boxes into one image-clamped bounding box.
    """
    if not boxes:
        return None

    x1 = min(box["x"] for box in boxes)
    y1 = min(box["y"] for box in boxes)
    x2 = max(box["x"] + box["width"] for box in boxes)
    y2 = max(box["y"] + box["height"] for box in boxes)

    return clamp_box_to_image(
        {
            "x": x1,
            "y": y1,
            "width": max(1, x2 - x1),
            "height": max(1, y2 - y1),
        },
        image_width,
        image_height,
    )


def expand_type_icon_object_box(box, slot_width, slot_height):
    """
    Expands a raw type-icon component into a conservative square crop.

    This is intentionally smaller than before because cluster splitting now
    handles bad overlap cases.
    """
    padding_x = max(6, int(slot_width * 0.010))
    padding_y = max(6, int(slot_height * 0.030))

    expanded = expand_box_xy(
        box,
        padding_x,
        padding_y,
        slot_width,
        slot_height,
    )

    target_size = max(expanded["width"], expanded["height"])

    max_size = int(min(slot_width * 0.18, slot_height * 0.48))
    min_size = int(slot_height * 0.30)

    target_size = min(max_size, max(target_size, min_size))

    center_x = expanded["x"] + expanded["width"] / 2
    center_y = expanded["y"] + expanded["height"] / 2

    return clamp_box_to_image(
        {
            "x": center_x - target_size / 2,
            "y": center_y - target_size / 2,
            "width": target_size,
            "height": target_size,
        },
        slot_width,
        slot_height,
    )


def detect_fixed_type_icon_objects_from_slot(slot_image):
    if is_empty_image(slot_image):
        return []

    height, width = slot_image.shape[:2]
    fixed_boxes = [
        {
            "index": 1,
            "left": 0.595,
            "top": 0.045,
            "width": 0.155,
            "height": 0.385,
        },
        {
            "index": 2,
            "left": 0.745,
            "top": 0.045,
            "width": 0.155,
            "height": 0.385,
        },
    ]

    objects = []
    for box in fixed_boxes:
        pixel_box = clamp_box_to_image(
            {
                "x": width * box["left"],
                "y": height * box["top"],
                "width": width * box["width"],
                "height": height * box["height"],
            },
            width,
            height,
        )
        crop = crop_image_box(slot_image, pixel_box)
        if is_empty_image(crop):
            continue

        has_symbol = has_type_icon_symbol(crop)
        objects.append({
            "label": f"type_icon_{box['index']}",
            "index": box["index"],
            "confidence": 0.62 if has_symbol else 0.42,
            "box": pixel_box,
            "image": crop,
            "hasSymbol": has_symbol,
            "area": pixel_box["width"] * pixel_box["height"],
            "source": "fixed_type_icon_fallback",
        })

    return objects


def crop_object_type_icons_from_slot(slot_image):
    crops = []
    for detected_object in detect_type_icon_objects_from_slot(slot_image):
        box = detected_object["box"]
        crop = detected_object.get("image")
        if is_empty_image(crop):
            crop = crop_image_box(slot_image, box)

        crops.append({
            "index": detected_object.get("index", len(crops) + 1),
            "x": box["x"],
            "y": box["y"],
            "width": box["width"],
            "height": box["height"],
            "image": crop,
            "hasSymbol": bool(detected_object.get("hasSymbol")),
            "area": detected_object.get("area", box["width"] * box["height"]),
            "cropSource": detected_object.get("source", "object_proposal"),
            "confidence": detected_object.get("confidence", 0.0),
        })

    return crops


def crop_adaptive_type_icons_from_slot(slot_image):
    if is_empty_image(slot_image):
        return []

    object_layer = detect_slot_object_layer(slot_image)
    return type_icon_crops_from_object_layer(object_layer)


def type_icon_crops_from_object_layer(object_layer):
    crops = []
    for role in ("type_icon_1", "type_icon_2"):
        detected_object = object_layer.get(role)
        if not detected_object:
            continue

        box = detected_object.get("box") or {}
        crops.append({
            "index": detected_object.get("index", len(crops) + 1),
            "x": box.get("x", 0),
            "y": box.get("y", 0),
            "width": box.get("width", 0),
            "height": box.get("height", 0),
            "image": detected_object.get("image"),
            "hasSymbol": bool(detected_object.get("hasSymbol")),
            "area": detected_object.get("area", box.get("width", 0) * box.get("height", 0)),
            "cropSource": detected_object.get("source", ""),
            "source": detected_object.get("source", ""),
            "confidence": detected_object.get("confidence", 0.0),
            "cropQuality": detected_object.get("cropQuality"),
        })

    return crops


def generate_type_icon_crop_candidates(slot_image):
    if is_empty_image(slot_image):
        return []

    candidates = []

    for icon_crop in crop_object_type_icons_from_slot(slot_image):
        candidates.append(make_type_icon_candidate(
            icon_crop,
            icon_crop.get("cropSource", "type_icon_object_proposal"),
            icon_crop.get("confidence", 0.82),
        ))

    for fixed_crop in crop_fixed_type_icons_from_slot(slot_image):
        candidates.append(make_type_icon_candidate(
            fixed_crop,
            "fixed_type_icon_fallback",
            0.62 if fixed_crop.get("hasSymbol") else 0.42,
        ))

    for shifted_crop in crop_shifted_fixed_type_icons_from_slot(slot_image):
        candidates.append(make_type_icon_candidate(
            shifted_crop,
            shifted_crop.get("cropSource", "shifted_fixed_type_icon_fallback"),
            0.58 if shifted_crop.get("hasSymbol") else 0.38,
        ))

    for wide_crop in crop_wide_panel_type_icons_from_slot(slot_image):
        candidates.append(make_type_icon_candidate(
            wide_crop,
            "wide_panel_type_icon_candidate",
            0.55 if wide_crop.get("hasSymbol") else 0.35,
        ))

    return dedupe_type_icon_candidates(candidates)


def make_type_icon_candidate(icon_crop, crop_source, confidence):
    crop = icon_crop.get("image")
    quality = score_type_icon_crop_quality(crop)
    return {
        **icon_crop,
        "cropSource": crop_source,
        "source": crop_source,
        "confidence": round(max(0.0, min(0.98, confidence)), 4),
        "cropQuality": round(quality, 4),
        "hasSymbol": bool(icon_crop.get("hasSymbol")) or quality >= 0.62,
    }


def dedupe_type_icon_candidates(candidates):
    selected = []
    sorted_candidates = sorted(
        candidates,
        key=lambda candidate: (
            candidate.get("cropQuality", 0),
            candidate.get("confidence", 0),
        ),
        reverse=True,
    )

    for candidate in sorted_candidates:
        if is_empty_image(candidate.get("image")):
            continue
        if any(type_icon_boxes_overlap(candidate, old) for old in selected):
            continue
        selected.append(candidate)

    return selected


def crop_shifted_fixed_type_icons_from_slot(slot_image):
    if is_empty_image(slot_image):
        return []

    height, width = slot_image.shape[:2]
    base_boxes = [
        {"index": 1, "left": 0.595, "top": 0.045, "width": 0.155, "height": 0.385},
        {"index": 2, "left": 0.745, "top": 0.045, "width": 0.155, "height": 0.385},
    ]
    shifts = [
        ("left", -0.025, 0.0),
        ("right", 0.025, 0.0),
        ("down", 0.0, 0.045),
        ("wide", -0.015, -0.015),
    ]
    crops = []

    for base_box in base_boxes:
        for suffix, x_shift, y_shift in shifts:
            width_ratio = base_box["width"] + (0.030 if suffix == "wide" else 0.0)
            height_ratio = base_box["height"] + (0.035 if suffix == "wide" else 0.0)
            pixel_box = clamp_box_to_image(
                {
                    "x": width * (base_box["left"] + x_shift),
                    "y": height * (base_box["top"] + y_shift),
                    "width": width * width_ratio,
                    "height": height * height_ratio,
                },
                width,
                height,
            )
            crop = crop_image_box(slot_image, pixel_box)
            if is_empty_image(crop):
                continue
            crops.append({
                "index": base_box["index"],
                "x": pixel_box["x"],
                "y": pixel_box["y"],
                "width": pixel_box["width"],
                "height": pixel_box["height"],
                "image": crop,
                "hasSymbol": has_type_icon_symbol(crop),
                "area": pixel_box["width"] * pixel_box["height"],
                "cropSource": f"shifted_fixed_type_icon_{suffix}",
            })

    return crops


def crop_wide_panel_type_icons_from_slot(slot_image):
    if is_empty_image(slot_image):
        return []

    height, width = slot_image.shape[:2]
    panel_box = clamp_box_to_image(
        {
            "x": int(width * 0.50),
            "y": int(height * 0.00),
            "width": int(width * 0.49),
            "height": int(height * 0.60),
        },
        width,
        height,
    )
    panel = crop_image_box(slot_image, panel_box)
    if is_empty_image(panel):
        return []

    crops = []
    panel_height, panel_width = panel.shape[:2]
    split_boxes = [
        {"index": 1, "x": int(panel_width * 0.16), "y": int(panel_height * 0.06)},
        {"index": 2, "x": int(panel_width * 0.48), "y": int(panel_height * 0.06)},
    ]
    size = int(min(panel_width * 0.34, panel_height * 0.72))

    for split_box in split_boxes:
        local_box = clamp_box_to_image(
            {
                "x": split_box["x"],
                "y": split_box["y"],
                "width": size,
                "height": size,
            },
            panel_width,
            panel_height,
        )
        global_box = clamp_box_to_image(
            {
                "x": panel_box["x"] + local_box["x"],
                "y": panel_box["y"] + local_box["y"],
                "width": local_box["width"],
                "height": local_box["height"],
            },
            width,
            height,
        )
        crop = crop_image_box(slot_image, global_box)
        if is_empty_image(crop):
            continue
        crops.append({
            "index": split_box["index"],
            "x": global_box["x"],
            "y": global_box["y"],
            "width": global_box["width"],
            "height": global_box["height"],
            "image": crop,
            "hasSymbol": has_type_icon_symbol(crop),
            "area": global_box["width"] * global_box["height"],
            "cropSource": "wide_panel_type_icon_candidate",
        })

    return crops


def crop_fixed_type_icons_from_slot(slot_image, position=None):
    """
    Crops predictable type icon boxes from a rectified opponent card.

    This is better than contour detection for guided-camera images because
    the card layout is stable after rectification.
    """
    if is_empty_image(slot_image):
        return []

    height, width = slot_image.shape[:2]

    # These ratios are based on your debug crops.
    # They intentionally crop the square icon area, not the whole right panel.
    fixed_boxes = [
        {
            "index": 1,
            "left": 0.595,
            "top": 0.045,
            "width": 0.155,
            "height": 0.385,
        },
        {
            "index": 2,
            "left": 0.745,
            "top": 0.045,
            "width": 0.155,
            "height": 0.385,
        },
    ]

    crops = []

    for box in fixed_boxes:
        x1 = int(width * box["left"])
        y1 = int(height * box["top"])
        x2 = int(width * (box["left"] + box["width"]))
        y2 = int(height * (box["top"] + box["height"]))

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))

        crop = slot_image[y1:y2, x1:x2]

        if is_empty_image(crop):
            continue

        # Keep the crop even if symbol check is weak.
        # Embedding classifier can decide if it is valid.
        crops.append({
            "index": box["index"],
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
            "image": crop,
            "hasSymbol": has_type_icon_symbol(crop),
            "area": (x2 - x1) * (y2 - y1),
            "cropSource": "fixed_type_icon_box",
        })

    return crops

def crop_type_icons_from_slot(slot_image, position=None):
    if is_empty_image(slot_image):
        return []

    cv2, np = load_cv_dependencies()

    region = extract_type_icon_region(slot_image)
    if is_empty_image(region):
        return []

    region_h, region_w = region.shape[:2]

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Remove red card background.
    red_background = (
        ((hue <= 12) | (hue >= 160))
        & (saturation > 45)
        & (value > 45)
    )

    # Remove very dark background.
    black_background = value < 35

    # Keep colored type box pixels and white symbol pixels.
    white_symbol = (saturation < 120) & (value > 90)
    colored_box = (~red_background) & (~black_background) & (saturation > 25) & (value > 45)

    mask = (colored_box | white_symbol).astype("uint8") * 255

    # Clean noise and connect broken icon parts.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        area = w * h
        if area < region_h * region_w * 0.018:
            continue

        # Type icon boxes should be reasonably tall and square-ish.
        if w < region_w * 0.10 or h < region_h * 0.25:
            continue

        aspect = w / max(1, h)
        if aspect < 0.45 or aspect > 1.75:
            continue

        # Ignore huge merged regions.
        if w > region_w * 0.55:
            continue

        pad = int(region_h * 0.06)

        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(region_w, x + w + pad)
        y2 = min(region_h, y + h + pad)

        crop = region[y1:y2, x1:x2]

        if is_empty_image(crop):
            continue

        if not has_type_icon_symbol(crop):
            continue

        candidates.append({
            "index": 0,
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
            "image": crop,
            "hasSymbol": True,
            "area": area,
        })

    # Sort left to right.
    candidates.sort(key=lambda c: c["x"])

    # Remove overlapping duplicates.
    selected = []
    
    for candidate in candidates:
        if any(type_icon_boxes_overlap(candidate, old) for old in selected):
            continue
        
        candidate_center_x = candidate["x"] + candidate["width"] / 2

        if any(
        abs(candidate_center_x - (old["x"] + old["width"] / 2)) < region_w * 0.12
        for old in selected
        ):
            continue

        selected.append(candidate)

        if len(selected) == 2:
            break

    selected.sort(key=lambda c: c["x"])

    for index, item in enumerate(selected, start=1):
        item["index"] = index

    if not selected:
        fallback_region = extract_fallback_type_icon_region(slot_image)
        selected = add_fallback_type_icon_crops(fallback_region, selected)

    return selected

def is_real_type_icon_crop(icon_crop):
    if is_empty_image(icon_crop):
        return False

    cv2, _np = load_cv_dependencies()

    hsv = cv2.cvtColor(icon_crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    red_card = ((hue <= 10) | (hue >= 160)) & (saturation > 45) & (value > 45)
    white_symbol = (saturation < 110) & (value > 95)
    black_bg = value < 45

    colored_icon = (~red_card) & (~black_bg) & (saturation > 25) & (value > 45)

    total = max(1, icon_crop.shape[0] * icon_crop.shape[1])
    colored_ratio = colored_icon.sum() / total
    white_ratio = white_symbol.sum() / total
    red_ratio = red_card.sum() / total
    black_ratio = black_bg.sum() / total

    # Reject empty card/background crops.
    if red_ratio > 0.55:
        return False

    # Reject right-side black/background crops.
    if black_ratio > 0.35:
        return False

    # Keep actual type icons.
    return colored_ratio >= 0.22 and white_ratio >= 0.025


# Splits the combined type area into two fixed icon boxes.
def crop_two_type_icons_from_region(icon_region):
    return crop_type_icon_boxes_by_square_detection(icon_region)


# Finds colored type icon boxes inside an already-cropped type region.
def crop_type_icon_boxes_by_square_detection(icon_region):
    if is_empty_image(icon_region):
        return []

    cv2, np = load_cv_dependencies()

    region_h, region_w = icon_region.shape[:2]

    hsv = cv2.cvtColor(icon_region, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # Remove the red card background while keeping colored type boxes.
    red_background = (
        ((hue <= 12) | (hue >= 160))
        & (saturation > 45)
        & (value > 45)
    )
    black_background = value < 35
    white_symbol = (saturation < 120) & (value > 90)
    colored_box = (~red_background) & (~black_background) & (saturation > 25) & (value > 45)

    mask = (colored_box | white_symbol).astype("uint8") * 255

    # Connect split white symbol pieces inside the same colored icon square.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if area < region_h * region_w * 0.018:
            continue

        if w < region_w * 0.10 or h < region_h * 0.25:
            continue

        aspect = w / max(1, h)
        if aspect < 0.45 or aspect > 1.75:
            continue

        if w > region_w * 0.55:
            continue

        pad = int(region_h * 0.06)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(region_w, x + w + pad)
        y2 = min(region_h, y + h + pad)

        crop = icon_region[y1:y2, x1:x2]

        if is_empty_image(crop) or not has_type_icon_symbol(crop):
            continue

        candidates.append({
            "index": 0,
            "x": x1,
            "y": y1,
            "width": x2 - x1,
            "height": y2 - y1,
            "image": crop,
            "hasSymbol": True,
            "area": area,
        })

    candidates.sort(key=lambda c: c["x"])

    selected = []
    for candidate in candidates:
        if any(type_icon_boxes_overlap(candidate, old) for old in selected):
            continue

        candidate_center_x = candidate["x"] + candidate["width"] / 2
        if any(
            abs(candidate_center_x - (old["x"] + old["width"] / 2)) < region_w * 0.12
            for old in selected
        ):
            continue

        selected.append(candidate)

        if len(selected) == 2:
            break

    for index, item in enumerate(selected, start=1):
        item["index"] = index

    if not selected:
        selected = add_fallback_type_icon_crops(icon_region, selected)

    return selected


# Adds predictable top-row crops when contour detection misses clipped type icons.
def add_fallback_type_icon_crops(icon_region, selected):
    if is_empty_image(icon_region):
        return selected

    region_h, region_w = icon_region.shape[:2]
    fallback_boxes = [
        {"left": 0.20, "top": 0.02, "width": 0.30, "height": 0.72},
        {"left": 0.48, "top": 0.02, "width": 0.30, "height": 0.72},
        {"left": 0.08, "top": 0.02, "width": 0.34, "height": 0.76},
        {"left": 0.38, "top": 0.02, "width": 0.34, "height": 0.76},
    ]

    completed = list(selected)

    for box in fallback_boxes:
        x = int(region_w * box["left"])
        y = int(region_h * box["top"])
        w = int(region_w * box["width"])
        h = int(region_h * box["height"])

        x2 = min(region_w, x + w)
        y2 = min(region_h, y + h)
        crop = icon_region[y:y2, x:x2]

        if is_empty_image(crop):
            continue

        candidate = {
            "index": 0,
            "x": x,
            "y": y,
            "width": x2 - x,
            "height": y2 - y,
            "image": crop,
            "hasSymbol": has_type_icon_symbol(crop),
            "area": (x2 - x) * (y2 - y),
        }

        if not candidate["hasSymbol"] or not is_real_type_icon_crop(crop):
            continue

        if any(type_icon_boxes_overlap(candidate, old) for old in completed):
            continue

        completed.append(candidate)

        if len(completed) == 2:
            break

    completed.sort(key=lambda c: c["x"])

    for index, item in enumerate(completed, start=1):
        item["index"] = index

    return completed[:2]


# Detects both type icons from the combined type region.
def detect_types_from_combined_region(icon_region, type_references=None):
    icon_crops = crop_two_type_icons_from_region(icon_region)
    detected_types = []

    for icon_crop in icon_crops:
        detected_type = classify_type_by_template(
            icon_crop["image"],
            type_references or [],
        )

        if detected_type and detected_type not in detected_types:
            detected_types.append(detected_type)

        if len(detected_types) == 2:
            break

    return normalize_detected_type_pair(detected_types)

def classify_type_by_template(icon_crop, type_references):
    if is_empty_image(icon_crop):
        return ""

    features = get_white_symbol_shape_features(icon_crop)
    if features and features["fill"] < 0.16:
        return ""

    color_group = classify_type_color_group(icon_crop)
    shape_guess = classify_type_by_shape(icon_crop, color_group)
    if shape_guess:
        return shape_guess

    candidates = [
        reference for reference in type_references
        if not color_group or reference["type"] in color_group
    ]

    result = classify_white_symbol_against_candidates(icon_crop, candidates)

    if result:
        return result

    return classify_against_all_references(icon_crop, type_references)

def classify_against_all_references(icon_crop, type_references):
    if is_empty_image(icon_crop):
        return ""

    cv2, _np = load_cv_dependencies()

    crop_mask = extract_white_symbol_mask(icon_crop)
    crop_color = extract_type_color_signature(icon_crop)

    if crop_mask is None:
        return ""

    scored_results = []

    for reference in type_references:
        ref_image = reference.get("image")
        ref_mask = reference.get("whiteMask")
        ref_color = reference.get("colorSignature")

        if ref_image is None or ref_mask is None:
            continue

        ref_mask = cv2.resize(ref_mask, (96, 96))

        symbol_difference = cv2.absdiff(crop_mask, ref_mask)

        symbol_score = 1.0 - (
            symbol_difference.sum()
            / (255.0 * symbol_difference.shape[0] * symbol_difference.shape[1])
        )

        color_score = compare_type_color_signatures(crop_color, ref_color)

        orb_score = compute_orb_similarity(icon_crop, ref_image)

        final_score = (
            symbol_score * 0.45
            + color_score * 0.20
            + orb_score * 0.35
        )
        
        if color_score == 0.0 and orb_score == 0.0:
            final_score *= 0.55
        
        scored_results.append({
            "type": reference["type"],
            "score": final_score,
            "symbol": symbol_score,
            "color": color_score,
            "orb": orb_score,
        })

    scored_results.sort(key=lambda item: item["score"], reverse=True)

    if not scored_results:
        return ""

    best = scored_results[0]
    return best["type"] if best["score"] >= 0.35 else ""

def classify_type_color_group(icon_crop):
    if is_empty_image(icon_crop):
        return []

    cv2, np = load_cv_dependencies()

    hsv = cv2.cvtColor(icon_crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    white_symbol = (saturation < 115) & (value > 95)
    red_card = ((hue <= 10) | (hue >= 160)) & (saturation > 45) & (value > 45)
    black_bg = value < 40

    useful = (~white_symbol) & (~red_card) & (~black_bg) & (saturation > 8) & (value > 35)

    if useful.sum() < icon_crop.shape[0] * icon_crop.shape[1] * 0.012:
        return []

    mean_h = float(np.mean(hue[useful]))
    mean_s = float(np.mean(saturation[useful]))
    mean_v = float(np.mean(value[useful]))

    if 42 <= mean_h <= 78 and mean_s > 55:
        return ["grass"]

    if 88 <= mean_h <= 132 and mean_s > 45:
        return ["water", "flying", "dragon"]

    if 8 <= mean_h <= 28 and mean_s > 60:
        return ["ground", "fighting"]

    if 12 <= mean_h <= 42 and mean_s <= 170:
        return ["rock", "dark"]

    if 100 <= mean_h <= 155 and mean_s <= 130 and mean_v <= 145:
        return ["dark"]

    if 128 <= mean_h <= 162 and mean_s > 35:
        return ["ghost", "poison", "fairy"]

    return []


# Classifies clear type symbols by simple geometry before template scoring.
def classify_type_by_shape(icon_crop, color_group):
    if is_empty_image(icon_crop):
        return ""

    features = get_white_symbol_shape_features(icon_crop)
    color_signature = extract_type_color_signature(icon_crop)
    color_group = color_group or []

    if not features:
        return ""

    angle = features["angle"]
    horizontal_angle = min(abs(angle), abs(180 - abs(angle)))
    fill = features["fill"]
    component_count = features["componentCount"]

    if color_signature:
        hue = color_signature["hue"]
        saturation = color_signature["saturation"]
        value = color_signature["value"]

        if (
            38 <= hue <= 62
            and saturation >= 70
            and component_count <= 2
            and 0.24 <= fill <= 0.52
            and 18 <= horizontal_angle <= 62
        ):
            return "fire"

        if (
            42 <= hue <= 70
            and saturation > 145
            and component_count >= 3
            and fill >= 0.54
            and (abs(angle) >= 105 or horizontal_angle >= 35)
        ):
            return "bug"

        if (
            130 <= hue <= 158
            and saturation > 85
            and component_count >= 2
            and fill >= 0.58
            and 70 <= abs(angle) <= 112
        ):
            return "fairy"

        if (
            128 <= hue <= 150
            and saturation > 95
            and value > 150
            and component_count >= 3
            and 0.26 <= fill <= 0.48
            and 58 <= abs(angle) <= 112
        ):
            return "psychic"

        if (
            "dark" in color_group
            and component_count <= 2
            and fill >= 0.70
            and 18 <= horizontal_angle <= 62
        ):
            return "dark"

        if (
            136 <= hue <= 158
            and saturation > 95
            and component_count <= 2
            and fill >= 0.50
            and (58 <= abs(angle) <= 150)
        ):
            return "ghost"

        if (
            38 <= hue <= 65
            and 80 <= saturation <= 175
            and value >= 120
            and component_count <= 2
            and fill >= 0.72
            and 20 <= horizontal_angle <= 58
        ):
            return "dark"

        if (
            95 <= hue <= 110
            and value > 190
            and component_count >= 2
            and 0.46 <= fill <= 0.64
            and horizontal_angle <= 18
        ):
            return "rock"

        if (
            108 <= hue <= 125
            and value < 185
            and component_count >= 3
            and 0.44 <= fill <= 0.64
            and horizontal_angle <= 18
        ):
            return "steel"

        if (
            62 <= hue <= 84
            and component_count >= 3
            and 0.30 <= fill <= 0.46
            and horizontal_angle <= 18
        ):
            return "ground"

        if (
            94 <= hue <= 104
            and saturation > 135
            and component_count >= 2
            and 0.42 <= fill <= 0.50
            and 30 <= horizontal_angle <= 52
        ):
            return "dragon"

        if (
            96 <= hue <= 111
            and component_count <= 2
            and 0.40 <= fill <= 0.56
            and 30 <= horizontal_angle <= 58
        ):
            return "flying"

        if (
            18 <= hue <= 35
            and saturation > 150
            and component_count <= 3
            and 0.44 <= fill <= 0.66
            and 70 <= abs(angle) <= 112
        ):
            return "fighting"

        if (
            92 <= hue <= 112
            and saturation > 120
            and component_count <= 3
            and fill >= 0.45
            and 75 <= abs(angle) <= 115
        ):
            return "steel"

        if (
            88 <= hue <= 102
            and saturation > 120
            and component_count <= 2
            and 0.36 <= fill <= 0.48
            and 75 <= abs(angle) <= 115
        ):
            return "ice"

        if (
            135 <= hue <= 155
            and saturation > 110
            and fill >= 0.64
            and component_count <= 1
            and 24 <= horizontal_angle <= 56
        ):
            return "flying"

        if (
            136 <= hue <= 154
            and 0.34 <= fill <= 0.52
            and component_count <= 3
            and 72 <= abs(angle) <= 112
        ):
            return "ghost"

        if (
            100 <= hue <= 125
            and saturation > 130
            and component_count >= 3
            and 0.26 <= fill <= 0.44
            and horizontal_angle <= 18
        ):
            return "psychic"

        if (
            145 <= hue <= 165
            and saturation > 80
            and fill >= 0.55
            and component_count <= 2
        ):
            return "fairy"

        if (
            88 <= hue <= 128
            and fill >= 0.55
            and component_count <= 2
            and 35 <= abs(angle) <= 145
        ):
            return "flying"

        if (
            88 <= hue <= 130
            and component_count <= 2
            and 0.28 <= fill <= 0.52
            and 42 <= abs(angle) <= 78
        ):
            return "dragon"

        if (
            34 <= hue <= 60
            and component_count >= 3
            and 0.28 <= fill <= 0.48
            and horizontal_angle <= 25
        ):
            return "ground"

        if (
            128 <= hue <= 155
            and component_count >= 3
            and 0.26 <= fill <= 0.58
            and horizontal_angle <= 25
        ):
            return "poison"

        if (
            45 <= hue <= 78
            and saturation > 140
            and component_count >= 3
            and fill >= 0.48
            and horizontal_angle <= 30
        ):
            return "fairy"

        if (
            118 <= hue <= 155
            and component_count <= 2
            and 0.20 <= fill <= 0.38
            and 25 <= horizontal_angle <= 60
        ):
            return "fighting"

        if 10 <= hue <= 35 and 45 <= saturation <= 170 and fill >= 0.68:
            return "rock"

        if (
            105 <= hue <= 155
            and saturation <= 135
            and value <= 150
            and component_count <= 3
            and 0.25 <= fill <= 0.55
            and horizontal_angle <= 25
        ):
            return "dark"

    if (
        not color_group
        and component_count <= 2
        and fill >= 0.62
        and (
            70 <= abs(angle) <= 112
            or 18 <= horizontal_angle <= 38
        )
    ):
        return "dark"

    if (
        not color_group
        and component_count <= 2
        and 0.28 <= fill <= 0.55
        and 25 <= horizontal_angle <= 55
    ):
        return "fire"

    if fill >= 0.76 and not color_group:
        return "rock"

    if any(detected_type in color_group for detected_type in ("water", "flying", "dragon")):
        if component_count >= 3 and fill >= 0.36 and horizontal_angle <= 25:
            return "dragon"

        if 20 <= horizontal_angle <= 70 and 0.22 <= fill <= 0.48:
            return "flying"

        if 65 <= abs(angle) <= 115 and component_count <= 2:
            return "water"

    if any(detected_type in color_group for detected_type in ("ground", "fighting")):
        if angle >= 115 or angle <= -115 or fill >= 0.50:
            return "fighting"

        return "ground"

    return ""


# Extracts simple shape measurements from the white symbol in a type icon.
def get_white_symbol_shape_features(icon_crop):
    if is_empty_image(icon_crop):
        return None

    cv2, np = load_cv_dependencies()
    mask = extract_white_symbol_mask(icon_crop)

    if mask is None:
        return None

    white_pixels = cv2.findNonZero(mask)
    if white_pixels is None:
        return None

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    component_count = len([
        contour
        for contour in contours
        if cv2.contourArea(contour) > 20
    ])

    points = white_pixels.reshape(-1, 2).astype(np.float32)
    _mean, eigenvectors = cv2.PCACompute(points, mean=None, maxComponents=2)
    angle = float(np.degrees(np.arctan2(eigenvectors[0][1], eigenvectors[0][0])))

    return {
        "angle": angle,
        "fill": cv2.countNonZero(mask) / max(1, mask.shape[0] * mask.shape[1]),
        "componentCount": component_count,
    }

def classify_white_symbol_against_candidates(icon_crop, candidates):
    if is_empty_image(icon_crop) or not candidates:
        return ""

    cv2, np = load_cv_dependencies()

    crop_mask = extract_white_symbol_mask(icon_crop)
    crop_color = extract_type_color_signature(icon_crop)

    if crop_mask is None or crop_color is None:
        return ""

    best_type = ""
    best_score = -1.0

    for reference in candidates:
        ref_mask = reference.get("whiteMask")
        ref_image = reference.get("image")

        if ref_mask is None or ref_image is None:
            continue

        ref_mask = cv2.resize(ref_mask, (96, 96), interpolation=cv2.INTER_AREA)

        symbol_difference = cv2.absdiff(crop_mask, ref_mask)
        symbol_score = 1.0 - (
            symbol_difference.sum()
            / (255.0 * symbol_difference.shape[0] * symbol_difference.shape[1])
        )

        ref_color = reference.get("colorSignature") or extract_type_color_signature(ref_image)
        color_score = compare_type_color_signatures(crop_color, ref_color)

        final_score = (symbol_score * 0.75) + (color_score * 0.25)

        if final_score > best_score:
            best_score = final_score
            best_type = reference["type"]

    return best_type if best_score >= 0.50 else ""

def extract_type_color_signature(icon_crop):
    if is_empty_image(icon_crop):
        return None

    cv2, np = load_cv_dependencies()

    hsv = cv2.cvtColor(icon_crop, cv2.COLOR_BGR2HSV)

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    white_symbol = (saturation < 100) & (value > 105)
    red_card = ((hue <= 10) | (hue >= 160)) & (saturation > 45) & (value > 45)
    black_bg = value < 40

    useful = (
        (~white_symbol)
        & (~red_card)
        & (~black_bg)
        & (saturation > 20)
        & (value > 45)
    )

    if useful.sum() < icon_crop.shape[0] * icon_crop.shape[1] * 0.015:
        return None

    return {
        "hue": float(np.mean(hue[useful])),
        "saturation": float(np.mean(saturation[useful])),
        "value": float(np.mean(value[useful])),
    }
    
def compare_type_color_signatures(first_color, second_color):
    if not first_color or not second_color:
        return 0.0

    hue_diff = abs(first_color["hue"] - second_color["hue"])
    hue_diff = min(hue_diff, 180 - hue_diff)

    sat_diff = abs(first_color["saturation"] - second_color["saturation"])
    val_diff = abs(first_color["value"] - second_color["value"])

    hue_score = max(0.0, 1.0 - (hue_diff / 35.0))
    sat_score = max(0.0, 1.0 - (sat_diff / 120.0))
    val_score = max(0.0, 1.0 - (val_diff / 120.0))

    return (hue_score * 0.60) + (sat_score * 0.20) + (val_score * 0.20)

# These are kept so the existing debug/output code still works.
def detect_types_from_fixed_icon_boxes(icon_region, type_references=None):
    return detect_types_from_combined_region(icon_region, type_references or [])


def detect_types_from_symbol_squares(icon_region, type_references=None):
    return detect_types_from_combined_region(icon_region, type_references or [])


def crop_fixed_type_icon_boxes(icon_region):
    return crop_two_type_icons_from_region(icon_region)


def crop_debug_type_icon_boxes(icon_region):
    return crop_two_type_icons_from_region(icon_region)


def select_detected_types_from_methods(method_results):
    component_types = method_results.get("component", [])
    if component_types:
        return normalize_detected_type_pair(component_types[:2])

    combined_types = method_results.get("combinedTemplate", method_results.get("template", []))

    if len(combined_types) >= 2:
        return normalize_detected_type_pair(combined_types[:2])

    if len(combined_types) == 1:
        return combined_types

    fixed_types = method_results.get("fixedSlot", method_results.get("fixed", []))
    symbol_square_types = method_results.get("symbolSquare", [])

    if fixed_types and symbol_square_types:
        agreed_types = [
            detected_type
            for detected_type in fixed_types
            if detected_type in symbol_square_types
        ]
        if agreed_types:
            return normalize_detected_type_pair(agreed_types[:2])

    if fixed_types == ["fire"]:
        return fixed_types

    return []


# Checks whether a crop contains enough white symbol pixels to be a type icon.
def has_type_icon_symbol(icon_crop):
    if is_empty_image(icon_crop):
        return False

    cv2, _np = load_cv_dependencies()

    hsv_crop = cv2.cvtColor(icon_crop, cv2.COLOR_BGR2HSV)
    saturation = hsv_crop[:, :, 1]
    value = hsv_crop[:, :, 2]

    white_symbol = (saturation < 110) & (value > 95)

    white_ratio = white_symbol.sum() / max(
        1,
        icon_crop.shape[0] * icon_crop.shape[1]
    )

    return white_ratio >= 0.018


def should_accept_type_icon_crop(icon_crop):
    if is_empty_image(icon_crop):
        return False

    quality = score_type_icon_crop_quality(icon_crop)
    return quality >= 0.48


def score_type_icon_crop_quality(icon_crop):
    if is_empty_image(icon_crop):
        return 0.0

    cv2, np = load_cv_dependencies()
    hsv = cv2.cvtColor(icon_crop, cv2.COLOR_BGR2HSV)

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    white_symbol = (saturation < 115) & (value > 95)
    red_card = ((hue <= 10) | (hue >= 160)) & (saturation > 45) & (value > 45)
    dark_bg = value < 40
    colored_icon = (~red_card) & (~dark_bg) & (saturation > 25) & (value > 45)

    total = max(1, icon_crop.shape[0] * icon_crop.shape[1])
    white_ratio = white_symbol.sum() / total
    colored_ratio = colored_icon.sum() / total
    red_ratio = red_card.sum() / total

    if red_ratio > 0.55:
        return 0.0

    if white_ratio < 0.018:
        return 0.0

    if colored_ratio < 0.15:
        return 0.0

    height, width = icon_crop.shape[:2]
    aspect = width / max(1, height)
    aspect_score = max(0.0, 1.0 - min(1.0, abs(1.0 - aspect)))
    white_score = min(0.35, white_ratio * 5.0)
    color_score = min(0.45, colored_ratio * 1.5)
    red_penalty = min(0.20, red_ratio * 0.35)

    return max(0.0, min(1.0, 0.10 + white_score + color_score + (aspect_score * 0.20) - red_penalty))


# Fixes known color collisions that need paired-icon context.
def normalize_detected_type_pair(detected_types):
    detected_type_set = set(detected_types)

    if detected_type_set == {"ground", "poison"}:
        return ["fighting", "poison"]

    if detected_type_set == {"grass", "water"}:
        return ["grass", "poison"]

    if detected_type_set == {"fire", "dark"}:
        return ["fire", "dark"]

    if detected_type_set == {"dark", "steel"}:
        return ["dark", "steel"]

    if detected_type_set == {"dragon", "poison"}:
        return ["dragon", "ghost"]

    if detected_type_set == {"water", "fighting"}:
        return ["psychic", "fighting"]

    if detected_type_set == {"water", "fighting", "poison"}:
        return ["psychic", "fighting"]

    if detected_type_set == {"psychic", "fighting", "poison"}:
        return ["psychic", "fighting"]

    return detected_types

# Keeps references that match at least one detected type.
def filter_references_by_types(references, detected_types):
    if not detected_types:
        return references

    detected_type_set = set(detected_types)
    if len(detected_type_set) > 1:
        exact_references = [
            reference
            for reference in references
            if detected_type_set.issubset(set(reference.get("types", [])))
        ]
        if exact_references:
            return exact_references

    filtered_references = [
        reference
        for reference in references
        if detected_type_set.intersection(reference.get("types", []))
    ]

    return filtered_references or references


# Loads sprite metadata so reference filenames map to Pokemon names.
def load_reference_metadata(metadata_path=REFERENCE_METADATA_PATH):
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        logger.warning("Reference metadata file was not found: %s", metadata_path)
        return {}

    import json

    records = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {
        record["local_filename"]: record
        for record in records
        if record.get("local_filename")
    }


# Resizes an image to a target width while preserving its shape.
def resize_to_width(image, target_width):
    if is_empty_image(image):
        raise ComputerVisionError("Cannot resize an empty image crop.")

    cv2, _np = load_cv_dependencies()
    height, width = image.shape[:2]
    if width == target_width:
        return image

    scale = target_width / width
    target_height = max(1, int(round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


# Converts an image to normalized grayscale for matching.
def normalize_grayscale(image):
    if is_empty_image(image):
        raise ComputerVisionError("Cannot normalize an empty image crop.")

    cv2, _np = load_cv_dependencies()
    if len(image.shape) == 2:
        grayscale = image
    else:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(grayscale, (3, 3), 0)
    return cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)


# Builds ORB feature descriptors for a sprite image.
def build_feature_descriptors(image):
    if is_empty_image(image):
        return [], None

    cv2, _np = load_cv_dependencies()
    match_image = preprocess_for_matching(image)
    orb = cv2.ORB_create(nfeatures=350, scaleFactor=1.2, nlevels=8)
    keypoints, descriptors = orb.detectAndCompute(match_image, None)
    return keypoints or [], descriptors


# Prepares an image for template matching.
def preprocess_for_matching(image):
    resized = resize_to_width(image, MATCH_REGION_WIDTH)
    return normalize_grayscale(resized)


# Prepares a color image for template matching.
def preprocess_color_for_matching(image):
    resized = resize_to_width(image, MATCH_REGION_WIDTH)
    return normalize_color(resized)


# Normalizes color channels so lighting differences have less impact.
def normalize_color(image):
    if is_empty_image(image):
        raise ComputerVisionError("Cannot normalize an empty color image crop.")

    cv2, _np = load_cv_dependencies()
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    channels = cv2.split(lab_image)
    normalized_channels = [
        cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
        for channel in channels
    ]
    return cv2.merge(normalized_channels)


# Extracts the white symbol from a type icon image.
def extract_white_symbol_mask(image):
    if is_empty_image(image):
        return None

    cv2, np = load_cv_dependencies()

    if len(image.shape) == 2:
        gray = image
        _, mask = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]

        mask = (
            (saturation < 95)
            & (value > 105)
        ).astype("uint8") * 255

    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((4, 4), np.uint8),
    )

    points = cv2.findNonZero(mask)

    if points is not None:
        x, y, w, h = cv2.boundingRect(points)

        pad = 8
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(mask.shape[1], x + w + pad)
        y2 = min(mask.shape[0], y + h + pad)

        mask = mask[y1:y2, x1:x2]

    return cv2.resize(mask, (96, 96), interpolation=cv2.INTER_AREA)


# Resizes and normalizes a reference template before it is compared.
def resize_template_for_matching(reference_image, scale, normalize=True):
    cv2, _np = load_cv_dependencies()
    reference_height, reference_width = reference_image.shape[:2]
    resized_width = int(reference_width * scale)
    resized_height = int(reference_height * scale)
    if resized_width < 8 or resized_height < 8:
        return None

    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized_reference = cv2.resize(reference_image, (resized_width, resized_height), interpolation=interpolation)
    if not normalize:
        return resized_reference

    if len(resized_reference.shape) == 2:
        return normalize_grayscale(resized_reference)
    return normalize_color(resized_reference)


# Builds scaled templates once so detection does less work per request.
def build_scaled_templates(reference_image, normalize=True, scales=MATCH_SCALES):
    templates = []
    for scale in scales:
        resized_reference = resize_template_for_matching(reference_image, scale, normalize=normalize)
        if resized_reference is not None:
            templates.append(resized_reference)

    return templates


# Computes the best template-match score for one prepared reference.
def score_reference(slot_match_image, reference_match_image):
    cv2, _np = load_cv_dependencies()
    slot_height, slot_width = slot_match_image.shape[:2]

    best_score = 0.0
    for scale in MATCH_SCALES:
        resized_reference = resize_template_for_matching(reference_match_image, scale)
        if resized_reference is None:
            continue

        resized_height, resized_width = resized_reference.shape[:2]
        if resized_width > slot_width or resized_height > slot_height:
            continue

        result = cv2.matchTemplate(slot_match_image, resized_reference, cv2.TM_CCOEFF_NORMED)
        _min_value, max_value, _min_location, _max_location = cv2.minMaxLoc(result)
        best_score = max(best_score, float(max_value))

    return best_score


# Computes a color template-match score for one prepared reference.
def score_color_reference(slot_match_image, reference_match_image):
    cv2, _np = load_cv_dependencies()
    slot_height, slot_width = slot_match_image.shape[:2]

    best_score = 0.0
    for scale in MATCH_SCALES:
        resized_reference = resize_template_for_matching(reference_match_image, scale, normalize=False)
        if resized_reference is None:
            continue

        resized_height, resized_width = resized_reference.shape[:2]
        if resized_width > slot_width or resized_height > slot_height:
            continue

        result = cv2.matchTemplate(slot_match_image, resized_reference, cv2.TM_CCOEFF_NORMED)
        _min_value, max_value, _min_location, _max_location = cv2.minMaxLoc(result)
        best_score = max(best_score, float(max_value))

    return best_score


# Scores already scaled reference templates against one slot crop.
def score_prepared_templates(slot_match_image, templates):
    cv2, _np = load_cv_dependencies()
    slot_height, slot_width = slot_match_image.shape[:2]

    best_score = 0.0
    for template in templates:
        template_height, template_width = template.shape[:2]
        if template_width > slot_width or template_height > slot_height:
            continue

        result = cv2.matchTemplate(slot_match_image, template, cv2.TM_CCOEFF_NORMED)
        _min_value, max_value, _min_location, _max_location = cv2.minMaxLoc(result)
        best_score = max(best_score, float(max_value))

    return best_score


# Scores a crop against one reference using ORB feature matches.
def score_feature_reference(slot_descriptors, slot_keypoint_count, reference):
    reference_descriptors = reference.get("featureDescriptors")
    reference_keypoint_count = reference.get("featureKeypointCount", 0)
    if slot_descriptors is None or reference_descriptors is None:
        return 0.0
    if slot_keypoint_count < 4 or reference_keypoint_count < 4:
        return 0.0

    cv2, _np = load_cv_dependencies()
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(slot_descriptors, reference_descriptors)
    if not matches:
        return 0.0

    good_matches = [match for match in matches if match.distance <= 72]
    if not good_matches:
        return 0.0

    coverage = len(good_matches) / max(1, min(slot_keypoint_count, reference_keypoint_count))
    average_distance = sum(match.distance for match in good_matches) / len(good_matches)
    distance_quality = max(0.0, 1.0 - (average_distance / 96.0))
    return min(1.0, coverage * distance_quality * FEATURE_SCORE_WEIGHT)


# Returns a stable fallback reference when type filtering narrows to one species.
def select_type_confirmed_reference(filtered_references, best_reference):
    if best_reference:
        return best_reference

    normal_references = [
        reference
        for reference in filtered_references
        if not reference.get("isShiny")
    ] or filtered_references
    default_form_references = [
        reference
        for reference in normal_references
        if "female" not in reference.get("form", "").lower()
    ]
    return (default_form_references or normal_references)[0]


# Finds the closest Pokemon reference image for one cropped slot.
def detect_pokemon_from_slot(slot_image, references, type_references=None):
    if not references or is_empty_image(slot_image):
        return {"pokemonName": "unknown", "confidence": 0.0, "referenceImage": ""}

    pokemon_region = extract_opponent_pokemon_region(slot_image)
    detected_types = detect_types_from_opponent_slot(slot_image, type_references=type_references)
    return detect_pokemon_from_region(pokemon_region, references, detected_types=detected_types)


# Finds the closest Pokemon reference image for one extracted sprite crop.
def detect_pokemon_from_region(pokemon_region, references, detected_types=None):
    if not references or is_empty_image(pokemon_region):
        return {"pokemonName": "unknown", "confidence": 0.0, "referenceImage": ""}

    if is_empty_image(pokemon_region):
        return {"pokemonName": "unknown", "confidence": 0.0, "referenceImage": ""}

    filtered_references = filter_references_by_types(references, detected_types or [])
    slot_match_image = preprocess_for_matching(pokemon_region)
    slot_color_match_image = preprocess_color_for_matching(pokemon_region)
    slot_keypoints, slot_descriptors = build_feature_descriptors(pokemon_region)
    best_match = {"pokemonName": "unknown", "confidence": 0.0, "referenceImage": ""}
    best_reference = None

    grayscale_candidates = []
    for reference in filtered_references:
        grayscale_score = score_prepared_templates(slot_match_image, reference.get("grayscaleTemplates", []))
        grayscale_candidates.append((grayscale_score, reference))
        if grayscale_score > best_match["confidence"]:
            best_match = {
                "pokemonName": reference["name"],
                "confidence": round(grayscale_score, 4),
                "referenceImage": reference["path"],
                "referenceTypes": reference.get("types", []),
            }
            best_reference = reference

    grayscale_candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    reranked_candidates = []
    for grayscale_score, reference in grayscale_candidates[:COLOR_RERANK_LIMIT]:
        color_score = score_prepared_templates(slot_color_match_image, reference.get("colorTemplates", []))
        score = max(grayscale_score, color_score)
        reranked_candidates.append((score, reference))
        if score > best_match["confidence"]:
            best_match = {
                "pokemonName": reference["name"],
                "confidence": round(score, 4),
                "referenceImage": reference["path"],
                "referenceTypes": reference.get("types", []),
            }
            best_reference = reference

    reranked_candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    for current_score, reference in reranked_candidates[:FEATURE_RERANK_LIMIT]:
        feature_score = score_feature_reference(slot_descriptors, len(slot_keypoints), reference)
        score = max(current_score, feature_score)
        if score > best_match["confidence"]:
            best_match = {
                "pokemonName": reference["name"],
                "confidence": round(score, 4),
                "referenceImage": reference["path"],
                "referenceTypes": reference.get("types", []),
            }
            best_reference = reference

    if best_match["confidence"] < 0.35:
        partial_candidates = []
        for reference in filtered_references:
            standard_score = max(
                score_prepared_templates(slot_match_image, reference.get("grayscaleTemplates", [])),
                score_prepared_templates(slot_color_match_image, reference.get("colorTemplates", [])),
            )
            grayscale_score = score_prepared_templates(
                slot_match_image,
                reference.get("partialGrayscaleTemplates", []),
            )
            color_score = score_prepared_templates(
                slot_color_match_image,
                reference.get("partialColorTemplates", []),
            )
            partial_score = max(grayscale_score, color_score)
            score = (partial_score * 0.70) + (standard_score * 0.30)
            partial_candidates.append((score, reference))

        partial_candidates.sort(key=lambda candidate: candidate[0], reverse=True)
        for score, reference in partial_candidates[:FEATURE_RERANK_LIMIT]:
            if score > best_match["confidence"]:
                best_match = {
                    "pokemonName": reference["name"],
                    "confidence": round(score, 4),
                    "referenceImage": reference["path"],
                    "referenceTypes": reference.get("types", []),
                }
                best_reference = reference

    preferred_reference = select_close_visual_preference(
        filtered_references,
        detected_types or [],
        best_reference,
        slot_match_image,
        slot_color_match_image,
    )
    if preferred_reference and preferred_reference is not best_reference:
        preferred_score = score_reference_for_visual_preference(
            preferred_reference,
            slot_match_image,
            slot_color_match_image,
        )
        best_match = {
            "pokemonName": preferred_reference["name"],
            "confidence": round(preferred_score, 4),
            "referenceImage": preferred_reference["path"],
            "referenceTypes": preferred_reference.get("types", []),
        }
        best_reference = preferred_reference

    visual_rescue_match, visual_rescue_reference = detect_visual_rescue_match(
        references,
        best_match,
        slot_match_image,
        slot_color_match_image,
    )
    if visual_rescue_match:
        best_match = visual_rescue_match
        best_reference = visual_rescue_reference

    if best_match["confidence"] < MIN_DETECTION_CONFIDENCE:
        species_names = {
            reference.get("species", reference["name"])
            for reference in filtered_references
        }
        has_narrow_candidate_set = detected_types and len(filtered_references) <= 8 and len(species_names) <= 3
        if has_narrow_candidate_set and (len(species_names) == 1 or best_match["confidence"] >= 0.35):
            fallback_reference = select_type_confirmed_reference(filtered_references, best_reference)
            type_floor = TYPE_CONFIRMED_CONFIDENCE if len(species_names) == 1 else TYPE_SUPPORTED_CONFIDENCE
            confidence = max(best_match["confidence"], type_floor)

            return {
                "pokemonName": fallback_reference["name"],
                "confidence": round(confidence, 4),
                "referenceImage": fallback_reference["path"],
                "referenceTypes": fallback_reference.get("types", []),
                "matchReason": "type-confirmed",
            }

        return {
            "pokemonName": "unknown",
            "confidence": best_match["confidence"],
            "referenceImage": best_match["referenceImage"],
        }

    return best_match


def detect_visual_rescue_match(references, best_match, slot_match_image, slot_color_match_image):
    if best_match.get("confidence", 0) >= 0.70:
        return None, None

    rescue_candidates = [
        reference
        for reference in references
        if Path(reference.get("path", "")).parent == EXTRA_REFERENCE_IMAGE_DIR
    ]
    if not rescue_candidates:
        return None, None

    scored_candidates = []
    for reference in rescue_candidates:
        score = score_reference_for_visual_preference(
            reference,
            slot_match_image,
            slot_color_match_image,
        )
        scored_candidates.append((score, reference))

    scored_candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    if not scored_candidates:
        return None, None

    score, reference = scored_candidates[0]
    if score < 0.82:
        return None, None

    current_confidence = float(best_match.get("confidence", 0) or 0)
    if score < current_confidence + 0.08:
        return None, None

    return {
        "pokemonName": reference["name"],
        "confidence": round(score, 4),
        "referenceImage": reference["path"],
        "referenceTypes": reference.get("types", []),
        "matchReason": "guided-visual-rescue",
    }, reference


def score_reference_for_visual_preference(reference, slot_match_image, slot_color_match_image):
    standard_score = max(
        score_prepared_templates(slot_match_image, reference.get("grayscaleTemplates", [])),
        score_prepared_templates(slot_color_match_image, reference.get("colorTemplates", [])),
    )
    partial_score = max(
        score_prepared_templates(slot_match_image, reference.get("partialGrayscaleTemplates", [])),
        score_prepared_templates(slot_color_match_image, reference.get("partialColorTemplates", [])),
    )
    return max(standard_score, (partial_score * 0.70) + (standard_score * 0.30))


def select_close_visual_preference(
    filtered_references,
    detected_types,
    best_reference,
    slot_match_image,
    slot_color_match_image,
):
    if not best_reference:
        return None

    detected_type_set = set(detected_types)
    best_score = score_reference_for_visual_preference(
        best_reference,
        slot_match_image,
        slot_color_match_image,
    )

    preference_names = []
    if detected_type_set == {"normal", "psychic"} and best_reference["name"] == "Wyrdeer":
        preference_names = ["Farigiraf"]
    elif detected_type_set == {"fairy"} and best_reference["name"] != "Sylveon":
        preference_names = ["Sylveon"]
    elif detected_type_set == {"ghost"} and best_reference["name"] == "Sableye Mega":
        preference_names = ["Aerodactyl", "Aerodactyl Mega"]
    elif detected_type_set == {"flying", "ground"} and best_reference["name"] == "Gliscor":
        preference_names = ["Garchomp"]
    elif detected_type_set == {"grass"} and best_reference["name"] == "Meganium":
        preference_names = ["Tsareena"]

    for preferred_name in preference_names:
        candidates = [
            reference
            for reference in filtered_references
            if reference["name"] == preferred_name
        ]
        if not candidates:
            continue

        preferred_reference = max(
            candidates,
            key=lambda reference: score_reference_for_visual_preference(
                reference,
                slot_match_image,
                slot_color_match_image,
            ),
        )
        preferred_score = score_reference_for_visual_preference(
            preferred_reference,
            slot_match_image,
            slot_color_match_image,
        )
        if preferred_score >= best_score - 0.055:
            return preferred_reference

    return None

def compute_orb_similarity(first_image, second_image):
    if is_empty_image(first_image) or is_empty_image(second_image):
        return 0.0

    cv2, _np = load_cv_dependencies()

    gray1 = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(second_image, cv2.COLOR_BGR2GRAY)

    gray1 = cv2.resize(gray1, (96, 96))
    gray2 = cv2.resize(gray2, (96, 96))

    orb = cv2.ORB_create(
        nfeatures=200,
        scaleFactor=1.2,
        nlevels=8,
    )

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    if des1 is None or des2 is None:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = matcher.match(des1, des2)

    if not matches:
        return 0.0

    matches = sorted(matches, key=lambda m: m.distance)

    good_matches = [
        match for match in matches
        if match.distance < 55
    ]

    return min(1.0, len(good_matches) / 35.0)


# Detects all guided opponent team slots with a cached default detector.
def detect_opponent_team(image_path, save_debug=True):
    global _DEFAULT_DETECTOR

    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = OpponentTeamDetector()

    return _DEFAULT_DETECTOR.detect_team(image_path, save_debug=save_debug)


# Detects only opponent Pokemon type icons from one uploaded image.
def detect_opponent_team_types(image_path, save_debug=True):
    global _DEFAULT_DETECTOR

    if _DEFAULT_DETECTOR is None:
        _DEFAULT_DETECTOR = OpponentTeamDetector()

    return _DEFAULT_DETECTOR.detect_team_types(image_path, save_debug=save_debug)

# Enlarges small type icon regions so debug crops and template matching have enough pixels.
def enlarge_small_type_icon_region(icon_region):
    if is_empty_image(icon_region):
        return icon_region

    height, width = icon_region.shape[:2]

    if height >= MIN_TYPE_ICON_REGION_HEIGHT:
        return icon_region

    cv2, _np = load_cv_dependencies()

    scale = MIN_TYPE_ICON_REGION_HEIGHT / max(1, height)
    resized_width = max(1, int(round(width * scale)))

    return cv2.resize(
        icon_region,
        (resized_width, MIN_TYPE_ICON_REGION_HEIGHT),
        interpolation=cv2.INTER_CUBIC,
    )

# Detects type icons for each opponent slot and returns JSON-safe results.
def detect_opponent_team_types_with_references(
    image_path,
    type_references=None,
    debug_dir=OPPONENT_DEBUG_CROP_DIR,
    save_debug=True,
):
    image_path = Path(image_path)
    quality = assess_opponent_image_quality(image_path)
    crops = crop_opponent_team_slots(image_path, save_debug=save_debug)
    debug_dir = Path(debug_dir)

    detected_slots = []
    for crop in crops:
        type_icon_region = extract_type_icon_region(crop["image"])
        type_method_results = detect_type_method_results(crop["image"], type_references=type_references or [])
        debug_type_icon_crop_path = ""
        debug_type_icon_crop_paths = []

        if save_debug and not is_empty_image(type_icon_region):
            cv2, _np = load_cv_dependencies()
            debug_type_icon_region = enlarge_small_type_icon_region(type_icon_region)
            debug_type_icon_crop_path = str(
                debug_dir / f"{image_path.stem}-type-only-icons-{crop['position']}.jpg"
            )
            if not cv2.imwrite(debug_type_icon_crop_path, debug_type_icon_region):
                raise ComputerVisionError(f"Could not write type icon crop: {debug_type_icon_crop_path}")

            for icon_crop in crop_type_icons_from_slot(crop["image"]):
                debug_icon_path = str(
                    debug_dir / f"{image_path.stem}-type-only-icon-{crop['position']}-{icon_crop['index']}.jpg"
                )
                if not cv2.imwrite(debug_icon_path, icon_crop["image"]):
                    raise ComputerVisionError(f"Could not write type icon crop: {debug_icon_path}")
                debug_type_icon_crop_paths.append({
                    "path": debug_icon_path,
                    "hasSymbol": bool(icon_crop["hasSymbol"]),
                })

        detected_slots.append({
            "position": crop["position"],
            "types": type_method_results["selected"],
            "typeMethodResults": type_method_results,
            "box": crop["box"],
            "typeIconCropPath": debug_type_icon_crop_path,
            "typeIconCropPaths": debug_type_icon_crop_paths,
        })

    return {
        "image": str(image_path),
        "mode": "opponent-type-only",
        "quality": quality,
        "teamTypes": detected_slots,
    }



# Detects all guided opponent team slots and returns JSON-safe results.
def detect_opponent_team_with_references(
    image_path,
    references,
    type_references=None,
    debug_dir=OPPONENT_DEBUG_CROP_DIR,
    save_debug=True,
):
    image_path = Path(image_path)
    quality = assess_opponent_image_quality(image_path)
    crops = crop_opponent_team_slots(image_path, save_debug=save_debug)
    debug_dir = Path(debug_dir)

    detected_team = []
    for crop in crops:
        pokemon_region = extract_opponent_pokemon_region(crop["image"])
        type_icon_region = extract_type_icon_region(crop["image"])
        type_method_results = detect_type_method_results(crop["image"], type_references=type_references or [])
        detected_types = type_method_results["selected"]
        debug_pokemon_crop_path = ""
        debug_type_icon_crop_path = ""
        debug_type_icon_crop_paths = []
        if save_debug and not is_empty_image(pokemon_region):
            cv2, _np = load_cv_dependencies()
            debug_pokemon_crop_path = str(
                debug_dir / f"{image_path.stem}-opponent-pokemon-{crop['position']}.jpg"
            )
            if not cv2.imwrite(debug_pokemon_crop_path, pokemon_region):
                raise ComputerVisionError(f"Could not write Pokemon crop: {debug_pokemon_crop_path}")

        if save_debug and not is_empty_image(type_icon_region):
            cv2, _np = load_cv_dependencies()
            debug_type_icon_region = enlarge_small_type_icon_region(type_icon_region)
            debug_type_icon_crop_path = str(
                debug_dir / f"{image_path.stem}-opponent-type-icons-{crop['position']}.jpg"
            )
            if not cv2.imwrite(debug_type_icon_crop_path, debug_type_icon_region):
                raise ComputerVisionError(f"Could not write type icon crop: {debug_type_icon_crop_path}")

            for icon_crop in crop_debug_type_icon_boxes(debug_type_icon_region):
                debug_icon_path = str(
                    debug_dir / f"{image_path.stem}-opponent-type-icon-{crop['position']}-{icon_crop['index']}.jpg"
                )
                if not cv2.imwrite(debug_icon_path, icon_crop["image"]):
                    raise ComputerVisionError(f"Could not write type icon crop: {debug_icon_path}")
                debug_type_icon_crop_paths.append({
                    "path": debug_icon_path,
                    "hasSymbol": bool(icon_crop["hasSymbol"]),
                })

        match = detect_pokemon_from_region(pokemon_region, references, detected_types=detected_types)
        detected_team.append({
            "position": crop["position"],
            "pokemonName": match["pokemonName"],
            "confidence": match["confidence"],
            "detectedTypes": detected_types,
            "typeMethodResults": type_method_results,
            "box": crop["box"],
            "debugCropPath": crop["debugCropPath"],
            "debugPokemonCropPath": debug_pokemon_crop_path,
            "debugTypeIconCropPath": debug_type_icon_crop_path,
            "debugTypeIconCropPaths": debug_type_icon_crop_paths,
            "referenceImage": match["referenceImage"],
            "matchReason": match.get("matchReason", "visual-match"),
        })

    return {
        "image": str(image_path),
        "referenceCount": len(references),
        "quality": quality,
        "detectedTeam": detected_team,
    }
def normalize_type_box(box, image_width, image_height, padding_ratio=0.12):
    x, y, w, h = box["x"], box["y"], box["width"], box["height"]

    # Use center of detected object
    cx = x + w / 2
    cy = y + h / 2

    # Force square around the detected tile
    size = max(w, h)

    # Add padding so the full tile is not clipped
    size = size * (1 + padding_ratio)

    new_x = int(cx - size / 2)
    new_y = int(cy - size / 2)
    new_w = int(size)
    new_h = int(size)

    # Clamp to image bounds
    new_x = max(0, min(new_x, image_width - new_w))
    new_y = max(0, min(new_y, image_height - new_h))

    return {
        "x": new_x,
        "y": new_y,
        "width": new_w,
        "height": new_h,
    }
