"""Runs opponent detection on every uploaded opponent image and saves a focused debug report.

Purpose:
- Debug image/card placement
- See why detection worked or failed
- See whether matches are coming from clear references or camera/custom references
- Identify slow parts of the pipeline
- Keep only fields useful for improving accuracy and speed
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND_DIR))

from paths import CV_DEBUG_REPORT_DIR, UPLOAD_DIR
from services.cv_detection_service import detect_opponent_team
from services.cv_service import ComputerVisionError


OUTPUT_DIR = CV_DEBUG_REPORT_DIR
OUTPUT_PATH = OUTPUT_DIR / "uploaded_opponent_detection_debug_report.json"

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


# Tune these after looking at your own scores.
GOOD_CONFIDENCE = 0.35
OK_CONFIDENCE = 0.28
GOOD_GAP = 0.05
WEAK_GAP = 0.025

# Photo-quality guide rails for the guided phone capture.
BEST_MIN_SHORT_EDGE = 1000
BEST_MIN_SHARPNESS = 85.0
USABLE_MIN_SHARPNESS = 45.0
BEST_MIN_CARD_AREA_RATIO = 0.12
USABLE_MIN_CARD_AREA_RATIO = 0.08
MAX_BEST_EXPOSURE_RATIO = 0.06
MAX_USABLE_EXPOSURE_RATIO = 0.18
MAX_BEST_CENTER_DRIFT_RATIO = 0.12
MAX_USABLE_CENTER_DRIFT_RATIO = 0.22
MAX_BEST_SPACING_VARIATION_RATIO = 0.22
MAX_USABLE_SPACING_VARIATION_RATIO = 0.38

def make_json_safe(value):
    """Converts NumPy/OpenCV values into normal JSON-safe Python values."""
    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

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

def safe_float(value):
    """Converts numeric values safely for JSON/report comparisons."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_capture_quality_debug(quality):
    """Turns backend photo metrics into actionable capture guidance."""
    quality = quality or {}
    metrics = quality.get("metrics") or {}

    issues = []
    strengths = []
    guidance = []
    issue_score = 0

    image_width = safe_float(metrics.get("imageWidth"))
    image_height = safe_float(metrics.get("imageHeight"))
    short_edge = min(image_width, image_height) if image_width and image_height else None

    if short_edge is None:
        issues.append("Image dimensions were not available in quality metrics.")
        guidance.append("Check that the upload path is passing the real captured image into detection.")
        issue_score += 2
    elif short_edge < 720:
        issues.append(f"Capture resolution is low; short edge is {short_edge:.0f}px.")
        guidance.append("Use guided camera capture and keep browser zoom off so type clusters stay detailed.")
        issue_score += 2
    elif short_edge < BEST_MIN_SHORT_EDGE:
        issues.append(f"Capture resolution is usable but not ideal; short edge is {short_edge:.0f}px.")
        guidance.append("If the phone supports it, use the rear camera and let the page request the highest camera mode.")
        issue_score += 1
    else:
        strengths.append(f"Capture resolution is strong; short edge is {short_edge:.0f}px.")

    detected_card_count = metrics.get("detectedCardCount")
    if detected_card_count != 6:
        issues.append(f"Detected {detected_card_count} card boxes instead of 6.")
        guidance.append("Retake with all six opponent slots inside the guide boxes and avoid cutting off the top or bottom slot.")
        issue_score += 2
    else:
        strengths.append("All six opponent card boxes were detected.")

    sharpness = safe_float(metrics.get("sharpnessScore"))
    if sharpness is None:
        issues.append("Sharpness metric is missing.")
        issue_score += 1
    elif sharpness < USABLE_MIN_SHARPNESS:
        issues.append(f"Photo is blurry; sharpness score is {sharpness:.1f}.")
        guidance.append("Hold still for a moment after lining up the boxes, then capture after autofocus settles.")
        issue_score += 3
    elif sharpness < BEST_MIN_SHARPNESS:
        issues.append(f"Photo is usable but could be sharper; sharpness score is {sharpness:.1f}.")
        guidance.append("Move slightly closer and keep the phone square to the screen before capture.")
        issue_score += 1
    else:
        strengths.append(f"Sharpness is strong at {sharpness:.1f}.")

    card_area = safe_float(metrics.get("cardAreaRatio"))
    if card_area is not None:
        if card_area < USABLE_MIN_CARD_AREA_RATIO:
            issues.append(f"Cards are too small in frame; card area ratio is {card_area:.3f}.")
            guidance.append("Move the phone closer until the six cards fill more of the guided boxes.")
            issue_score += 2
        elif card_area < BEST_MIN_CARD_AREA_RATIO:
            issues.append(f"Cards are a little small; card area ratio is {card_area:.3f}.")
            guidance.append("Nudge closer so the type cluster has more pixels for matching.")
            issue_score += 1
        else:
            strengths.append(f"Card size in frame is good at {card_area:.3f}.")

    overexposed = safe_float(metrics.get("overexposedRatio")) or 0.0
    underexposed = safe_float(metrics.get("underexposedRatio")) or 0.0
    exposure_ratio = max(overexposed, underexposed)
    if exposure_ratio > MAX_USABLE_EXPOSURE_RATIO:
        issues.append(f"Exposure is hurting detail; worst exposure ratio is {exposure_ratio:.3f}.")
        guidance.append("Reduce glare/brightness, tilt away from reflections, or retake with steadier lighting.")
        issue_score += 3
    elif exposure_ratio > MAX_BEST_EXPOSURE_RATIO:
        issues.append(f"Exposure is usable but not clean; worst exposure ratio is {exposure_ratio:.3f}.")
        guidance.append("Avoid bright glare on the type boxes before capturing.")
        issue_score += 1
    else:
        strengths.append(f"Exposure looks clean; worst exposure ratio is {exposure_ratio:.3f}.")

    center_drift = safe_float(metrics.get("centerDriftRatio"))
    if center_drift is not None:
        if center_drift > MAX_USABLE_CENTER_DRIFT_RATIO:
            issues.append(f"Cards are off-center; center drift ratio is {center_drift:.3f}.")
            guidance.append("Recenter the six slots in the guide boxes before capture.")
            issue_score += 2
        elif center_drift > MAX_BEST_CENTER_DRIFT_RATIO:
            issues.append(f"Cards are slightly off-center; center drift ratio is {center_drift:.3f}.")
            guidance.append("Use the guide boxes as the final alignment check.")
            issue_score += 1
        else:
            strengths.append(f"Card alignment is centered; drift ratio is {center_drift:.3f}.")

    spacing = safe_float(metrics.get("spacingVariationRatio"))
    if spacing is not None:
        if spacing > MAX_USABLE_SPACING_VARIATION_RATIO:
            issues.append(f"Card spacing is uneven; spacing variation ratio is {spacing:.3f}.")
            guidance.append("Hold the phone flatter to the screen so the vertical slot spacing stays even.")
            issue_score += 2
        elif spacing > MAX_BEST_SPACING_VARIATION_RATIO:
            issues.append(f"Card spacing is slightly uneven; spacing variation ratio is {spacing:.3f}.")
            guidance.append("Square the phone to the display before taking the photo.")
            issue_score += 1
        else:
            strengths.append(f"Card spacing looks consistent; variation ratio is {spacing:.3f}.")

    if not quality.get("canAnalyze", True):
        issues.append("Backend quality gate rejected this image for analysis.")
        guidance.append("Retake the photo; the detector did not trust the image enough to analyze it.")
        issue_score += 4

    if not guidance and not issues:
        guidance.append("This photo is a best-candidate capture; use it for checking type-cluster matching.")

    if issue_score >= 5:
        grade = "retake"
    elif issue_score >= 2:
        grade = "usable_needs_improvement"
    else:
        grade = "best_candidate"

    return {
        "grade": grade,
        "issueScore": issue_score,
        "qualityLevel": quality.get("qualityLevel"),
        "canAnalyze": quality.get("canAnalyze", True),
        "strengths": strengths,
        "issues": issues,
        "guidance": guidance,
    }


def get_box_area(box):
    """Returns area for a box if possible."""
    if not box:
        return None

    try:
        if isinstance(box, dict):
            width = box.get("width")
            height = box.get("height")

            if width is not None and height is not None:
                return float(width) * float(height)

            x1 = box.get("x1")
            y1 = box.get("y1")
            x2 = box.get("x2")
            y2 = box.get("y2")

            if None not in (x1, y1, x2, y2):
                return abs(float(x2) - float(x1)) * abs(float(y2) - float(y1))

        if isinstance(box, (list, tuple)) and len(box) >= 4:
            x1, y1, x2, y2 = box[:4]
            return abs(float(x2) - float(x1)) * abs(float(y2) - float(y1))

    except (TypeError, ValueError):
        return None

    return None


def get_candidate_gap(slot):
    """Finds the gap between top candidate and second candidate when available."""
    candidates = (
        slot.get("topCandidates")
        or slot.get("candidates")
        or slot.get("pokemonCandidates")
        or []
    )

    if not isinstance(candidates, list) or len(candidates) < 2:
        return None

    def candidate_score(candidate):
        return safe_float(
            candidate.get("similarity")
            or candidate.get("confidence")
            or candidate.get("score")
            or candidate.get("finalScore")
        )

    first_score = candidate_score(candidates[0])
    second_score = candidate_score(candidates[1])

    if first_score is None or second_score is None:
        return None

    return first_score - second_score


def simplify_candidate(candidate):
    """Keeps candidate info useful for debugging ranking."""
    if not isinstance(candidate, dict):
        return candidate

    return {
        "pokemonName": candidate.get("pokemonName") or candidate.get("name"),
        "confidence": candidate.get("confidence"),
        "similarity": candidate.get("similarity"),
        "distance": candidate.get("distance"),
        "score": candidate.get("score"),
        "finalScore": candidate.get("finalScore"),
        "referenceImage": candidate.get("referenceImage"),
        "predictionSource": candidate.get("predictionSource"),
        "referenceTypes": candidate.get("referenceTypes"),
        "matchReason": candidate.get("matchReason"),
    }


def get_top_candidates(slot, limit=5):
    """Returns top candidates from whichever field your detector provides."""
    candidates = (
        slot.get("topCandidates")
        or slot.get("candidates")
        or slot.get("pokemonCandidates")
        or []
    )

    if not isinstance(candidates, list):
        return []

    return [simplify_candidate(candidate) for candidate in candidates[:limit]]


def classify_reference_source(reference_image, prediction_source):
    """Classifies whether match likely came from a clear ref, camera ref, or unknown."""
    ref = str(reference_image or "").lower()
    source = str(prediction_source or "").lower()

    if "camera" in ref or "camera" in source:
        return "camera_or_custom_reference"

    if "upload" in ref or "debug" in ref or "crop" in ref:
        return "camera_or_custom_reference"

    if "menu_cp" in ref or "sprite" in ref or "clear" in ref or "reference" in source:
        return "clear_reference"

    if reference_image:
        return "reference_image_unknown_type"

    return "unknown"


def analyze_slot(slot):
    """Creates a compact AI/debug-friendly analysis of one detected slot."""
    confidence = safe_float(slot.get("confidence"))
    similarity = safe_float(slot.get("similarity"))
    distance = safe_float(slot.get("distance"))
    candidate_gap = get_candidate_gap(slot)

    pokemon_name = slot.get("pokemonName")
    reference_image = slot.get("referenceImage")
    prediction_source = slot.get("predictionSource")

    detected_types = slot.get("detectedTypes")
    reference_types = slot.get("referenceTypes")

    box = slot.get("box")
    crop_path = slot.get("debugPokemonCropPath")
    type_crop_path = slot.get("debugTypeIconCropPath")
    type_crop_paths = slot.get("debugTypeIconCropPaths")

    reference_source_kind = classify_reference_source(reference_image, prediction_source)

    issues = []
    strengths = []
    recommendations = []

    if not pokemon_name:
        issues.append("No PokÃ©mon name returned.")
        recommendations.append("Check card crop placement and whether the PokÃ©mon crop contains the sprite/card image.")

    if confidence is None:
        issues.append("No confidence score returned.")
        recommendations.append("Add confidence/similarity output to the detector for easier threshold tuning.")
    elif confidence >= GOOD_CONFIDENCE:
        strengths.append("PokÃ©mon confidence is strong.")
    elif confidence >= OK_CONFIDENCE:
        issues.append("PokÃ©mon confidence is usable but not strong.")
        recommendations.append("Use top-candidate gap and type filtering before accepting this prediction.")
    else:
        issues.append("PokÃ©mon confidence is weak.")
        recommendations.append("Run type-combo fallback and inspect the PokÃ©mon crop for blur, glare, or bad placement.")

    if candidate_gap is None:
        issues.append("Top candidate gap unavailable.")
        recommendations.append("Return top 5 PokÃ©mon candidates from the detector so weak/confusing matches can be analyzed.")
    elif candidate_gap >= GOOD_GAP:
        strengths.append("Top candidate is clearly separated from second candidate.")
    elif candidate_gap >= WEAK_GAP:
        issues.append("Top candidate gap is small.")
        recommendations.append("Use type detection or candidate reranking before accepting this slot.")
    else:
        issues.append("Top candidate gap is very small.")
        recommendations.append("Treat this as uncertain. The crop likely matches multiple PokÃ©mon too closely.")

    if not box:
        issues.append("No card/slot box returned.")
        recommendations.append("Save object overlay and verify slot/card detector placement.")
    else:
        strengths.append("Card/slot box was returned.")

    if not crop_path:
        issues.append("No debug PokÃ©mon crop path returned.")
        recommendations.append("Save PokÃ©mon crop images for every slot. This is required to debug placement.")
    else:
        strengths.append("Debug PokÃ©mon crop was saved.")

    if detected_types and reference_types:
        detected_set = set(str(t).lower() for t in detected_types)
        reference_set = set(str(t).lower() for t in reference_types)

        if detected_set == reference_set:
            strengths.append("Detected types match reference types.")
        elif detected_set.intersection(reference_set):
            issues.append("Detected types partially match reference types.")
            recommendations.append("Use type result as a soft bonus, not a hard override.")
        else:
            issues.append("Detected types do not match reference types.")
            recommendations.append("Inspect type crop placement or disable type influence for this slot.")
    elif detected_types and not reference_types:
        issues.append("Types were detected, but PokÃ©mon reference types are missing.")
        recommendations.append("Load known PokÃ©mon types locally so type detection can be used for candidate filtering.")
    elif reference_types and not detected_types:
        recommendations.append("Type detection was skipped or failed. This is okay if PokÃ©mon confidence/gap is strong.")

    if reference_source_kind == "camera_or_custom_reference":
        strengths.append("Prediction used camera/custom-style reference.")
    elif reference_source_kind == "clear_reference":
        strengths.append("Prediction used clear reference image.")
    else:
        issues.append("Reference source type is unclear.")
        recommendations.append("Include a referenceSource field when building/loading embeddings.")

    if confidence is not None and candidate_gap is not None:
        if confidence >= GOOD_CONFIDENCE and candidate_gap >= GOOD_GAP:
            decision = "accept_fast_path"
        elif confidence >= OK_CONFIDENCE and candidate_gap >= WEAK_GAP:
            decision = "accept_with_review_or_type_check"
        else:
            decision = "needs_type_or_manual_review"
    elif confidence is not None and confidence >= GOOD_CONFIDENCE:
        decision = "accept_but_gap_unknown"
    else:
        decision = "needs_review"

    top_candidates = get_top_candidates(slot)

    return {
        "position": slot.get("position"),
        "pokemonName": pokemon_name,
        "decision": decision,

        "scores": {
            "confidence": confidence,
            "similarity": similarity,
            "distance": distance,
            "candidateGap": candidate_gap,
        },

        "reference": {
            "referenceImage": reference_image,
            "predictionSource": prediction_source,
            "referenceSourceKind": reference_source_kind,
            "matchReason": slot.get("matchReason"),
        },

        "types": {
            "detectedTypes": detected_types,
            "referenceTypes": reference_types,
            "candidateMode": slot.get("candidateMode"),
            "candidateCount": slot.get("candidateCount"),
            "typeFilteredBest": simplify_candidate(slot.get("typeFilteredBest"))
            if isinstance(slot.get("typeFilteredBest"), dict)
            else slot.get("typeFilteredBest"),
            "globalBest": simplify_candidate(slot.get("globalBest"))
            if isinstance(slot.get("globalBest"), dict)
            else slot.get("globalBest"),
        },

        "placement": {
            "box": box,
            "boxArea": get_box_area(box),
            "debugPokemonCropPath": crop_path,
            "debugTypeIconCropPath": type_crop_path,
            "debugTypeIconCropPaths": type_crop_paths,
            "debugObjectOverlayPath": slot.get("debugObjectOverlayPath"),
        },

        "topCandidates": top_candidates,

        "debugAnalysis": {
            "strengths": strengths,
            "issues": issues,
            "recommendations": recommendations,
        },
    }


def summarize_image_result(filename, result, elapsed_ms):
    """Builds image-level summary for quick review."""
    detected_team = result.get("detectedTeam", [])
    capture_quality = build_capture_quality_debug(result.get("quality", {}))

    accepted = 0
    review = 0
    missing = 0

    decisions = {}

    for slot in detected_team:
        decision = slot.get("decision", "unknown")
        decisions[decision] = decisions.get(decision, 0) + 1

        if decision in {"accept_fast_path", "accept_but_gap_unknown"}:
            accepted += 1
        elif slot.get("pokemonName"):
            review += 1
        else:
            missing += 1

    image_notes = []
    image_notes.extend(capture_quality["guidance"][:3])

    if result.get("status") != "success":
        image_notes.append("Image failed detection or was skipped.")
    elif len(detected_team) < 6:
        image_notes.append("Fewer than 6 slots were returned. Check screen placement, card detector, or crop layout.")
    elif missing > 0:
        image_notes.append("Some slots returned no PokÃ©mon name.")
    elif review > accepted:
        image_notes.append("Most slots need review. Image may have glare, blur, poor angle, or bad crop placement.")
    elif accepted >= 4:
        image_notes.append("Most slots are accepted by fast path. This image is useful for checking speed.")
    else:
        image_notes.append("Mixed result. Inspect uncertain crops and top candidate gaps.")

    return {
        "filename": filename,
        "status": result.get("status"),
        "elapsedMs": elapsed_ms,
        "slotCount": len(detected_team),
        "acceptedFastPathCount": accepted,
        "needsReviewCount": review,
        "missingCount": missing,
        "decisions": decisions,
        "captureQuality": capture_quality,
        "imageNotes": image_notes,
    }


def detect_one_image(image_path):
    """Runs detection for one image and returns focused JSON-safe output."""
    started = time.perf_counter()

    try:
        result = detect_opponent_team(image_path, save_debug=True)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        quality = result.get("quality", {})

        if not quality.get("canAnalyze", True):
            output = {
                "filename": image_path.name,
                "imagePath": str(image_path),
                "status": "skipped_bad_quality",
                "elapsedMs": elapsed_ms,
                "quality": quality,
                "captureQuality": build_capture_quality_debug(quality),
                "referenceCount": result.get("referenceCount"),
                "detectedTeam": [],
            }
            output["summary"] = summarize_image_result(image_path.name, output, elapsed_ms)
            return output

        detected_team = [
            analyze_slot(slot)
            for slot in result.get("detectedTeam", [])
        ]

        output = {
            "filename": image_path.name,
            "imagePath": str(image_path),
            "status": "success",
            "elapsedMs": elapsed_ms,
            "quality": quality,
            "captureQuality": build_capture_quality_debug(quality),
            "referenceCount": result.get("referenceCount"),
            "detectedTeam": detected_team,

            # Useful if your service already returns these.
            # They help detect whether the backend is reloading indexes every image.
            "pipelineDebug": {
                "pokemonIndexLoaded": result.get("pokemonIndexLoaded"),
                "typeIndexLoaded": result.get("typeIndexLoaded"),
                "typeComboIndexLoaded": result.get("typeComboIndexLoaded"),
                "pokemonReferenceCount": result.get("pokemonReferenceCount"),
                "typeReferenceCount": result.get("typeReferenceCount"),
                "typeComboReferenceCount": result.get("typeComboReferenceCount"),
            },
        }

        output["summary"] = summarize_image_result(image_path.name, output, elapsed_ms)
        return output

    except ComputerVisionError as error:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        output = {
            "filename": image_path.name,
            "imagePath": str(image_path),
            "status": "cv_error",
            "elapsedMs": elapsed_ms,
            "error": str(error),
            "detectedTeam": [],
        }
        output["summary"] = summarize_image_result(image_path.name, output, elapsed_ms)
        return output

    except Exception as error:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        output = {
            "filename": image_path.name,
            "imagePath": str(image_path),
            "status": "error",
            "elapsedMs": elapsed_ms,
            "error": repr(error),
            "detectedTeam": [],
        }
        output["summary"] = summarize_image_result(image_path.name, output, elapsed_ms)
        return output


def build_overall_summary(results):
    """Creates a short report-level summary."""
    total_images = len(results)
    success_images = sum(1 for r in results if r.get("status") == "success")
    failed_images = total_images - success_images

    total_slots = 0
    accepted_fast = 0
    needs_review = 0
    missing = 0

    total_elapsed = 0.0
    slowest_images = []

    decision_counts = {}
    capture_quality_counts = {}
    retake_images = []

    for result in results:
        elapsed = safe_float(result.get("elapsedMs")) or 0.0
        total_elapsed += elapsed

        slowest_images.append({
            "filename": result.get("filename"),
            "elapsedMs": elapsed,
            "status": result.get("status"),
        })

        summary = result.get("summary", {})
        total_slots += int(summary.get("slotCount") or 0)
        accepted_fast += int(summary.get("acceptedFastPathCount") or 0)
        needs_review += int(summary.get("needsReviewCount") or 0)
        missing += int(summary.get("missingCount") or 0)

        for decision, count in summary.get("decisions", {}).items():
            decision_counts[decision] = decision_counts.get(decision, 0) + count

        capture_quality = summary.get("captureQuality") or result.get("captureQuality") or {}
        capture_grade = capture_quality.get("grade", "unknown")
        capture_quality_counts[capture_grade] = capture_quality_counts.get(capture_grade, 0) + 1

        if capture_grade in {"retake", "usable_needs_improvement"}:
            retake_images.append({
                "filename": result.get("filename"),
                "grade": capture_grade,
                "issueScore": capture_quality.get("issueScore"),
                "issues": capture_quality.get("issues", [])[:4],
                "guidance": capture_quality.get("guidance", [])[:3],
            })

    slowest_images = sorted(
        slowest_images,
        key=lambda item: item["elapsedMs"],
        reverse=True,
    )[:10]

    avg_elapsed = round(total_elapsed / total_images, 2) if total_images else 0.0

    return {
        "imageCount": total_images,
        "successImageCount": success_images,
        "failedOrSkippedImageCount": failed_images,
        "totalSlotCount": total_slots,
        "acceptedFastPathSlotCount": accepted_fast,
        "needsReviewSlotCount": needs_review,
        "missingSlotCount": missing,
        "decisionCounts": decision_counts,
        "captureQualityCounts": capture_quality_counts,
        "imagesToRetakeOrImprove": retake_images[:10],
        "averageElapsedMsPerImage": avg_elapsed,
        "slowestImages": slowest_images,
        "debugInterpretation": [
            "Use captureQuality.grade first: best_candidate photos are the ones to judge type-cluster matching with.",
            "If imagesToRetakeOrImprove is populated, fix those photo issues before tuning prediction thresholds.",
            "If acceptedFastPathSlotCount is high, PokÃ©mon image matching is strong enough for many slots.",
            "If needsReviewSlotCount is high, type-combo fallback and better camera/custom references are needed.",
            "If missingSlotCount is high, card placement/cropping is probably the main issue.",
            "If slowest images are much slower than average, inspect whether type detection or index loading is running too often.",
            "If referenceSourceKind is mostly clear_reference and results are wrong on phone photos, add more camera-style references.",
            "If referenceSourceKind is mostly camera_or_custom_reference and results improve, your added crops are helping.",
        ],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not UPLOAD_DIR.exists():
        raise FileNotFoundError(f"Upload folder was not found: {UPLOAD_DIR}")

    image_paths = sorted(
        path
        for path in UPLOAD_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    )

    print(f"Found {len(image_paths)} uploaded images.")
    print(f"Writing focused debug report to: {OUTPUT_PATH}")
    print()

    results = []

    for index, image_path in enumerate(image_paths, start=1):
        print(f"[{index}/{len(image_paths)}] Detecting {image_path.name}...")

        result = detect_one_image(image_path)
        results.append(result)

        status = result.get("status")
        elapsed_ms = result.get("elapsedMs")
        summary = result.get("summary", {})

        if status == "success":
            names = [
                slot.get("pokemonName") or "unknown"
                for slot in result.get("detectedTeam", [])
            ]

            print(f"  -> {', '.join(names)}")
            print(
                "  -> "
                f"fast={summary.get('acceptedFastPathCount', 0)}, "
                f"review={summary.get('needsReviewCount', 0)}, "
                f"missing={summary.get('missingCount', 0)}, "
                f"photo={summary.get('captureQuality', {}).get('grade', 'unknown')}, "
                f"time={elapsed_ms}ms"
            )
        else:
            print(f"  -> {status}: {result.get('error', 'no error message')}")
            print(f"  -> time={elapsed_ms}ms")

    report = {
        "createdAt": datetime.now().isoformat(timespec="seconds"),
        "uploadDir": str(UPLOAD_DIR),
        "outputPath": str(OUTPUT_PATH),
        "thresholds": {
            "GOOD_CONFIDENCE": GOOD_CONFIDENCE,
            "OK_CONFIDENCE": OK_CONFIDENCE,
            "GOOD_GAP": GOOD_GAP,
            "WEAK_GAP": WEAK_GAP,
            "BEST_MIN_SHORT_EDGE": BEST_MIN_SHORT_EDGE,
            "BEST_MIN_SHARPNESS": BEST_MIN_SHARPNESS,
            "USABLE_MIN_SHARPNESS": USABLE_MIN_SHARPNESS,
            "BEST_MIN_CARD_AREA_RATIO": BEST_MIN_CARD_AREA_RATIO,
            "USABLE_MIN_CARD_AREA_RATIO": USABLE_MIN_CARD_AREA_RATIO,
            "MAX_BEST_EXPOSURE_RATIO": MAX_BEST_EXPOSURE_RATIO,
            "MAX_USABLE_EXPOSURE_RATIO": MAX_USABLE_EXPOSURE_RATIO,
        },
        "overallSummary": build_overall_summary(results),
        "results": results,
    }

    OUTPUT_PATH.write_text(
    json.dumps(make_json_safe(report), indent=2, ensure_ascii=False),
    encoding="utf-8",
)

    print()
    print("Saved detection report to:")
    print(OUTPUT_PATH)

    print()
    print("Overall summary:")
    print(json.dumps(make_json_safe(report["overallSummary"]), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

