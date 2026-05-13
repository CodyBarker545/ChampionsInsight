"""Opponent team detection using YOLO crops plus DINOv2/FAISS matching."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from PIL import Image

from paths import CV_INDEX_DIR, DATA_DIR, OPPONENT_DEBUG_CROP_DIR
from services import cv_service
from services.dinov2_faiss_service import Dinov2FaissIndex
from services.pokemon_data_service import build_pokemon_summary


logger = logging.getLogger(__name__)

DEFAULT_INDEX_ROOT = CV_INDEX_DIR / "dinov2_faiss"
REVIEW_ROOT = DATA_DIR / "training_dataset" / "review"
UNSORTED_SLOT_DIR = REVIEW_ROOT / "unsorted_slots"
UNSORTED_POKEMON_DIR = REVIEW_ROOT / "unsorted_pokemon"
UNSORTED_TYPE_COMBO_DIR = REVIEW_ROOT / "unsorted_type_combos"


def cv_to_pil(image) -> Image.Image:
    cv2, _np = cv_service.load_cv_dependencies()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb).convert("RGB")


def normalize_type_name(type_name):
    return str(type_name or "").strip().lower().replace(" ", "_").replace("-", "_")


def split_type_combo_label(label):
    normalized_label = str(label or "").strip().lower()
    if not normalized_label or normalized_label == "unknown":
        return []
    return [part for part in normalized_label.split("_") if part]


def format_top_hits(hits, limit=5):
    formatted_hits = []
    for hit in hits[:limit]:
        formatted_hits.append({
            "label": hit.get("label", ""),
            "similarity": round(float(hit.get("similarity", 0) or 0), 4),
            "confidence": round(float(hit.get("confidence", 0) or 0), 4),
            "path": hit.get("path", ""),
        })
    return formatted_hits


def get_reference_types(pokemon_name):
    if not pokemon_name or pokemon_name == "unknown":
        return []
    summary = build_pokemon_summary(pokemon_name)
    return summary.get("types", []) or []


def build_type_mismatch(reference_types, detected_types):
    reference_set = {normalize_type_name(type_name) for type_name in reference_types}
    detected_set = {normalize_type_name(type_name) for type_name in detected_types}
    reference_set.discard("")
    detected_set.discard("")
    return bool(reference_set and detected_set and reference_set != detected_set)


def type_overlap_score(reference_types, detected_types):
    reference_set = {normalize_type_name(type_name) for type_name in reference_types}
    detected_set = {normalize_type_name(type_name) for type_name in detected_types}
    reference_set.discard("")
    detected_set.discard("")

    if not reference_set or not detected_set:
        return 0.0

    if reference_set == detected_set:
        return 0.14

    overlap_count = len(reference_set.intersection(detected_set))
    if overlap_count == len(detected_set):
        return 0.10

    if overlap_count:
        return 0.06

    return -0.08


def select_type_guided_pokemon_hit(pokemon_hits, detected_types):
    if not pokemon_hits:
        return None, ""

    if not detected_types:
        return pokemon_hits[0], "dinov2-faiss-pokemon"

    scored_hits = []
    for hit in pokemon_hits:
        reference_types = get_reference_types(hit.get("label"))
        base_similarity = float(hit.get("similarity", 0) or 0)
        scored_hits.append((
            base_similarity + type_overlap_score(reference_types, detected_types),
            base_similarity,
            hit,
        ))

    scored_hits.sort(key=lambda item: (item[0], item[1]), reverse=True)
    selected_hit = scored_hits[0][2]

    if selected_hit is pokemon_hits[0]:
        return selected_hit, "dinov2-faiss-pokemon"

    return selected_hit, "type-guided-dinov2-faiss-pokemon"


@lru_cache(maxsize=1)
def get_pokemon_index():
    return Dinov2FaissIndex(
        DEFAULT_INDEX_ROOT / "pokemon" / "index.faiss",
        DEFAULT_INDEX_ROOT / "pokemon" / "metadata.json",
    )


@lru_cache(maxsize=1)
def get_type_index():
    return Dinov2FaissIndex(
        DEFAULT_INDEX_ROOT / "types" / "index.faiss",
        DEFAULT_INDEX_ROOT / "types" / "metadata.json",
    )


def search_index(index, image, top_k=5):
    if image is None or cv_service.is_empty_image(image):
        return []
    return index.search(cv_to_pil(image), top_k=top_k)


def best_label(hits):
    return hits[0].get("label", "unknown") if hits else "unknown"


def best_similarity(hits):
    return round(float(hits[0].get("similarity", 0) or 0), 4) if hits else 0.0


def best_confidence(hits):
    return round(float(hits[0].get("confidence", 0) or 0), 4) if hits else 0.0


def safe_filename_part(value):
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in str(value or "unknown")
    ).strip("_") or "unknown"


def write_image(output_path, image):
    if image is None or cv_service.is_empty_image(image):
        return ""

    cv2, _np = cv_service.load_cv_dependencies()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise cv_service.ComputerVisionError(f"Could not write image: {output_path}")
    return str(output_path)


def write_review_crop(review_dir, prefix, image_stem, position, prediction, image):
    if image is None or cv_service.is_empty_image(image):
        return ""

    filename = (
        f"{safe_filename_part(prefix)}__{safe_filename_part(image_stem)}"
        f"__slot-{position}__{safe_filename_part(prediction)}__{uuid4().hex[:8]}.jpg"
    )
    return write_image(Path(review_dir) / filename, image)


class Dinov2OpponentDetectionService:
    """Runs the current opponent prediction path used by uploads and reports."""

    def __init__(
        self,
        pokemon_index=None,
        type_index=None,
        debug_dir=OPPONENT_DEBUG_CROP_DIR,
        classify_type_evidence=False,
    ):
        self.pokemon_index = pokemon_index or get_pokemon_index()
        self.classify_type_evidence = classify_type_evidence
        self.type_index = type_index or (get_type_index() if classify_type_evidence else None)
        self.debug_dir = Path(debug_dir)

    def assess_quality(self, image_path):
        return cv_service.assess_opponent_image_quality(image_path)

    def detect_team(self, image_path, save_debug=True, save_review_crops=True):
        image_path = Path(image_path)
        quality = self.assess_quality(image_path)
        debug_original_path = ""

        if save_debug:
            debug_original_path = self.write_debug_original_image(image_path)

        slot_crops = cv_service.crop_opponent_team_slots(image_path, save_debug=save_debug)
        detected_team = self.detect_slots(
            image_path=image_path,
            card_crops=slot_crops,
            save_debug=save_debug,
            save_review_crops=save_review_crops,
        )

        return {
            "image": str(image_path),
            "quality": quality,
            "debugOriginalPath": debug_original_path,
            "detectedTeam": detected_team,
            "predictionSource": "dinov2_faiss_yolo",
        }

    def detect_slots(self, image_path, card_crops, save_debug=True, save_review_crops=True):
        slot_records = []

        for card_crop in card_crops:
            slot_image = card_crop["image"]
            object_layer = cv_service.detect_slot_object_layer(slot_image)
            pokemon_crop = self.extract_pokemon_crop(slot_image, object_layer)
            type_crop_info = self.extract_type_crop(slot_image, object_layer)
            slot_records.append({
                "cardCrop": card_crop,
                "slotImage": slot_image,
                "objectLayer": object_layer,
                "pokemonCrop": pokemon_crop,
                "typeCropInfo": type_crop_info,
                "pokemonHits": [],
                "typeHits": [],
            })

        self.add_batched_hits(slot_records)

        return [
            self.build_slot_result(
                image_path=image_path,
                slot_record=slot_record,
                save_debug=save_debug,
                save_review_crops=save_review_crops,
            )
            for slot_record in slot_records
        ]

    def add_batched_hits(self, slot_records):
        pokemon_inputs = [
            (index, cv_to_pil(slot_record["pokemonCrop"]))
            for index, slot_record in enumerate(slot_records)
            if not cv_service.is_empty_image(slot_record.get("pokemonCrop"))
        ]
        if pokemon_inputs:
            pokemon_hits = self.pokemon_index.search_many(
                [image for _index, image in pokemon_inputs],
                top_k=40,
            )
            for (slot_index, _image), hits in zip(pokemon_inputs, pokemon_hits):
                slot_records[slot_index]["pokemonHits"] = hits

        if not self.classify_type_evidence or self.type_index is None:
            return

        type_inputs = [
            (index, cv_to_pil(slot_record["typeCropInfo"].get("image")))
            for index, slot_record in enumerate(slot_records)
            if not cv_service.is_empty_image(slot_record["typeCropInfo"].get("image"))
        ]
        if type_inputs:
            type_hits = self.type_index.search_many(
                [image for _index, image in type_inputs],
                top_k=5,
            )
            for (slot_index, _image), hits in zip(type_inputs, type_hits):
                slot_records[slot_index]["typeHits"] = hits

    def build_slot_result(self, image_path, slot_record, save_debug=True, save_review_crops=True):
        card_crop = slot_record["cardCrop"]
        slot_image = slot_record["slotImage"]
        object_layer = slot_record["objectLayer"]
        pokemon_crop = slot_record["pokemonCrop"]
        type_crop_info = slot_record["typeCropInfo"]
        type_crop = type_crop_info.get("image")
        position = card_crop["position"]

        pokemon_hits = slot_record["pokemonHits"]
        type_hits = slot_record["typeHits"]

        type_result = cv_service.classify_type_crop(type_crop_info)
        type_label = type_result.get("typeKey") or best_label(type_hits)
        detected_types = type_result.get("types") or split_type_combo_label(type_label)
        selected_pokemon_hit, match_reason = select_type_guided_pokemon_hit(
            pokemon_hits,
            detected_types,
        )
        pokemon_name = (
            selected_pokemon_hit.get("label", "unknown")
            if selected_pokemon_hit
            else "unknown"
        )
        reference_types = get_reference_types(pokemon_name)
        type_mismatch = build_type_mismatch(reference_types, detected_types)

        debug_pokemon_crop_path = ""
        debug_type_icon_crop_path = ""
        review_paths = {}

        if save_debug:
            debug_pokemon_crop_path = self.write_debug_pokemon_crop(
                image_path,
                position,
                pokemon_crop,
            )
            debug_type_icon_crop_path = self.write_debug_type_crop(
                image_path,
                position,
                type_crop,
            )

        if save_review_crops:
            review_paths = self.write_review_crops(
                image_path=image_path,
                position=position,
                slot_image=slot_image,
                pokemon_crop=pokemon_crop,
                type_crop=type_crop,
                pokemon_prediction=pokemon_name,
                type_prediction=type_label,
            )

        return {
            "position": position,
            "pokemonName": pokemon_name,
            "name": pokemon_name,
            "confidence": (
                round(float(selected_pokemon_hit.get("confidence", 0) or 0), 4)
                if selected_pokemon_hit
                else 0.0
            ),
            "similarity": (
                round(float(selected_pokemon_hit.get("similarity", 0) or 0), 4)
                if selected_pokemon_hit
                else 0.0
            ),
            "predictionSource": "dinov2_faiss_pokemon",
            "matchReason": match_reason,
            "needsReview": (
                pokemon_name == "unknown"
                or (
                    selected_pokemon_hit is not None
                    and float(selected_pokemon_hit.get("similarity", 0) or 0) < 0.68
                )
            ),
            "types": reference_types,
            "referenceTypes": reference_types,
            "detectedTypes": detected_types,
            "trustedDetectedTypes": reference_types,
            "typeMismatchWarning": type_mismatch,
            "typeGuardReason": "type-icon-disagrees-with-pokemon-reference" if type_mismatch else None,
            "typeEvidence": {
                "prediction": type_label,
                "types": detected_types,
                "similarity": type_result.get("score", best_similarity(type_hits)),
                "confidence": type_result.get("score", best_confidence(type_hits)),
                "topCandidates": format_top_hits(type_hits),
                "cropSource": type_crop_info.get("cropSource", ""),
                "typeCount": type_crop_info.get("typeCount"),
                "classificationSkipped": False,
                "predictionSource": type_result.get("predictionSource", ""),
            },
            "pokemonTopCandidates": format_top_hits(pokemon_hits),
            "typeTopCandidates": format_top_hits(type_hits),
            "box": card_crop.get("box"),
            "debugCropPath": card_crop.get("debugCropPath", ""),
            "debugPokemonCropPath": debug_pokemon_crop_path,
            "debugTypeIconCropPath": debug_type_icon_crop_path,
            "reviewCropPaths": review_paths,
            "referenceImage": selected_pokemon_hit.get("path", "") if selected_pokemon_hit else "",
            "objectLayerSource": object_layer.get("source", "heuristic"),
        }

    def extract_pokemon_crop(self, slot_image, object_layer):
        pokemon_object = object_layer.get("pokemon_sprite")
        if pokemon_object and not cv_service.is_empty_image(pokemon_object.get("image")):
            return pokemon_object["image"]

        fallback_crop = cv_service.extract_opponent_pokemon_region(slot_image)
        if not cv_service.is_empty_image(fallback_crop):
            return fallback_crop
        return None

    def extract_type_crop(self, slot_image, object_layer):
        type_icon_crops = cv_service.type_icon_crops_from_object_layer(object_layer)
        type_crop_info = cv_service.build_type_combo_candidate_crop(
            slot_image,
            type_icon_crops=type_icon_crops,
        )
        if type_crop_info and not cv_service.is_empty_image(type_crop_info.get("image")):
            return type_crop_info

        fallback_crop = cv_service.extract_type_icon_region(slot_image)
        return {
            "image": fallback_crop,
            "typeCount": None,
            "cropSource": "fallback_type_icon_region",
        }

    def write_debug_original_image(self, image_path):
        original = cv_service.read_cv_image(image_path)
        image_debug_dir = self.get_image_debug_dir(image_path)
        return write_image(image_debug_dir / "original.jpg", original)

    def write_debug_pokemon_crop(self, image_path, position, pokemon_crop):
        image_debug_dir = self.get_image_debug_dir(image_path)
        return write_image(image_debug_dir / f"opponent-pokemon-{position}.jpg", pokemon_crop)

    def write_debug_type_crop(self, image_path, position, type_crop):
        image_debug_dir = self.get_image_debug_dir(image_path)
        return write_image(image_debug_dir / f"opponent-type-icons-{position}.jpg", type_crop)

    def write_review_crops(
        self,
        image_path,
        position,
        slot_image,
        pokemon_crop,
        type_crop,
        pokemon_prediction,
        type_prediction,
    ):
        image_stem = Path(image_path).stem
        return {
            "slot": write_review_crop(
                UNSORTED_SLOT_DIR,
                "slot",
                image_stem,
                position,
                pokemon_prediction,
                slot_image,
            ),
            "pokemon": write_review_crop(
                UNSORTED_POKEMON_DIR,
                "pokemon",
                image_stem,
                position,
                pokemon_prediction,
                pokemon_crop,
            ),
            "typeCombo": write_review_crop(
                UNSORTED_TYPE_COMBO_DIR,
                "type_combo",
                image_stem,
                position,
                type_prediction,
                type_crop,
            ),
        }

    def get_image_debug_dir(self, image_path):
        image_debug_dir = self.debug_dir / Path(image_path).stem
        image_debug_dir.mkdir(parents=True, exist_ok=True)
        return image_debug_dir


@lru_cache(maxsize=1)
def get_detection_service():
    return Dinov2OpponentDetectionService()


def assess_opponent_image_quality(image_path):
    return get_detection_service().assess_quality(image_path)


def detect_opponent_team(image_path, save_debug=True, save_review_crops=True):
    return get_detection_service().detect_team(
        image_path,
        save_debug=save_debug,
        save_review_crops=save_review_crops,
    )
