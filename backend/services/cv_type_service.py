"""Detects Pokemon type icons from opponent team card crops."""

from pathlib import Path

from services import cv_service
from services.cv_card_service import OpponentCardService
from services.type_embedding_service import (
    TypeEmbeddingIndexMissing,
    get_type_embedding_service,
)


class PokemonTypeDetectionService:
    """Detects one or two Pokemon types for each opponent team slot."""

    # Loads type icon references once for repeated detection runs.
    def __init__(
        self,
        type_reference_dir=cv_service.TYPE_REFERENCE_IMAGE_DIR,
        debug_dir=cv_service.OPPONENT_DEBUG_CROP_DIR,
        card_service=None,
    ):
        self.type_reference_dir = Path(type_reference_dir)
        self.debug_dir = Path(debug_dir)
        self.card_service = card_service or OpponentCardService(debug_dir=self.debug_dir)
        self.type_references = cv_service.load_type_icon_references(self.type_reference_dir)

        try:
            self.type_embedding_service = get_type_embedding_service()
            self.type_embedding_available = True
        except TypeEmbeddingIndexMissing:
            self.type_embedding_service = None
            self.type_embedding_available = False

    # Detects type icons from every opponent card in an uploaded team image.
    def detect_team_types(self, image_path, save_debug=True):
        return cv_service.detect_opponent_team_types_with_references(
            image_path=Path(image_path),
            type_references=self.type_references,
            debug_dir=self.debug_dir,
            save_debug=save_debug,
        )

    # Detects type icons from one card crop and returns each method result.
    def detect_slot_types(self, slot_image, type_icon_crops=None):
        cv_results = cv_service.detect_type_method_results(
            slot_image,
            type_references=self.type_references,
            type_icon_crops=type_icon_crops,
        )

        if cv_results.get("typeComboDetails", {}).get("predictionSource", "").startswith("type_combo_template"):
            return {
                **cv_results,
                "allSelectedTypes": cv_results.get("selected", []),
                "typePredictionSource": cv_results["typeComboDetails"].get("predictionSource"),
                "embeddingSelected": [],
                "embeddingDetails": [],
                "cvSelected": cv_results.get("selected", []),
            }

        if not self.type_embedding_available or not self.type_embedding_service:
            return {
                **cv_results,
                **select_first_single_type_result(
                    cv_results.get("selected", []),
                    fallback_source="cv_template",
                ),
            }

        embedding_results = self.type_embedding_service.classify_slot_types(
            slot_image,
            type_icon_crops=type_icon_crops,
        )
        embedding_selected = embedding_results.get("selected", [])
        embedding_details = embedding_results.get("embeddingDetails", [])
        cv_selected = cv_results.get("selected", [])

        if should_combine_single_embedding_and_cv_type(
            cv_selected,
            embedding_selected,
            embedding_details,
        ):
            return {
                **cv_results,
                **select_strongest_single_type_result(
                    cv_service.normalize_detected_type_pair(
                        [embedding_selected[0], cv_selected[0]]
                    ),
                    embedding_details,
                    fallback_source="type_embedding_cv_complement",
                ),
                "allSelectedTypes": cv_service.normalize_detected_type_pair(
                    [embedding_selected[0], cv_selected[0]]
                ),
                "embeddingSelected": embedding_selected,
                "embeddingDetails": embedding_details,
                "cvSelected": cv_selected,
            }

        if should_trust_single_cv_type(cv_selected, embedding_selected, embedding_details):
            return {
                **cv_results,
                "selected": cv_selected,
                "allSelectedTypes": cv_selected,
                "embeddingSelected": embedding_selected,
                "embeddingDetails": embedding_details,
                "cvSelected": cv_selected,
                "typePredictionSource": "cv_template_embedding_disagreement",
            }

        if should_trust_cv_pair_over_weak_embedding(cv_selected, embedding_selected, embedding_details):
            return {
                **cv_results,
                "selected": cv_selected,
                "allSelectedTypes": cv_selected,
                "embeddingSelected": embedding_selected,
                "embeddingDetails": embedding_details,
                "cvSelected": cv_selected,
                "typePredictionSource": "cv_template_weak_embedding_fallback",
            }

        # Use embedding types when available. Keep CV results for debugging.
        if embedding_selected:
            return {
                **cv_results,
                **select_strongest_single_type_result(
                    embedding_selected,
                    embedding_details,
                    fallback_source="type_embedding",
                ),
                "allSelectedTypes": embedding_selected,
                "embeddingSelected": embedding_selected,
                "embeddingDetails": embedding_details,
                "cvSelected": cv_selected,
            }

        return {
            **cv_results,
            **select_first_single_type_result(
                cv_selected,
                fallback_source="cv_template_fallback",
            ),
            "embeddingSelected": [],
            "embeddingDetails": embedding_details,
            "cvSelected": cv_selected,
        }

    # Crops the individual type icon boxes from one card crop for debugging.
    def crop_slot_type_icons(self, slot_image):
        return cv_service.crop_adaptive_type_icons_from_slot(slot_image)

    # Classifies one isolated type icon crop.
    def classify_type_icon(self, icon_crop):
        if self.type_embedding_available and self.type_embedding_service:
            result = self.type_embedding_service.classify_type_icon(icon_crop)
            if result.get("type"):
                return result["type"]

        return cv_service.classify_type_by_template(icon_crop, self.type_references)


def should_combine_single_embedding_and_cv_type(cv_selected, embedding_selected, embedding_details):
    if len(cv_selected) != 1 or len(embedding_selected) != 1:
        return False

    if cv_selected[0] == embedding_selected[0]:
        return False

    confident_duplicate_embedding = [
        detail
        for detail in embedding_details
        if detail.get("type") == embedding_selected[0]
        and detail.get("hasSymbol")
        and detail.get("confidence", 0) >= 0.84
    ]

    return len(confident_duplicate_embedding) >= 2


def should_trust_single_cv_type(cv_selected, embedding_selected, embedding_details):
    if len(cv_selected) != 1 or len(embedding_selected) != 1:
        return False

    if cv_selected[0] == embedding_selected[0]:
        return False

    weak_or_fallback_details = [
        detail
        for detail in embedding_details
        if detail.get("cropSource") == "fixed_type_icon_fallback"
        or not detail.get("hasSymbol")
        or detail.get("confidence", 0) < 0.80
    ]
    selected_detail = next(
        (
            detail
            for detail in embedding_details
            if detail.get("type") == embedding_selected[0]
        ),
        {},
    )

    return bool(weak_or_fallback_details) and selected_detail.get("index") == 2


def should_trust_cv_pair_over_weak_embedding(cv_selected, embedding_selected, embedding_details):
    cv_selected = cv_service.normalize_detected_type_pair(cv_selected or [])[:2]
    embedding_selected = cv_service.normalize_detected_type_pair(embedding_selected or [])[:2]

    if len(cv_selected) < 2 or len(embedding_selected) < 2:
        return False

    cv_set = set(cv_selected)
    embedding_set = set(embedding_selected)
    if cv_set == embedding_set:
        return False

    conflicting_weak_details = [
        detail
        for detail in embedding_details or []
        if detail.get("type")
        and detail.get("type") not in cv_set
        and detail.get("cropSource") in {
            "fixed_type_icon_fallback",
            "shifted_fixed_type_icon_left",
            "shifted_fixed_type_icon_right",
            "shifted_fixed_type_icon_down",
            "shifted_fixed_type_icon_wide",
        }
        and float(detail.get("cropQuality", 0) or 0) <= 0.05
    ]

    return bool(conflicting_weak_details) and bool(cv_set.intersection(embedding_set))


def select_strongest_single_type_result(selected_types, embedding_details, fallback_source):
    selected_types = [
        detected_type
        for detected_type in selected_types or []
        if detected_type
    ]
    selected_types = cv_service.normalize_detected_type_pair(selected_types)[:2]

    if len(selected_types) <= 1:
        return {
            "selected": selected_types,
            "allSelectedTypes": selected_types,
            "typePredictionSource": fallback_source,
        }

    selected_type_set = set(selected_types)
    strongest_detail = max(
        (
            detail
            for detail in embedding_details or []
            if detail.get("type") in selected_type_set
        ),
        key=lambda detail: (
            float(detail.get("confidence", 0) or 0),
            bool(detail.get("hasSymbol")),
            -int(detail.get("index") or 99),
        ),
        default={},
    )

    strongest_type = strongest_detail.get("type") or selected_types[0]

    return {
        "selected": selected_types,
        "allSelectedTypes": selected_types,
        "strongestType": strongest_type,
        "strongestTypeConfidence": strongest_detail.get("confidence"),
        "typePredictionSource": f"{fallback_source}_dual",
    }


def select_first_single_type_result(selected_types, fallback_source):
    selected_types = [
        detected_type
        for detected_type in selected_types or []
        if detected_type
    ]
    selected_types = cv_service.normalize_detected_type_pair(selected_types)[:2]

    if len(selected_types) <= 1:
        return {
            "selected": selected_types,
            "allSelectedTypes": selected_types,
            "typePredictionSource": fallback_source,
        }

    return {
        "selected": selected_types,
        "allSelectedTypes": selected_types,
        "strongestType": selected_types[0],
        "strongestTypeConfidence": None,
        "typePredictionSource": f"{fallback_source}_dual",
    }


# Detects type icons for an uploaded opponent team image.
def detect_opponent_team_types(image_path, save_debug=True):
    service = PokemonTypeDetectionService()
    return service.detect_team_types(image_path, save_debug=save_debug)
