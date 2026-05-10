"""Detects Pokemon spirits from opponent team card crops."""

from pathlib import Path

from services import cv_service
from services.pokemon_embedding_service import (
    PokemonEmbeddingIndexMissing,
    get_pokemon_embedding_service,
)


EMBEDDING_ACCEPT_CONFIDENCE = 0.78
TYPE_FILTERED_FAST_ACCEPT_CONFIDENCE = 0.70
GLOBAL_FAST_ACCEPT_CONFIDENCE = 0.80
TEMPLATE_FALLBACK_ACCEPT_CONFIDENCE = 0.62


class PokemonSpiritDetectionService:
    """Matches cropped Pokemon spirits against Champion sprite references."""

    # Loads Pokemon sprite references once for repeated detection runs.
    def __init__(
        self,
        reference_dir=cv_service.REFERENCE_IMAGE_DIR,
        metadata_path=cv_service.REFERENCE_METADATA_PATH,
        extra_reference_dir=cv_service.EXTRA_REFERENCE_IMAGE_DIR,
    ):
        self.reference_dir = Path(reference_dir)
        self.metadata_path = Path(metadata_path)
        self.extra_reference_dir = Path(extra_reference_dir)
        self.references = cv_service.load_reference_images(
            reference_dir=self.reference_dir,
            metadata_path=self.metadata_path,
            extra_reference_dir=self.extra_reference_dir,
        )

        try:
            self.embedding_service = get_pokemon_embedding_service()
            self.embedding_available = True
        except PokemonEmbeddingIndexMissing:
            self.embedding_service = None
            self.embedding_available = False

    # Extracts the spirit area from one opponent card crop.
    def extract_spirit_region(self, slot_image):
        return cv_service.extract_opponent_pokemon_region(slot_image)

    # Matches one isolated spirit crop against reference sprites.
    def detect_spirit(self, spirit_region, detected_types=None):
        detected_types = detected_types or []

        embedding_match = None

        if self.embedding_available and self.embedding_service:
            embedding_match = self.embedding_service.detect_pokemon_by_embedding(
                spirit_region,
                candidate_types=detected_types,
            )

            if embedding_match.get("confidence", 0) >= EMBEDDING_ACCEPT_CONFIDENCE:
                return embedding_match

            candidate_mode = embedding_match.get("candidateMode")
            embedding_confidence = float(embedding_match.get("confidence", 0) or 0)
            if (
                candidate_mode in {"exact_type", "contains_all_types"}
                and embedding_confidence >= TYPE_FILTERED_FAST_ACCEPT_CONFIDENCE
            ):
                return {
                    **embedding_match,
                    "needsReview": True,
                    "matchReason": f"{embedding_match.get('matchReason', 'embedding')}-fast",
                }

            if candidate_mode == "global" and embedding_confidence >= GLOBAL_FAST_ACCEPT_CONFIDENCE:
                return {
                    **embedding_match,
                    "needsReview": True,
                    "matchReason": f"{embedding_match.get('matchReason', 'embedding')}-fast",
                }

        # Old matcher stays as fallback.
        template_match = cv_service.detect_pokemon_from_region(
            spirit_region,
            self.references,
            detected_types=detected_types,
        )

        template_confidence = float(template_match.get("confidence", 0) or 0)
        embedding_confidence = float((embedding_match or {}).get("confidence", 0) or 0)

        if template_confidence >= TEMPLATE_FALLBACK_ACCEPT_CONFIDENCE:
            return {
                **template_match,
                "predictionSource": "template_fallback",
                "matchReason": template_match.get("matchReason", "template-fallback"),
                "needsReview": template_confidence < TEMPLATE_FALLBACK_ACCEPT_CONFIDENCE,
                "embeddingBest": compact_embedding_match(embedding_match),
            }

        if embedding_match and embedding_confidence >= template_confidence:
            return {
                **embedding_match,
                "needsReview": True,
                "templateBest": compact_template_match(template_match),
            }

        return {
            **template_match,
            "predictionSource": "template_low_confidence",
            "matchReason": template_match.get("matchReason", "template-low-confidence"),
            "needsReview": True,
            "embeddingBest": compact_embedding_match(embedding_match),
        }

    # Extracts and matches a Pokemon spirit from one opponent card crop.
    def detect_slot_spirit(self, slot_image, detected_types=None):
        spirit_region = self.extract_spirit_region(slot_image)
        return self.detect_spirit(spirit_region, detected_types=detected_types)


def compact_embedding_match(match):
    if not match:
        return None

    return {
        "pokemonName": match.get("pokemonName", "unknown"),
        "confidence": match.get("confidence", 0.0),
        "distance": match.get("distance"),
        "referenceImage": match.get("referenceImage", ""),
        "referenceTypes": match.get("referenceTypes", []),
        "predictionSource": match.get("predictionSource", ""),
    }


def compact_template_match(match):
    if not match:
        return None

    return {
        "pokemonName": match.get("pokemonName", "unknown"),
        "confidence": match.get("confidence", 0.0),
        "referenceImage": match.get("referenceImage", ""),
        "referenceTypes": match.get("referenceTypes", []),
        "matchReason": match.get("matchReason", "visual-match"),
    }
