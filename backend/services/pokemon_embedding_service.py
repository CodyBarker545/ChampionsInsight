"""Embedding-based Pokémon identity matching for opponent spirit crops."""

import json
from functools import lru_cache

import numpy as np
from pathlib import Path
from paths import POKEMON_EMBEDDING_INDEX_DIR
from services import cv_service
from services.embedding_model_service import cosine_similarity_matrix, embed_image


POKEMON_EMBEDDINGS_PATH = POKEMON_EMBEDDING_INDEX_DIR / "embeddings.npy"
POKEMON_METADATA_PATH = POKEMON_EMBEDDING_INDEX_DIR / "metadata.json"

# Main thresholds.
TYPE_FILTER_CONFIDENCE_THRESHOLD = 0.78
LOW_CONFIDENCE_REVIEW_THRESHOLD = 0.74

# Prevent global fallback from replacing a type-consistent result unless clearly better.
TYPE_CONSISTENCY_BONUS = 0.08
GLOBAL_OVERRIDE_MARGIN = 0.10

# Stricter review rules for riskier candidate modes.
GLOBAL_ACCEPT_WITHOUT_REVIEW = 0.88
PARTIAL_ACCEPT_WITHOUT_REVIEW = 0.86
NEAR_EXACT_GLOBAL_CONFIDENCE = 0.985
NEAR_EXACT_GLOBAL_MARGIN = 0.06
STRONG_GLOBAL_RESCUE_CONFIDENCE = 0.91
STRONG_GLOBAL_RESCUE_MARGIN = 0.07


class PokemonEmbeddingIndexMissing(RuntimeError):
    """Raised when the Pokémon embedding index has not been built yet."""


def normalize_type_list(types):
    return [
        str(type_name).strip().lower()
        for type_name in types or []
        if str(type_name).strip()
    ]


def calculate_confidence_from_similarity(similarity):
    """
    Converts cosine similarity to a frontend-friendly confidence.
    MobileNet embedding cosine scores are not true probabilities.
    """
    confidence = (float(similarity) + 1.0) / 2.0
    return max(0.0, min(1.0, confidence))


def type_overlap_score(detected_types, reference_types):
    """
    Scores agreement between detected type icons and the matched Pokémon's known types.

    1.00 = exact same type set
    0.85 = detected type is subset of reference types
    0.55 = at least one type overlaps
    0.00 = no overlap
    """
    detected = set(normalize_type_list(detected_types))
    reference = set(normalize_type_list(reference_types))

    if not detected or not reference:
        return 0.0

    if detected == reference:
        return 1.0

    if detected.issubset(reference):
        return 0.85

    if detected.intersection(reference):
        return 0.55

    return 0.0


def should_need_review(result, detected_types):
    """
    Determines whether the frontend should ask the user to confirm this Pokémon.
    Global and partial matches are treated as riskier than exact type matches.
    """
    confidence = float(result.get("confidence", 0) or 0)
    candidate_mode = result.get("candidateMode")
    overlap = type_overlap_score(detected_types, result.get("referenceTypes", []))

    if result.get("pokemonName") == "unknown":
        return True

    if confidence < LOW_CONFIDENCE_REVIEW_THRESHOLD:
        return True

    # If type icons were detected and the chosen Pokémon shares no type, review it.
    if detected_types and overlap == 0.0:
        return True

    # Partial type pools are large, so require stronger confidence.
    if candidate_mode == "partial_type" and confidence < PARTIAL_ACCEPT_WITHOUT_REVIEW:
        return True

    # Global fallback is most risky, so require very high confidence.
    if candidate_mode == "global" and confidence < GLOBAL_ACCEPT_WITHOUT_REVIEW:
        return True

    return False


def build_candidate_indices(metadata, detected_types):
    """
    Builds candidate pools using type hints.

    Order:
    1. exact type match
    2. contains all detected types
    3. shares at least one detected type
    4. global fallback
    """
    detected = set(normalize_type_list(detected_types))
    all_indices = list(range(len(metadata)))

    if not detected:
        return all_indices, "global"

    exact = []
    contains_all = []
    partial = []

    for index, record in enumerate(metadata):
        reference_types = set(normalize_type_list(record.get("types", [])))

        if not reference_types:
            continue

        if reference_types == detected:
            exact.append(index)

        if detected.issubset(reference_types):
            contains_all.append(index)

        if detected.intersection(reference_types):
            partial.append(index)

    if exact:
        return exact, "exact_type"

    if contains_all:
        return contains_all, "contains_all_types"

    if partial:
        return partial, "partial_type"

    return all_indices, "global"


class PokemonEmbeddingService:
    """Finds the nearest Pokémon reference embedding for a cropped spirit image."""

    def __init__(
        self,
        embeddings_path=POKEMON_EMBEDDINGS_PATH,
        metadata_path=POKEMON_METADATA_PATH,
    ):
        self.embeddings_path = Path(embeddings_path)
        self.metadata_path = Path(metadata_path)
        self.embeddings, self.metadata = self.load_index()

    def load_index(self):
        if not self.embeddings_path.exists() or not self.metadata_path.exists():
            raise PokemonEmbeddingIndexMissing(
                "Pokémon embedding index was not found. Run: python scripts/data_build/build_pokemon_embedding_index.py"
            )

        embeddings = np.load(self.embeddings_path).astype("float32")
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

        if len(embeddings) != len(metadata):
            raise PokemonEmbeddingIndexMissing(
                "Pokémon embedding index is invalid: embeddings and metadata lengths do not match."
            )

        return embeddings, metadata

    def find_nearest_pokemon(
        self,
        crop_image,
        candidate_types=None,
        candidate_indices=None,
        query_vector=None,
    ):
        if query_vector is None and cv_service.is_empty_image(crop_image):
            return self.unknown_result("empty-image")

        if query_vector is None:
            query_vector = embed_image(crop_image)

        if candidate_indices is None:
            candidate_indices, candidate_mode = build_candidate_indices(
                self.metadata,
                candidate_types or [],
            )
        else:
            candidate_mode = "manual_candidates"

        if not candidate_indices:
            candidate_indices = list(range(len(self.metadata)))
            candidate_mode = "global"

        candidate_embeddings = self.embeddings[candidate_indices]
        similarities = cosine_similarity_matrix(query_vector, candidate_embeddings)

        if len(similarities) == 0:
            return self.unknown_result("no-candidates")

        best_local_index = int(np.argmax(similarities))
        best_global_index = candidate_indices[best_local_index]
        best_similarity = float(similarities[best_local_index])
        best_metadata = self.metadata[best_global_index]

        confidence = calculate_confidence_from_similarity(best_similarity)
        distance = 1.0 - best_similarity

        return {
            "pokemonName": best_metadata.get("name", "unknown"),
            "confidence": round(confidence, 4),
            "similarity": round(best_similarity, 4),
            "distance": round(distance, 4),
            "referenceImage": best_metadata.get("path", ""),
            "referenceTypes": best_metadata.get("types", []),
            "isShiny": bool(best_metadata.get("isShiny", False)),
            "predictionSource": f"embedding_{candidate_mode}",
            "matchReason": f"embedding-{candidate_mode.replace('_', '-')}",
            "needsReview": confidence < LOW_CONFIDENCE_REVIEW_THRESHOLD,
            "candidateMode": candidate_mode,
            "candidateCount": len(candidate_indices),
        }

    def detect_pokemon_by_embedding(self, crop_image, candidate_types=None):
        """
        Runs type-filtered embedding first, then global fallback only if it is clearly better.
        """
        candidate_types = normalize_type_list(candidate_types)
        if cv_service.is_empty_image(crop_image):
            return self.unknown_result("empty-image")

        query_vector = embed_image(crop_image)

        type_filtered_result = self.find_nearest_pokemon(
            crop_image,
            candidate_types=candidate_types,
            query_vector=query_vector,
        )

        global_result = self.find_nearest_pokemon(
            crop_image,
            candidate_types=None,
            query_vector=query_vector,
        )

        typed_overlap = type_overlap_score(
            candidate_types,
            type_filtered_result.get("referenceTypes", []),
        )

        global_overlap = type_overlap_score(
            candidate_types,
            global_result.get("referenceTypes", []),
        )

        typed_score = type_filtered_result["confidence"] + (
            typed_overlap * TYPE_CONSISTENCY_BONUS
        )

        global_score = global_result["confidence"] + (
            global_overlap * TYPE_CONSISTENCY_BONUS
        )

        # No detected types means there is no useful type-filtered pool.
        if not candidate_types:
            chosen = {
                **global_result,
                "predictionSource": "embedding_global",
                "matchReason": "embedding-global",
                "typeFilteredBest": compact_match(type_filtered_result),
                "globalBest": compact_match(global_result),
            }

            chosen["typeOverlapScore"] = global_overlap
            chosen["needsReview"] = should_need_review(chosen, candidate_types)
            return chosen

        # A camera reference that is almost identical to the crop is stronger
        # evidence than a noisy type read from a glared or partial icon.
        if (
            global_result["confidence"] >= NEAR_EXACT_GLOBAL_CONFIDENCE
            and global_result["confidence"] >= type_filtered_result["confidence"] + NEAR_EXACT_GLOBAL_MARGIN
        ):
            chosen = {
                **global_result,
                "predictionSource": "embedding_near_exact_global_rescue",
                "matchReason": "embedding-near-exact-global-rescue",
                "typeFilteredBest": compact_match(type_filtered_result),
                "globalBest": compact_match(global_result),
            }

            chosen["typeOverlapScore"] = global_overlap
            chosen["needsReview"] = should_need_review(chosen, candidate_types)
            return chosen

        # Type icons can be wrong when a red icon blends into the card or a
        # glare-heavy crop is misclassified. Let a clearly stronger visual
        # match win before accepting a confident-but-wrong type-filtered result.
        if (
            global_result["confidence"] >= STRONG_GLOBAL_RESCUE_CONFIDENCE
            and global_result["confidence"] >= type_filtered_result["confidence"] + STRONG_GLOBAL_RESCUE_MARGIN
        ):
            chosen = {
                **global_result,
                "predictionSource": "embedding_strong_global_rescue",
                "matchReason": "embedding-strong-global-rescue",
                "typeFilteredBest": compact_match(type_filtered_result),
                "globalBest": compact_match(global_result),
            }

            chosen["typeOverlapScore"] = global_overlap
            chosen["needsReview"] = should_need_review(chosen, candidate_types)
            return chosen

        # Prefer a strong type-filtered result.
        if (
            type_filtered_result.get("candidateMode") != "global"
            and type_filtered_result["confidence"] >= TYPE_FILTER_CONFIDENCE_THRESHOLD
        ):
            chosen = {
                **type_filtered_result,
                "typeFilteredBest": compact_match(type_filtered_result),
                "globalBest": compact_match(global_result),
            }

            chosen["typeOverlapScore"] = typed_overlap
            chosen["needsReview"] = should_need_review(chosen, candidate_types)
            return chosen

        # Only let global fallback override if it is clearly better.
        if global_score >= typed_score + GLOBAL_OVERRIDE_MARGIN:
            chosen = {
                **global_result,
                "predictionSource": "embedding_global_fallback",
                "matchReason": "embedding-global-fallback",
                "typeFilteredBest": compact_match(type_filtered_result),
                "globalBest": compact_match(global_result),
            }

            chosen["typeOverlapScore"] = global_overlap
            chosen["needsReview"] = should_need_review(chosen, candidate_types)
            return chosen

        # Otherwise keep the type-filtered result.
        chosen = {
            **type_filtered_result,
            "typeFilteredBest": compact_match(type_filtered_result),
            "globalBest": compact_match(global_result),
        }

        chosen["typeOverlapScore"] = typed_overlap
        chosen["needsReview"] = should_need_review(chosen, candidate_types)
        return chosen

    def unknown_result(self, reason):
        return {
            "pokemonName": "unknown",
            "confidence": 0.0,
            "similarity": 0.0,
            "distance": 1.0,
            "referenceImage": "",
            "referenceTypes": [],
            "predictionSource": "embedding",
            "matchReason": reason,
            "needsReview": True,
            "candidateMode": "none",
            "candidateCount": 0,
        }


def compact_match(result):
    if not result:
        return None

    return {
        "pokemonName": result.get("pokemonName", "unknown"),
        "confidence": result.get("confidence", 0.0),
        "distance": result.get("distance"),
        "referenceImage": result.get("referenceImage", ""),
        "referenceTypes": result.get("referenceTypes", []),
        "predictionSource": result.get("predictionSource", ""),
        "candidateMode": result.get("candidateMode", ""),
        "candidateCount": result.get("candidateCount", 0),
    }


@lru_cache(maxsize=1)
def get_pokemon_embedding_service():
    return PokemonEmbeddingService()
