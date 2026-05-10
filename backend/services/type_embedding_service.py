"""Embedding-based type icon classification."""

import json
from functools import lru_cache

import numpy as np
from pathlib import Path
from paths import TYPE_EMBEDDING_INDEX_DIR
from services import cv_service
from services.embedding_model_service import cosine_similarity_matrix, embed_image


TYPE_EMBEDDINGS_PATH = TYPE_EMBEDDING_INDEX_DIR / "embeddings.npy"
TYPE_METADATA_PATH = TYPE_EMBEDDING_INDEX_DIR / "metadata.json"

TYPE_CONFIDENCE_THRESHOLD = 0.78


class TypeEmbeddingIndexMissing(RuntimeError):
    """Raised when type embedding index has not been built."""


def calculate_confidence_from_similarity(similarity):
    confidence = (float(similarity) + 1.0) / 2.0
    return max(0.0, min(1.0, confidence))


def preprocess_type_icon_for_embedding(icon_crop):
    """
    Normalizes type icons without simply darkening them.
    This helps with glare and washed-out icons.
    """
    if cv_service.is_empty_image(icon_crop):
        return icon_crop

    cv2, np = cv_service.load_cv_dependencies()

    icon = cv2.resize(icon_crop, (96, 96), interpolation=cv2.INTER_CUBIC)

    lab = cv2.cvtColor(icon, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    normalized = cv2.merge([l_channel, a_channel, b_channel])
    icon = cv2.cvtColor(normalized, cv2.COLOR_LAB2BGR)

    return icon


def preprocess_type_icon_glare_reduced(icon_crop):
    """
    More aggressive version for bright/glared type icons.
    Keeps hue but caps value and slightly boosts saturation.
    """
    if cv_service.is_empty_image(icon_crop):
        return icon_crop

    cv2, np = cv_service.load_cv_dependencies()

    icon = cv2.resize(icon_crop, (96, 96), interpolation=cv2.INTER_CUBIC)

    hsv = cv2.cvtColor(icon, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)

    value = np.clip(value, 0, 215).astype("uint8")
    saturation = np.clip(saturation.astype("float32") * 1.25, 0, 255).astype("uint8")

    hsv = cv2.merge([hue, saturation, value])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class TypeEmbeddingService:
    """Classifies type icon crops with embedding nearest-neighbor lookup."""

    def __init__(
        self,
        embeddings_path=TYPE_EMBEDDINGS_PATH,
        metadata_path=TYPE_METADATA_PATH,
    ):
        self.embeddings_path = Path(embeddings_path)
        self.metadata_path = Path(metadata_path)
        self.embeddings, self.metadata = self.load_index()

    def load_index(self):
        if not self.embeddings_path.exists() or not self.metadata_path.exists():
            raise TypeEmbeddingIndexMissing(
                "Type embedding index was not found. Run: python scripts/build_type_embedding_index.py"
            )

        embeddings = np.load(self.embeddings_path).astype("float32")
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))

        if len(embeddings) != len(metadata):
            raise TypeEmbeddingIndexMissing(
                "Type embedding index is invalid: embeddings and metadata lengths do not match."
            )

        kept_indices = [
            index
            for index, record in enumerate(metadata)
            if not cv_service.is_blocked_type_reference_path(record.get("path", ""))
        ]

        if len(kept_indices) != len(metadata):
            embeddings = embeddings[kept_indices]
            metadata = [metadata[index] for index in kept_indices]

        return embeddings, metadata

    def classify_type_icon(self, icon_crop):
        if cv_service.is_empty_image(icon_crop):
            return unknown_type_result("empty-image")

        variants = [
            ("original", icon_crop),
            ("lab_normalized", preprocess_type_icon_for_embedding(icon_crop)),
            ("glare_reduced", preprocess_type_icon_glare_reduced(icon_crop)),
        ]

        best_result = unknown_type_result("no-match")

        for variant_name, variant_image in variants:
            try:
                vector = embed_image(variant_image)
            except Exception:
                continue

            similarities = cosine_similarity_matrix(vector, self.embeddings)

            if len(similarities) == 0:
                continue

            best_index = int(np.argmax(similarities))
            best_similarity = float(similarities[best_index])
            confidence = calculate_confidence_from_similarity(best_similarity)
            distance = 1.0 - best_similarity
            record = self.metadata[best_index]

            candidate = {
                "type": record.get("type", ""),
                "confidence": round(confidence, 4),
                "similarity": round(best_similarity, 4),
                "distance": round(distance, 4),
                "referenceImage": record.get("path", ""),
                "predictionSource": f"type_embedding_{variant_name}",
                "needsReview": confidence < TYPE_CONFIDENCE_THRESHOLD,
            }

            if candidate["confidence"] > best_result["confidence"]:
                best_result = candidate

        if best_result["confidence"] < TYPE_CONFIDENCE_THRESHOLD:
            return {
                **best_result,
                "type": best_result.get("type", ""),
                "needsReview": True,
            }

        return best_result

    def classify_slot_types(self, slot_image, type_icon_crops=None):
        icon_crops = (
            type_icon_crops
            if type_icon_crops is not None
            else cv_service.crop_adaptive_type_icons_from_slot(slot_image)
        )
        detected = []
        details = []
        accepted_results = []

        for icon_crop in icon_crops:
            crop_accepted = cv_service.is_usable_type_icon_candidate(icon_crop)
            if not crop_accepted:
                details.append({
                    **unknown_type_result("rejected-crop-quality"),
                    "index": icon_crop.get("index"),
                    "hasSymbol": bool(icon_crop.get("hasSymbol")),
                    "cropSource": icon_crop.get("cropSource", ""),
                    "cropConfidence": icon_crop.get("confidence"),
                    "cropAccepted": False,
                    "cropQuality": icon_crop.get("cropQuality"),
                })
                continue

            result = self.classify_type_icon(icon_crop["image"])
            detail = {
                **result,
                "index": icon_crop.get("index"),
                "x": icon_crop.get("x"),
                "y": icon_crop.get("y"),
                "width": icon_crop.get("width"),
                "height": icon_crop.get("height"),
                "hasSymbol": bool(icon_crop.get("hasSymbol")),
                "cropSource": icon_crop.get("cropSource", ""),
                "cropConfidence": icon_crop.get("confidence"),
                "cropAccepted": True,
                "cropQuality": icon_crop.get("cropQuality"),
                "selectionScore": round(
                    float(result.get("confidence", 0) or 0)
                    + (float(icon_crop.get("cropQuality", 0) or 0) * 0.12),
                    4,
                ),
            }
            details.append(detail)

            detected_type = result.get("type", "")
            if detected_type and result.get("confidence", 0) >= TYPE_CONFIDENCE_THRESHOLD:
                accepted_results.append(detail)

        accepted_results.sort(
            key=lambda detail: (
                detail.get("selectionScore", 0),
                detail.get("confidence", 0),
            ),
            reverse=True,
        )

        selected_details = []
        for detail in accepted_results:
            if detail.get("type") in detected:
                continue
            if any(type_candidate_overlaps(detail, old) for old in selected_details):
                continue

            detected.append(detail["type"])
            selected_details.append(detail)
            if len(selected_details) == 2:
                break

        selected_details.sort(key=lambda detail: detail.get("x", 0))

        return {
            "selected": cv_service.normalize_detected_type_pair(detected),
            "embeddingDetails": details,
            "selectedEmbeddingDetails": selected_details,
        }


def unknown_type_result(reason):
    return {
        "type": "",
        "confidence": 0.0,
        "similarity": 0.0,
        "distance": 1.0,
        "referenceImage": "",
        "predictionSource": f"type_embedding_{reason}",
        "needsReview": True,
    }


def type_candidate_overlaps(first, second):
    if first.get("x") is None or second.get("x") is None:
        return False

    first_box = {
        "x": first.get("x", 0),
        "y": first.get("y", 0),
        "width": first.get("width", 0),
        "height": first.get("height", 0),
    }
    second_box = {
        "x": second.get("x", 0),
        "y": second.get("y", 0),
        "width": second.get("width", 0),
        "height": second.get("height", 0),
    }
    return cv_service.type_icon_boxes_overlap(first_box, second_box)


@lru_cache(maxsize=1)
def get_type_embedding_service():
    return TypeEmbeddingService()
