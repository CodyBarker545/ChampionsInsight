"""Coordinates card, type, and spirit detection for opponent team images."""

from pathlib import Path

from services import cv_service
from services.cv_card_service import OpponentCardService
from services.cv_spirit_service import PokemonSpiritDetectionService
from services.cv_type_service import PokemonTypeDetectionService


class OpponentDetectionService:
    """Runs the full opponent detection pipeline using focused CV services."""

    # Creates reusable card, type, and spirit services for repeated detections.
    def __init__(
        self,
        card_service=None,
        type_service=None,
        spirit_service=None,
        debug_dir=cv_service.OPPONENT_DEBUG_CROP_DIR,
    ):
        self.debug_dir = Path(debug_dir)
        self.card_service = card_service or OpponentCardService(debug_dir=self.debug_dir)
        self.type_service = type_service or PokemonTypeDetectionService(
            debug_dir=self.debug_dir,
            card_service=self.card_service,
        )
        self.spirit_service = spirit_service or PokemonSpiritDetectionService()

    # Checks whether an opponent image is good enough for detection.
    def assess_quality(self, image_path):
        return self.card_service.assess_quality(image_path)

    # Detects only the type icons from an uploaded opponent team image.
    def detect_team_types(self, image_path, save_debug=True):
        return self.type_service.detect_team_types(image_path, save_debug=save_debug)

    # Detects opponent Pokemon and their types from an uploaded team image.
    # Detects opponent Pokemon and their types from an uploaded team image.
    def detect_team(self, image_path, save_debug=True):
        image_path = Path(image_path)

        quality = self.assess_quality(image_path)

        if not quality.get("canAnalyze", True):
            return {
            "image": str(image_path),
            "referenceCount": len(self.spirit_service.references),
            "quality": quality,
            "debugOriginalPath": "",
            "detectedTeam": [],
            "skippedReason": "bad_quality",
        }

        debug_original_path = ""
        if save_debug:
            debug_original_path = self.write_debug_original_image(image_path)

        card_crops = self.card_service.crop_team_slots(image_path, save_debug=save_debug)
        detected_team = []

        for card_crop in card_crops:
            slot_detection = self.detect_slot(image_path, card_crop, save_debug=save_debug)
            detected_team.append(slot_detection)

        return {
        "image": str(image_path),
        "referenceCount": len(self.spirit_service.references),
        "quality": quality,
        "debugOriginalPath": debug_original_path,
        "detectedTeam": detected_team,
    }
    # Detects one opponent slot using type filtering before spirit matching.
    def detect_slot(self, image_path, card_crop, save_debug=True):
        slot_image = card_crop["image"]
        position = card_crop["position"]

        object_layer = cv_service.detect_slot_object_layer(slot_image)
        type_icon_crops = cv_service.type_icon_crops_from_object_layer(object_layer)
        type_method_results = self.type_service.detect_slot_types(
            slot_image,
            type_icon_crops=type_icon_crops,
        )
        detected_types = type_method_results["selected"]
        object_detections = [
            cv_service.strip_object_image(detected_object)
            for detected_object in cv_service.detect_slot_objects_from_layer(object_layer)
        ]
        pokemon_crop_candidates = [
            cv_service.strip_object_image(candidate)
            for candidate in object_layer.get("candidates", {}).get("pokemon_sprite", [])
        ]
        type_icon_crop_candidates = [
            cv_service.strip_object_image(candidate)
            for candidate in object_layer.get("candidates", {}).get("type_icons", [])
        ]

        pokemon_object = object_layer.get("pokemon_sprite")
        pokemon_region = (
            pokemon_object.get("image")
            if pokemon_object and not cv_service.is_empty_image(pokemon_object.get("image"))
            else self.spirit_service.extract_spirit_region(slot_image)
        )

        debug_pokemon_crop_path = ""
        debug_type_icon_crop_path = ""
        debug_type_icon_crop_paths = []
        debug_object_overlay_path = ""

        if save_debug:
            debug_pokemon_crop_path = self.write_debug_pokemon_crop(
                image_path,
                position,
                pokemon_region,
            )

            debug_type_icon_crop_path, debug_type_icon_crop_paths = self.write_debug_type_crops(
                image_path,
                position,
                slot_image,
                type_icon_crops,
            )
            debug_object_overlay_path = self.write_debug_object_overlay(
                image_path,
                position,
                slot_image,
                object_detections,
            )

        match = self.spirit_service.detect_spirit(
            pokemon_region,
            detected_types=detected_types,
        )

        fixed_match = None

        match_confidence = float(match.get("confidence", 0) or 0)
        should_try_fixed_visual_rescue = (
            match.get("pokemonName") == "unknown"
            or match_confidence < 0.55
            or (len(detected_types or []) <= 1 and match_confidence < 0.72)
        )

        if should_try_fixed_visual_rescue:
            fixed_pokemon_region = cv_service.extract_fixed_opponent_pokemon_region(slot_image)
            fixed_match = self.detect_fixed_visual_match(fixed_pokemon_region)

            if self.should_use_fixed_visual_match(match, fixed_match, detected_types):
                match = {
                    **fixed_match,
                    "matchReason": "fixed-visual-rescue",
                    "predictionSource": "fixed_visual_rescue",
                    "needsReview": False,
                }

        if match.get("matchReason") == "guided-visual-rescue":
            detected_types = match.get("referenceTypes", [])
            type_method_results = {
                **type_method_results,
                "selected": detected_types,
                "referenceFallback": detected_types,
            }

        elif match.get("matchReason") == "fixed-visual-rescue":
            detected_types = match.get("referenceTypes", [])
            type_method_results = {
                **type_method_results,
                "selected": detected_types,
                "referenceFallback": detected_types,
            }

        elif not detected_types and match.get("confidence", 0) >= 0.55:
            detected_types = match.get("referenceTypes", [])
            type_method_results = {
                **type_method_results,
                "selected": detected_types,
                "referenceFallback": detected_types,
            }

        return {
            "position": position,
            "pokemonName": match.get("pokemonName", "unknown"),
            "confidence": match.get("confidence", 0.0),

            # New embedding/debug fields.
            "distance": match.get("distance"),
            "similarity": match.get("similarity"),
            "predictionSource": match.get(
                "predictionSource",
                match.get("matchReason", "visual-match"),
            ),
            "needsReview": match.get(
                "needsReview",
                match.get("confidence", 0) < 0.60,
            ),
            "candidateMode": match.get("candidateMode"),
            "candidateCount": match.get("candidateCount"),
            "typeFilteredBest": match.get("typeFilteredBest"),
            "globalBest": match.get("globalBest"),
            "embeddingBest": match.get("embeddingBest"),
            "templateBest": match.get("templateBest"),

            # Existing fields.
            "detectedTypes": detected_types,
            "referenceTypes": match.get("referenceTypes", []),
            "typeMethodResults": type_method_results,
            "objectDetections": object_detections,
            "pokemonCropCandidates": pokemon_crop_candidates,
            "typeIconCropCandidates": type_icon_crop_candidates,
            "box": card_crop["box"],
            "debugCropPath": card_crop.get("debugCropPath", ""),
            "debugPokemonCropPath": debug_pokemon_crop_path,
            "debugTypeIconCropPath": debug_type_icon_crop_path,
            "debugTypeIconCropPaths": debug_type_icon_crop_paths,
            "debugObjectOverlayPath": debug_object_overlay_path,
            "referenceImage": match.get("referenceImage", ""),
            "matchReason": match.get("matchReason", "visual-match"),
        }

    # Saves the original uploaded image beside its debug crops.
    def write_debug_original_image(self, image_path):
        cv2, _np = cv_service.load_cv_dependencies()
        original = cv_service.read_cv_image(image_path)

        image_debug_dir = self.get_image_debug_dir(image_path)
        output_path = str(image_debug_dir / "original.jpg")

        if not cv2.imwrite(output_path, original):
            raise cv_service.ComputerVisionError(f"Could not write original debug image: {output_path}")

        return output_path

    def should_use_fixed_visual_match(self, match, fixed_match, detected_types):
        if not fixed_match:
            return False

        fixed_confidence = float(fixed_match.get("confidence", 0) or 0)
        current_confidence = float(match.get("confidence", 0) or 0)
        fixed_types = fixed_match.get("referenceTypes", []) or []

        if fixed_match.get("pokemonName") == "unknown" or fixed_confidence < 0.50:
            return False

        if len(fixed_types) < 2:
            return False

        if match.get("pokemonName") == "unknown":
            return fixed_confidence >= 0.52

        if fixed_confidence < current_confidence + 0.08:
            return False

        detected_type_set = set(detected_types or [])
        fixed_type_set = set(fixed_types)
        if detected_type_set and not detected_type_set.intersection(fixed_type_set):
            return fixed_confidence >= 0.62 and fixed_confidence >= current_confidence + 0.14

        if fixed_confidence >= current_confidence + 0.08:
            return True

        return False

    def detect_fixed_visual_match(self, pokemon_region):
        if cv_service.is_empty_image(pokemon_region):
            return None

        slot_match_image = cv_service.preprocess_for_matching(pokemon_region)
        slot_color_match_image = cv_service.preprocess_color_for_matching(pokemon_region)
        best_score = 0.0
        best_reference = None

        for reference in self.spirit_service.references:
            score = cv_service.score_reference_for_visual_preference(
                reference,
                slot_match_image,
                slot_color_match_image,
            )
            if score > best_score:
                best_score = score
                best_reference = reference

        if not best_reference or best_score < 0.50:
            return None

        return {
            "pokemonName": best_reference["name"],
            "confidence": round(best_score, 4),
            "referenceImage": best_reference["path"],
            "referenceTypes": best_reference.get("types", []),
        }

    # Returns a dedicated debug folder for one uploaded image.
    def get_image_debug_dir(self, image_path):
        safe_name = Path(image_path).stem
        image_debug_dir = self.debug_dir / safe_name
        image_debug_dir.mkdir(parents=True, exist_ok=True)
        return image_debug_dir
    
    # Saves the Pokemon spirit crop for debugging and tuning.
    def write_debug_pokemon_crop(self, image_path, position, pokemon_region):
        if cv_service.is_empty_image(pokemon_region):
            return ""

        cv2, _np = cv_service.load_cv_dependencies()
        image_debug_dir = self.get_image_debug_dir(image_path)

        debug_path = str(image_debug_dir / f"opponent-pokemon-{position}.jpg")
        if not cv2.imwrite(debug_path, pokemon_region):
            raise cv_service.ComputerVisionError(
                f"Could not write Pokemon crop: {debug_path}"
            )

        return debug_path

    # Saves the combined type area, the detected type cluster, and each individual type icon crop.
    def write_debug_type_crops(self, image_path, position, slot_image, type_icon_crops=None):
        type_icon_region = self.card_service.extract_type_icon_region(slot_image)
        if cv_service.is_empty_image(type_icon_region):
            return "", []

        cv2, _np = cv_service.load_cv_dependencies()
        image_debug_dir = self.get_image_debug_dir(image_path)

        debug_region = cv_service.enlarge_small_type_icon_region(type_icon_region)
        debug_type_icon_crop_path = str(
            image_debug_dir / f"opponent-type-icons-{position}.jpg"
        )
        if not cv2.imwrite(debug_type_icon_crop_path, debug_region):
            raise cv_service.ComputerVisionError(
                f"Could not write type icon crop: {debug_type_icon_crop_path}"
            )

        debug_type_icon_crop_paths = []

        # Extra debug crop: the full detected type cluster before splitting.
        cluster_box = None
        if hasattr(cv_service, "extract_detected_type_icon_cluster_box"):
            cluster_box = cv_service.extract_detected_type_icon_cluster_box(slot_image)

        if cluster_box:
            cluster_crop = cv_service.crop_image_box(slot_image, cluster_box)
            if not cv_service.is_empty_image(cluster_crop):
                cluster_path = str(
                    image_debug_dir / f"opponent-type-cluster-{position}.jpg"
                )
                if not cv2.imwrite(cluster_path, cluster_crop):
                    raise cv_service.ComputerVisionError(
                        f"Could not write type cluster crop: {cluster_path}"
                    )

                debug_type_icon_crop_paths.append({
                    "path": cluster_path,
                    "hasSymbol": cv_service.has_type_icon_symbol(cluster_crop),
                    "cropSource": "type_icon_cluster_debug",
                    "confidence": None,
                    "box": cluster_box,
                })

        for icon_crop in (
            type_icon_crops
            if type_icon_crops is not None
            else self.type_service.crop_slot_type_icons(slot_image)
        ):
            debug_icon_path = str(
                image_debug_dir / f"opponent-type-icon-{position}-{icon_crop['index']}.jpg"
            )
            if not cv2.imwrite(debug_icon_path, icon_crop["image"]):
                raise cv_service.ComputerVisionError(
                    f"Could not write type icon crop: {debug_icon_path}"
                )

            debug_type_icon_crop_paths.append({
                "path": debug_icon_path,
                "hasSymbol": bool(icon_crop["hasSymbol"]),
                "cropSource": icon_crop.get("cropSource", ""),
                "confidence": icon_crop.get("confidence"),
                "box": {
                    "x": icon_crop.get("x", 0),
                    "y": icon_crop.get("y", 0),
                    "width": icon_crop.get("width", 0),
                    "height": icon_crop.get("height", 0),
                },
            })

        return debug_type_icon_crop_path, debug_type_icon_crop_paths

    # Saves one slot image with detected object boxes drawn over it.
    def write_debug_object_overlay(self, image_path, position, slot_image, object_detections):
        if cv_service.is_empty_image(slot_image):
            return ""

        cv2, _np = cv_service.load_cv_dependencies()
        overlay = slot_image.copy()

        colors = {
            "pokemon_sprite": (0, 220, 255),
            "type_icon_1": (0, 255, 0),
            "type_icon_2": (255, 180, 0),
            "type_cluster": (255, 0, 255),
        }

        # Draw the full detected type cluster first, so individual boxes are visible on top.
        cluster_box = None
        if hasattr(cv_service, "extract_detected_type_icon_cluster_box"):
            cluster_box = cv_service.extract_detected_type_icon_cluster_box(slot_image)

        if cluster_box:
            x1 = int(cluster_box.get("x", 0))
            y1 = int(cluster_box.get("y", 0))
            x2 = x1 + int(cluster_box.get("width", 0))
            y2 = y1 + int(cluster_box.get("height", 0))
            color = colors["type_cluster"]

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                overlay,
                "type_cluster",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        for detected_object in object_detections:
            box = detected_object.get("box") or {}
            label = detected_object.get("label", "object")
            if not box:
                continue

            x1 = int(box.get("x", 0))
            y1 = int(box.get("y", 0))
            x2 = x1 + int(box.get("width", 0))
            y2 = y1 + int(box.get("height", 0))
            color = colors.get(label, (255, 255, 255))

            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                overlay,
                label,
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )

        image_debug_dir = self.get_image_debug_dir(image_path)
        debug_path = str(image_debug_dir / f"opponent-objects-{position}.jpg")
        if not cv2.imwrite(debug_path, overlay):
            raise cv_service.ComputerVisionError(
                f"Could not write object overlay: {debug_path}"
            )

        return debug_path


_DEFAULT_DETECTION_SERVICE = None


# Returns a cached detection service so references load once per process.
def get_detection_service():
    global _DEFAULT_DETECTION_SERVICE

    if _DEFAULT_DETECTION_SERVICE is None:
        _DEFAULT_DETECTION_SERVICE = OpponentDetectionService()

    return _DEFAULT_DETECTION_SERVICE


# Checks whether an uploaded opponent image is good enough for detection.
def assess_opponent_image_quality(image_path):
    return get_detection_service().assess_quality(image_path)


# Detects only type icons from an uploaded opponent team image.
def detect_opponent_team_types(image_path, save_debug=True):
    return get_detection_service().detect_team_types(image_path, save_debug=save_debug)


# Detects opponent Pokemon and types from an uploaded opponent team image.
def detect_opponent_team(image_path, save_debug=True):
    return get_detection_service().detect_team(image_path, save_debug=save_debug)
