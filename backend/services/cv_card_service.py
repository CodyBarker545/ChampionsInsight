"""Handles opponent card image loading, cropping, and quality checks."""

from pathlib import Path

from services import cv_service


class OpponentCardService:
    """Provides card-level computer vision operations for opponent team images."""

    # Stores the debug output directory used when saving card crops.
    def __init__(self, debug_dir=cv_service.OPPONENT_DEBUG_CROP_DIR):
        self.debug_dir = Path(debug_dir)

    # Reads an uploaded image and returns quality guidance for the user.
    def assess_quality(self, image_path):
        return cv_service.assess_opponent_image_quality(image_path)

    # Crops the six opponent card slots from an uploaded team image.
    def crop_team_slots(self, image_path, save_debug=False):
        return cv_service.crop_opponent_team_slots(image_path, save_debug=save_debug)

    # Straightens a detected card crop when there is enough skew to justify it.
    def rectify_card_crop(self, card_crop):
        return cv_service.rectify_opponent_card_crop(card_crop)

    # Returns the combined type-icon area from one opponent card crop.
    def extract_type_icon_region(self, slot_image):
        return cv_service.extract_type_icon_region(slot_image)
