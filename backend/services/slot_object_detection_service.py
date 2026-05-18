"""Named entrypoints for detecting objects inside one red opponent slot card."""

from __future__ import annotations

from services import cv_service


def detect_slot_objects(slot_image):
    """Return the detected Pokémon sprite and individual type-icon objects."""
    return cv_service.detect_slot_object_layer(slot_image)


def get_type_icon_crops(object_layer):
    """Return the individual type-icon crops from a detected slot object layer."""
    return cv_service.type_icon_crops_from_object_layer(object_layer)
