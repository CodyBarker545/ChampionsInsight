"""Shared image embedding model utilities for Pokémon and type icon matching."""

from functools import lru_cache

import numpy as np
from PIL import Image


EMBEDDING_IMAGE_SIZE = 224


@lru_cache(maxsize=1)
def load_embedding_model():
    """Loads a lightweight pretrained MobileNetV3 feature extractor once."""
    try:
        import torch
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
    except ImportError as error:
        raise RuntimeError(
            "Embedding dependencies are missing. Install torch, torchvision, pillow, numpy, and scikit-learn."
        ) from error

    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)

    # Remove the classifier. Keep convolutional feature extractor + pooling.
    model.classifier = torch.nn.Identity()
    model.eval()

    return model, weights.transforms(), torch


def convert_cv_image_to_pil(image):
    """Converts an OpenCV BGR image or path-like image into RGB PIL format."""
    if image is None:
        raise ValueError("Cannot embed an empty image.")

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    try:
        import cv2
    except ImportError as error:
        raise RuntimeError("OpenCV is required to convert CV images.") from error

    if isinstance(image, np.ndarray):
        if image.size == 0:
            raise ValueError("Cannot embed an empty image.")

        if len(image.shape) == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return Image.fromarray(rgb).convert("RGB")

    return Image.open(image).convert("RGB")


def center_pad_square(pil_image, background=(16, 16, 16)):
    """Pads a PIL image to a square so sprites/icons are not distorted."""
    image = pil_image.convert("RGB")
    width, height = image.size
    side = max(width, height)

    padded = Image.new("RGB", (side, side), background)
    left = (side - width) // 2
    top = (side - height) // 2
    padded.paste(image, (left, top))

    return padded


def embed_image(image):
    """Creates a normalized embedding vector for one image."""
    model, transform, torch = load_embedding_model()

    pil_image = convert_cv_image_to_pil(image)
    pil_image = center_pad_square(pil_image)
    tensor = transform(pil_image).unsqueeze(0)

    with torch.no_grad():
        vector = model(tensor).detach().cpu().numpy()[0]

    vector = vector.astype("float32")
    norm = np.linalg.norm(vector)

    if norm <= 0:
        return vector

    return vector / norm


def cosine_similarity_matrix(query_vector, matrix):
    """Returns cosine similarities between one normalized vector and a matrix."""
    if matrix is None or len(matrix) == 0:
        return np.array([], dtype="float32")

    query = query_vector.astype("float32")
    matrix = matrix.astype("float32")

    return matrix @ query