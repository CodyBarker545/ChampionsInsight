"""Inference helpers for the supervised MobileNetV3 Pokemon crop classifier."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any
from threading import Lock

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    mobilenet_v3_small,
)

from paths import POKEMON_MOBILENETV3_MODEL_DIR


MODEL_NAME = "mobilenetv3"
DEFAULT_VARIANT = "small"
DEFAULT_IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_FILENAME = "pokemon_mobilenetv3.pt"
IDX_TO_CLASS_FILENAME = "idx_to_class.json"
CLASS_TO_IDX_FILENAME = "class_to_idx.json"
_MODEL_LOAD_LOCK = Lock()


def get_default_model_dir() -> Path:
    return POKEMON_MOBILENETV3_MODEL_DIR


def get_model_artifact_paths(model_dir: str | Path | None = None) -> dict[str, Path]:
    root = Path(model_dir) if model_dir is not None else get_default_model_dir()
    return {
        "root": root,
        "model": root / MODEL_FILENAME,
        "idx_to_class": root / IDX_TO_CLASS_FILENAME,
        "class_to_idx": root / CLASS_TO_IDX_FILENAME,
    }


def get_eval_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_mobilenetv3_model(
    num_classes: int,
    variant: str = DEFAULT_VARIANT,
    use_pretrained: bool = True,
):
    variant_normalized = str(variant).strip().lower()

    if variant_normalized == "large":
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if use_pretrained else None
        model = _load_model_with_optional_weights(
            mobilenet_v3_large,
            weights,
            mobilenet_v3_large,
        )
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
        return model

    if variant_normalized == "small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if use_pretrained else None
        model = _load_model_with_optional_weights(
            mobilenet_v3_small,
            weights,
            mobilenet_v3_small,
        )
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
        return model

    raise ValueError(f"Unsupported MobileNetV3 variant: {variant}")


def _load_model_with_optional_weights(builder, weights, fallback_builder):
    if weights is not None:
        try:
            return builder(weights=weights)
        except Exception:
            pass
    return fallback_builder(weights=None)


def _coerce_pil_image(image_or_path: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image_or_path, Image.Image):
        image = image_or_path
    elif isinstance(image_or_path, (str, Path)):
        image = Image.open(Path(image_or_path))
    elif isinstance(image_or_path, np.ndarray):
        array = image_or_path
        if array.ndim == 2:
            image = Image.fromarray(array).convert("RGB")
        elif array.ndim == 3 and array.shape[2] == 4:
            image = Image.fromarray(array[:, :, [2, 1, 0, 3]], mode="RGBA").convert("RGB")
        elif array.ndim == 3 and array.shape[2] == 3:
            image = Image.fromarray(array[:, :, ::-1]).convert("RGB")
        else:
            raise ValueError("Unsupported numpy image shape for MobileNetV3 prediction.")
    else:
        raise TypeError("image_or_path must be a file path, PIL image, or OpenCV/numpy image.")

    return image.convert("RGB")


def _empty_prediction(error_message: str) -> dict[str, Any]:
    return {
        "pokemon": None,
        "confidence": 0.0,
        "top5": [],
        "model": MODEL_NAME,
        "error": error_message,
        "productionDecision": "needs_review",
    }


def _select_inference_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@lru_cache(maxsize=4)
def _load_model_bundle(model_dir: str) -> dict[str, Any]:
    artifacts = get_model_artifact_paths(model_dir)
    model_path = artifacts["model"]
    idx_to_class_path = artifacts["idx_to_class"]

    if not model_path.exists():
        raise FileNotFoundError(
            f"MobileNetV3 model checkpoint was not found at {model_path}. "
            "Train it with backend/scripts/train_mobilenetv3_pokemon_classifier.py first."
        )

    if not idx_to_class_path.exists():
        raise FileNotFoundError(
            f"Class mapping was not found at {idx_to_class_path}. "
            "Train it with backend/scripts/train_mobilenetv3_pokemon_classifier.py first."
        )

    device = _select_inference_device()
    checkpoint = torch.load(model_path, map_location=device)
    idx_to_class = json.loads(idx_to_class_path.read_text(encoding="utf-8"))
    if not isinstance(idx_to_class, dict) or not idx_to_class:
        raise ValueError("idx_to_class.json is empty or invalid.")

    normalized_idx_to_class = {
        int(index): str(name)
        for index, name in idx_to_class.items()
    }
    num_classes = len(normalized_idx_to_class)
    variant = str(checkpoint.get("variant", DEFAULT_VARIANT))
    image_size = int(checkpoint.get("image_size", DEFAULT_IMAGE_SIZE))

    model = build_mobilenetv3_model(
        num_classes=num_classes,
        variant=variant,
        use_pretrained=False,
    )

    state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
    if not state_dict:
        raise ValueError(f"Checkpoint at {model_path} is missing model weights.")

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return {
        "model": model,
        "idx_to_class": normalized_idx_to_class,
        "variant": variant,
        "image_size": image_size,
        "device": device,
    }


def clear_model_cache() -> None:
    with _MODEL_LOAD_LOCK:
        _load_model_bundle.cache_clear()


def _get_model_bundle(model_dir):
    # lru_cache alone allows duplicate loads when warmup and upload race.
    with _MODEL_LOAD_LOCK:
        return _load_model_bundle(model_dir)


def suggest_prediction_action(prediction: dict[str, Any]) -> str:
    """Future integration hint for blending MobileNetV3 with FAISS and type cues."""
    confidence = float(prediction.get("confidence", 0.0) or 0.0)

    if confidence >= 0.85:
        return "accept_mobilenetv3"
    if confidence >= 0.60:
        return "compare_with_faiss_and_type"
    return "needs_review"


def predict_pokemon_crop(
    image_or_path: str | Path | Image.Image | np.ndarray,
    top_k: int = 5,
    model_dir: str | Path | None = None,
) -> dict[str, Any]:
    model_root = str((Path(model_dir) if model_dir is not None else get_default_model_dir()).resolve())

    try:
        bundle = _get_model_bundle(model_root)
    except (FileNotFoundError, ValueError) as error:
        return _empty_prediction(str(error))

    try:
        image = _coerce_pil_image(image_or_path)
    except (OSError, TypeError, ValueError) as error:
        return _empty_prediction(f"Could not prepare Pokemon crop for prediction: {error}")

    transform = get_eval_transform(image_size=bundle["image_size"])
    input_tensor = transform(image).unsqueeze(0).to(bundle["device"])

    with torch.inference_mode():
        logits = bundle["model"](input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        class_count = probabilities.shape[1]
        effective_top_k = max(1, min(int(top_k), class_count))
        confidence_values, class_indices = torch.topk(probabilities, k=effective_top_k, dim=1)

    top_predictions = []
    for confidence, class_index in zip(confidence_values[0].tolist(), class_indices[0].tolist()):
        top_predictions.append(
            {
                "name": bundle["idx_to_class"][int(class_index)],
                "confidence": round(float(confidence), 6),
            }
        )

    best_prediction = top_predictions[0]
    result = {
        "pokemon": best_prediction["name"],
        "confidence": best_prediction["confidence"],
        "top5": top_predictions,
        "model": MODEL_NAME,
        "variant": bundle["variant"],
        "device": str(bundle["device"]),
    }
    result["productionDecision"] = suggest_prediction_action(result)
    return result


def predict_pokemon_crops(
    images_or_paths: list[str | Path | Image.Image | np.ndarray],
    top_k: int = 5,
    model_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Predict several already-cropped Pokemon images with one batched model pass."""
    if not images_or_paths:
        return []

    model_root = str((Path(model_dir) if model_dir is not None else get_default_model_dir()).resolve())

    try:
        bundle = _get_model_bundle(model_root)
    except (FileNotFoundError, ValueError) as error:
        return [_empty_prediction(str(error)) for _ in images_or_paths]

    transform = get_eval_transform(image_size=bundle["image_size"])
    prepared_tensors = []
    prepared_indices = []
    results: list[dict[str, Any] | None] = [None] * len(images_or_paths)

    for index, image_or_path in enumerate(images_or_paths):
        try:
            image = _coerce_pil_image(image_or_path)
            prepared_tensors.append(transform(image))
            prepared_indices.append(index)
        except (OSError, TypeError, ValueError) as error:
            results[index] = _empty_prediction(f"Could not prepare Pokemon crop for prediction: {error}")

    if prepared_tensors:
        input_tensor = torch.stack(prepared_tensors, dim=0).to(bundle["device"])

        with torch.inference_mode():
            logits = bundle["model"](input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            class_count = probabilities.shape[1]
            effective_top_k = max(1, min(int(top_k), class_count))
            confidence_values, class_indices = torch.topk(probabilities, k=effective_top_k, dim=1)

        for batch_index, result_index in enumerate(prepared_indices):
            top_predictions = []
            for confidence, class_index in zip(
                confidence_values[batch_index].detach().cpu().tolist(),
                class_indices[batch_index].detach().cpu().tolist(),
            ):
                top_predictions.append({
                    "name": bundle["idx_to_class"][int(class_index)],
                    "confidence": round(float(confidence), 6),
                })

            best_prediction = top_predictions[0]
            result = {
                "pokemon": best_prediction["name"],
                "confidence": best_prediction["confidence"],
                "top5": top_predictions,
                "model": MODEL_NAME,
                "variant": bundle["variant"],
                "device": str(bundle["device"]),
            }
            result["productionDecision"] = suggest_prediction_action(result)
            results[result_index] = result

    return [result or _empty_prediction("Prediction was not generated.") for result in results]
