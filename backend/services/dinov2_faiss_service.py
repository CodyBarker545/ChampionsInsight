"""DINOv2 image embeddings with FAISS nearest-neighbor indexes."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_DINOV2_MODEL = "facebook/dinov2-small"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class Dinov2FaissIndexMissing(RuntimeError):
    """Raised when a DINOv2/FAISS index cannot be loaded."""


def require_faiss():
    try:
        import faiss
    except ImportError as error:
        raise RuntimeError(
            "faiss is not installed. Install it with: pip install faiss-cpu"
        ) from error
    return faiss


@lru_cache(maxsize=2)
def load_dinov2_model(model_name: str = DEFAULT_DINOV2_MODEL):
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as error:
        raise RuntimeError(
            "DINOv2 dependencies are missing. Install torch and transformers."
        ) from error

    try:
        processor = AutoImageProcessor.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(model_name, local_files_only=True)
    except OSError:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return processor, model, torch, device


def center_pad_square(image: Image.Image, background=(16, 16, 16)) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    side = max(width, height)

    padded = Image.new("RGB", (side, side), background)
    padded.paste(image, ((side - width) // 2, (side - height) // 2))
    return padded


def load_pil_image(path: Path) -> Image.Image:
    return center_pad_square(Image.open(path).convert("RGB"))


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype("float32", copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms <= 0] = 1.0
    return matrix / norms


def embed_images(
    images: list[Image.Image],
    model_name: str = DEFAULT_DINOV2_MODEL,
) -> np.ndarray:
    if not images:
        return np.empty((0, 0), dtype="float32")

    processor, model, torch, device = load_dinov2_model(model_name)
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        vectors = outputs.last_hidden_state[:, 0, :]

    matrix = vectors.detach().cpu().numpy().astype("float32")
    return l2_normalize(matrix)


def write_faiss_index(embeddings: np.ndarray, output_path: Path) -> None:
    faiss = require_faiss()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if embeddings.ndim != 2 or embeddings.shape[0] == 0:
        raise ValueError("Cannot build an empty FAISS index.")

    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, str(output_path))


def read_faiss_index(index_path: Path):
    faiss = require_faiss()
    if not index_path.exists():
        raise Dinov2FaissIndexMissing(f"Missing FAISS index: {index_path}")
    return faiss.read_index(str(index_path))


def write_metadata(metadata: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_metadata(metadata_path: Path) -> list[dict]:
    if not metadata_path.exists():
        raise Dinov2FaissIndexMissing(f"Missing metadata: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


class Dinov2FaissIndex:
    """Thin runtime wrapper around a DINOv2 FAISS cosine-similarity index."""

    def __init__(
        self,
        index_path: Path,
        metadata_path: Path,
        model_name: str = DEFAULT_DINOV2_MODEL,
    ):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.model_name = model_name
        self.index = read_faiss_index(self.index_path)
        self.metadata = read_metadata(self.metadata_path)

        if self.index.ntotal != len(self.metadata):
            raise Dinov2FaissIndexMissing(
                f"Index/metadata length mismatch: {self.index.ntotal} vs {len(self.metadata)}"
            )

    def search(self, image, top_k: int = 5) -> list[dict]:
        if isinstance(image, Image.Image):
            pil_image = center_pad_square(image)
        else:
            pil_image = load_pil_image(Path(image))

        query = embed_images([pil_image], model_name=self.model_name)
        return self.search_embeddings(query, top_k=top_k)[0]

    def search_many(self, images: list[Image.Image], top_k: int = 5) -> list[list[dict]]:
        pil_images = [
            center_pad_square(image) if isinstance(image, Image.Image) else load_pil_image(Path(image))
            for image in images
        ]
        query = embed_images(pil_images, model_name=self.model_name)
        return self.search_embeddings(query, top_k=top_k)

    def search_embeddings(self, query, top_k: int = 5) -> list[list[dict]]:
        scores, indices = self.index.search(query.astype("float32"), top_k)

        all_results = []
        for result_scores, result_indices in zip(scores, indices):
            results = []
            for score, index in zip(result_scores, result_indices):
                if index < 0:
                    continue
                record = dict(self.metadata[int(index)])
                record["similarity"] = float(score)
                record["confidence"] = max(0.0, min(1.0, (float(score) + 1.0) / 2.0))
                results.append(record)
            all_results.append(results)

        return all_results
