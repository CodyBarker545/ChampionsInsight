"""Train a supervised MobileNetV3 classifier for cropped Pokemon images."""

from __future__ import annotations

import argparse
import gc
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
import random
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from PIL import Image, ImageFile, UnidentifiedImageError
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_ROOT = SCRIPT_DIR.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from paths import BACKEND_DIR, POKEMON_MOBILENETV3_MODEL_DIR
from services.pokemon_mobilenetv3_classifier_service import (
    DEFAULT_IMAGE_SIZE,
    build_mobilenetv3_model,
    get_eval_transform,
)


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
DEFAULT_OUTPUT_DIR = POKEMON_MOBILENETV3_MODEL_DIR
DEFAULT_SYNTHETIC_ROOT = BACKEND_DIR / "data" / "training_dataset" / "slot_pokemon_synthetic"
DEFAULT_REAL_ROOT_CANDIDATES = [
    BACKEND_DIR / "data" / "training_dataset" / "slot_pokemon_real",
    BACKEND_DIR / "data" / "cv" / "debug_crops" / "pokemon",
    BACKEND_DIR / "data" / "cv" / "debug" / "crops" / "pokemon",
    BACKEND_DIR / "cv" / "debug_crops" / "pokemon",
    BACKEND_DIR / "uploads" / "pokemon",
]

ImageFile.LOAD_TRUNCATED_IMAGES = True


def default_num_workers() -> int:
    if os.name == "nt":
        return min(2, max((os.cpu_count() or 2) - 1, 0))
    return min(8, max((os.cpu_count() or 2) - 1, 1))


def resolve_user_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    if path.exists():
        return path.resolve()
    backend_relative = (BACKEND_DIR / path).resolve()
    if backend_relative.exists():
        return backend_relative
    return path.resolve()


@dataclass(frozen=True)
class ImageSample:
    image_path: Path
    label: str
    source: str
    source_kind: str


class PokemonImageDataset(Dataset):
    def __init__(
        self,
        samples: list[ImageSample],
        class_to_idx: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.bad_indices: set[int] = set()
        self.bad_paths: set[str] = set()
        self.max_logged_bad_samples = 20

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample_count = len(self.samples)
        if sample_count == 0:
            raise RuntimeError("PokemonImageDataset is empty.")

        for offset in range(sample_count):
            candidate_index = (index + offset) % sample_count
            if candidate_index in self.bad_indices:
                continue

            sample = self.samples[candidate_index]
            try:
                with Image.open(sample.image_path) as image:
                    image_tensor = self.transform(image.convert("RGB"))
                target = self.class_to_idx[sample.label]
                return image_tensor, target, str(sample.image_path)
            except (UnidentifiedImageError, OSError, ValueError) as error:
                self.bad_indices.add(candidate_index)
                bad_path = str(sample.image_path)
                if bad_path not in self.bad_paths and len(self.bad_paths) < self.max_logged_bad_samples:
                    self.bad_paths.add(bad_path)
                    print(f"[dataset] Skipping unreadable image: {bad_path} ({error})")
                continue

        raise RuntimeError(
            "No readable images remain in the dataset. "
            "Check the dataset roots for corrupt or mislabeled files."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MobileNetV3 Pokemon crop classifier.")
    parser.add_argument(
        "--synthetic-root",
        action="append",
        default=[],
        help="Synthetic dataset root. Can be passed more than once.",
    )
    parser.add_argument(
        "--real-root",
        action="append",
        default=[],
        help="Real/debug dataset root. Can be passed more than once.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where model artifacts and the PDF report will be saved.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-images-per-class", type=int, default=5)
    parser.add_argument(
        "--target-images-per-class",
        type=int,
        default=None,
        help="If set, keep all real images and randomly fill each class up to this many total images using synthetic samples.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=default_num_workers(),
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=1 if os.name == "nt" else 2,
        help="Batches prefetched per DataLoader worker. Lower values use less shared memory.",
    )
    parser.add_argument("--persistent-workers", action="store_true", help="Keep loader workers between epochs (uses more RAM across train/validation loaders).")
    parser.add_argument("--channels-last", action="store_true", help="Use channels-last tensors; benchmark this option on the training GPU first.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", choices=["small", "large"], default="small")
    parser.add_argument("--init-checkpoint", type=Path, help="Initialize from a trained Pokemon model, retaining output rows for matching labels.")
    parser.add_argument("--evaluate-only", action="store_true", help="Evaluate --init-checkpoint and export reports without further optimization.")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=500,
        help="Print a batch progress update every N batches.",
    )
    parser.add_argument(
        "--disable-amp",
        action="store_true",
        help="Disable CUDA automatic mixed precision.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device to use, for example cpu or cuda.",
    )
    return parser.parse_args()


def resolve_dataset_roots(input_roots: Iterable[str], defaults: Iterable[Path]) -> list[Path]:
    resolved = []
    seen = set()

    roots = [Path(value) for value in input_roots] if input_roots else list(defaults)
    for root in roots:
        root_path = resolve_user_path(root)
        if root_path in seen:
            continue
        seen.add(root_path)
        if root_path.exists():
            resolved.append(root_path)
        else:
            print(f"[dataset] Skipping missing root: {root_path}")

    return resolved


def discover_default_real_roots() -> list[Path]:
    return [path for path in DEFAULT_REAL_ROOT_CANDIDATES if path.exists()]


def collect_image_samples(
    dataset_roots: Iterable[Path],
    min_images_per_class: int = 5,
    source_kind: str = "mixed",
) -> tuple[list[ImageSample], dict[str, Any]]:
    samples_by_label: dict[str, list[ImageSample]] = {}
    visited_paths: set[Path] = set()
    scanned_roots = []
    skipped_unlabeled = 0

    for root in dataset_roots:
        scanned_roots.append(str(root))
        source_name = root.name
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            resolved_path = path.resolve()
            if resolved_path in visited_paths:
                continue
            visited_paths.add(resolved_path)

            if path.parent == root:
                skipped_unlabeled += 1
                continue

            label = path.parent.name.strip().lower()
            if not label:
                skipped_unlabeled += 1
                continue

            samples_by_label.setdefault(label, []).append(
                ImageSample(
                    image_path=resolved_path,
                    label=label,
                    source=source_name,
                    source_kind=source_kind,
                )
            )

    filtered_samples: list[ImageSample] = []
    skipped_classes: dict[str, int] = {}
    class_counts: dict[str, int] = {}

    for label, label_samples in sorted(samples_by_label.items()):
        if len(label_samples) < min_images_per_class:
            skipped_classes[label] = len(label_samples)
            continue
        filtered_samples.extend(sorted(label_samples, key=lambda sample: str(sample.image_path)))
        class_counts[label] = len(label_samples)

    summary = {
        "roots": scanned_roots,
        "class_counts": class_counts,
        "skipped_classes": skipped_classes,
        "skipped_unlabeled_images": skipped_unlabeled,
        "total_images": len(filtered_samples),
        "num_classes": len(class_counts),
        "min_images_per_class": min(class_counts.values()) if class_counts else 0,
        "max_images_per_class": max(class_counts.values()) if class_counts else 0,
    }
    return filtered_samples, summary


def is_shiny_sample(sample: ImageSample) -> bool:
    return "shiny" in sample.image_path.stem.lower()


def balance_samples_with_real_priority(
    real_samples: list[ImageSample],
    synthetic_samples: list[ImageSample],
    target_images_per_class: int,
    seed: int,
) -> tuple[list[ImageSample], dict[str, Any]]:
    rng = random.Random(seed)
    real_by_label: dict[str, list[ImageSample]] = {}
    synthetic_by_label: dict[str, list[ImageSample]] = {}

    for sample in real_samples:
        real_by_label.setdefault(sample.label, []).append(sample)
    for sample in synthetic_samples:
        synthetic_by_label.setdefault(sample.label, []).append(sample)

    all_labels = sorted(set(real_by_label) | set(synthetic_by_label))
    selected_samples: list[ImageSample] = []
    per_class_summary: dict[str, dict[str, int]] = {}

    for label in all_labels:
        selected_real = list(real_by_label.get(label, []))
        available_synthetic = list(synthetic_by_label.get(label, []))
        synthetic_needed = max(0, target_images_per_class - len(selected_real))
        selected_synthetic: list[ImageSample] = []

        if synthetic_needed > 0 and available_synthetic:
            shiny_pool = [sample for sample in available_synthetic if is_shiny_sample(sample)]
            non_shiny_pool = [sample for sample in available_synthetic if not is_shiny_sample(sample)]
            synthetic_take = min(synthetic_needed, len(available_synthetic))
            shiny_ratio = len(shiny_pool) / max(len(available_synthetic), 1)
            shiny_target = min(len(shiny_pool), round(synthetic_take * shiny_ratio))
            non_shiny_target = min(len(non_shiny_pool), synthetic_take - shiny_target)

            if shiny_target > 0:
                selected_synthetic.extend(rng.sample(shiny_pool, shiny_target))
            if non_shiny_target > 0:
                selected_synthetic.extend(rng.sample(non_shiny_pool, non_shiny_target))

            remaining_needed = synthetic_take - len(selected_synthetic)
            if remaining_needed > 0:
                selected_paths = {sample.image_path for sample in selected_synthetic}
                remaining_pool = [
                    sample for sample in available_synthetic
                    if sample.image_path not in selected_paths
                ]
                if remaining_pool:
                    selected_synthetic.extend(
                        rng.sample(remaining_pool, min(remaining_needed, len(remaining_pool)))
                    )

        selected_for_label = selected_real + selected_synthetic
        selected_samples.extend(selected_for_label)
        per_class_summary[label] = {
            "real_selected": len(selected_real),
            "synthetic_selected": len(selected_synthetic),
            "synthetic_shiny_selected": sum(1 for sample in selected_synthetic if is_shiny_sample(sample)),
            "synthetic_non_shiny_selected": sum(1 for sample in selected_synthetic if not is_shiny_sample(sample)),
            "total_selected": len(selected_for_label),
            "synthetic_available": len(available_synthetic),
        }

    selected_samples.sort(key=lambda sample: (sample.label, str(sample.image_path)))
    summary = {
        "strategy": "all_real_plus_synthetic_fill",
        "target_images_per_class": target_images_per_class,
        "per_class": per_class_summary,
        "total_selected_images": len(selected_samples),
        "total_real_selected": sum(item["real_selected"] for item in per_class_summary.values()),
        "total_synthetic_selected": sum(item["synthetic_selected"] for item in per_class_summary.values()),
    }
    return selected_samples, summary


def create_class_mappings(samples: Iterable[ImageSample]) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted({sample.label for sample in samples})
    class_to_idx = {label: index for index, label in enumerate(labels)}
    idx_to_class = {index: label for label, index in class_to_idx.items()}
    return class_to_idx, idx_to_class


def filter_samples_by_min_count(
    samples: list[ImageSample],
    min_images_per_class: int,
) -> tuple[list[ImageSample], dict[str, int]]:
    counts = Counter(sample.label for sample in samples)
    skipped_classes = {
        label: count
        for label, count in sorted(counts.items())
        if count < min_images_per_class
    }
    filtered_samples = [
        sample for sample in samples
        if counts[sample.label] >= min_images_per_class
    ]
    return filtered_samples, skipped_classes


def split_samples(
    samples: list[ImageSample],
    seed: int,
) -> tuple[list[ImageSample], list[ImageSample], list[ImageSample]]:
    labels = [sample.label for sample in samples]
    try:
        train_samples, temp_samples = train_test_split(
            samples,
            test_size=0.30,
            random_state=seed,
            stratify=labels,
        )
        temp_labels = [sample.label for sample in temp_samples]
        val_samples, test_samples = train_test_split(
            temp_samples,
            test_size=0.50,
            random_state=seed,
            stratify=temp_labels,
        )
    except ValueError as error:
        raise ValueError(
            "Stratified splitting failed. Increase the dataset size or raise "
            "--min-images-per-class so each label has enough examples."
        ) from error

    return train_samples, val_samples, test_samples


def build_train_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    augmentations: list[Any] = [
        transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.95, 1.05))],
            p=0.7,
        ),
        transforms.RandomHorizontalFlip(p=0.25),
        transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.2))], p=0.15),
        transforms.RandomPerspective(distortion_scale=0.12, p=0.12),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    return transforms.Compose(augmentations)


def make_data_loader(
    samples: list[ImageSample],
    class_to_idx: dict[str, int],
    transform: transforms.Compose,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    prefetch_factor: int,
    persistent_workers: bool = False,
) -> DataLoader:
    dataset = PokemonImageDataset(samples, class_to_idx, transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_training_device(device_arg: str) -> torch.device:
    normalized = str(device_arg).strip().lower()

    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is not available in this PyTorch install. "
                "Install a CUDA-enabled PyTorch build or run with --device cpu."
            )
        return torch.device(device_arg)

    return torch.device(device_arg)


def run_epoch(
    model: torch.nn.Module,
    data_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    use_amp: bool = False,
    log_interval: int = 500,
    epoch_label: str = "epoch",
    channels_last: bool = False,
) -> tuple[float, float]:
    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    total_batches = len(data_loader)
    phase_name = "train" if is_training else "val"
    phase_start_time = time.perf_counter()

    for batch_index, (images, targets, _paths) in enumerate(data_loader, start=1):
        images = images.to(device, non_blocking=device.type == "cuda")
        if device.type == "cuda" and channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        targets = targets.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, targets)

            if is_training:
                if scaler is not None and use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        predictions = torch.argmax(logits, dim=1)
        total_loss += float(loss.item()) * images.size(0)
        total_correct += int((predictions == targets).sum().item())
        total_examples += int(images.size(0))

        if log_interval > 0 and (batch_index % log_interval == 0 or batch_index == total_batches):
            elapsed = time.perf_counter() - phase_start_time
            running_loss = total_loss / max(total_examples, 1)
            running_accuracy = total_correct / max(total_examples, 1)
            batches_per_second = batch_index / max(elapsed, 1e-6)
            print(
                f"[{epoch_label}] {phase_name} batch {batch_index}/{total_batches} "
                f"loss={running_loss:.4f} acc={running_accuracy:.4f} "
                f"batches_per_sec={batches_per_second:.2f}"
            )

    average_loss = total_loss / max(total_examples, 1)
    accuracy = total_correct / max(total_examples, 1)
    return average_loss, accuracy


def evaluate_model(
    model: torch.nn.Module,
    data_loader: DataLoader,
    idx_to_class: dict[int, str],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    prediction_entries: list[dict[str, Any]] = []
    y_true: list[int] = []
    y_pred: list[int] = []
    top3_hits = 0
    top5_hits = 0

    with torch.inference_mode():
        for images, targets, image_paths in data_loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            top1 = torch.argmax(probabilities, dim=1)
            top3_count = min(3, probabilities.shape[1])
            top5_count = min(5, probabilities.shape[1])
            top3_indices = torch.topk(probabilities, k=top3_count, dim=1).indices
            top5_values, top5_indices = torch.topk(probabilities, k=top5_count, dim=1)

            batch_targets = targets.cpu().tolist()
            batch_predictions = top1.cpu().tolist()
            batch_top5_values = top5_values.cpu().tolist()
            batch_top5_indices = top5_indices.cpu().tolist()
            y_true.extend(batch_targets)
            y_pred.extend(batch_predictions)
            top3_hits += int((top3_indices == targets.unsqueeze(1)).any(dim=1).sum().item())
            top5_hits += int((top5_indices == targets.unsqueeze(1)).any(dim=1).sum().item())

            for batch_index, image_path in enumerate(image_paths):
                top5 = []
                for confidence, class_index in zip(
                    batch_top5_values[batch_index],
                    batch_top5_indices[batch_index],
                ):
                    top5.append([idx_to_class[int(class_index)], round(float(confidence), 6)])

                prediction_entries.append(
                    {
                        "image_path": image_path,
                        "true_label": idx_to_class[int(batch_targets[batch_index])],
                        "predicted_label": idx_to_class[int(batch_predictions[batch_index])],
                        "confidence": round(float(top5[0][1]), 6),
                        "top5": top5,
                    }
                )

    report = classification_report(
        y_true,
        y_pred,
        target_names=[idx_to_class[index] for index in sorted(idx_to_class)],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "test_accuracy": round(float(report["accuracy"]), 6),
        "top3_accuracy": round(top3_hits / max(len(y_true), 1), 6),
        "top5_accuracy": round(top5_hits / max(len(y_true), 1), 6),
        "macro_precision": round(float(report["macro avg"]["precision"]), 6),
        "macro_recall": round(float(report["macro avg"]["recall"]), 6),
        "macro_f1": round(float(report["macro avg"]["f1-score"]), 6),
        "classification_report": report,
        "y_true": y_true,
        "y_pred": y_pred,
    }
    return metrics, prediction_entries


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def plot_confusion_matrix(
    metrics: dict[str, Any],
    idx_to_class: dict[int, str],
    output_path: Path,
) -> None:
    labels = [idx_to_class[index] for index in sorted(idx_to_class)]
    matrix = confusion_matrix(metrics["y_true"], metrics["y_pred"], labels=list(sorted(idx_to_class)))

    figure_size = max(8, min(24, len(labels) * 0.5))
    fig, axis = plt.subplots(figsize=(figure_size, figure_size))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title("Pokemon MobileNetV3 Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(range(len(labels)))
    axis.set_yticks(range(len(labels)))
    axis.set_xticklabels(labels, rotation=90, fontsize=6)
    axis.set_yticklabels(labels, fontsize=6)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_training_history(history: dict[str, list[float]], output_path: Path) -> None:
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"], label="Val loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="Train accuracy")
    axes[1].plot(epochs, history["val_accuracy"], label="Val accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_sample_prediction_grid(
    predictions: list[dict[str, Any]],
    output_path: Path,
    max_examples: int = 24,
) -> list[dict[str, Any]]:
    selected_predictions = predictions[:max_examples]
    columns = 4
    rows = math.ceil(max(len(selected_predictions), 1) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4.2, rows * 4.2))
    axes_array = np.atleast_1d(axes).ravel()

    for axis in axes_array:
        axis.axis("off")

    for axis, prediction in zip(axes_array, selected_predictions):
        with Image.open(prediction["image_path"]) as image:
            axis.imshow(image.convert("RGB"))
        axis.axis("off")
        is_correct = prediction["true_label"] == prediction["predicted_label"]
        prefix = "OK" if is_correct else "WRONG"
        title_color = "#137333" if is_correct else "#c5221f"
        axis.set_title(
            (
                f"{prefix}\n"
                f"true: {prediction['true_label']}\n"
                f"pred: {prediction['predicted_label']} ({prediction['confidence']:.3f})"
            ),
            fontsize=8,
            color=title_color,
        )

    fig.suptitle("Sample Test Predictions", fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return selected_predictions


def create_pdf_report(
    output_path: Path,
    run_summary: dict[str, Any],
    dataset_summary: dict[str, Any],
    history: dict[str, Any],
    metrics: dict[str, Any],
    history_plot_path: Path,
    confusion_matrix_path: Path,
    sample_predictions_path: Path,
) -> None:
    with PdfPages(output_path) as pdf:
        fig, axis = plt.subplots(figsize=(11, 8.5))
        axis.axis("off")
        lines = [
            "ChampionsInsight MobileNetV3 Pokemon Classifier Report",
            "",
            f"Generated: {run_summary['generated_at']}",
            "",
            "Dataset Summary",
            f"Synthetic roots: {', '.join(run_summary['synthetic_roots']) or 'None'}",
            f"Real roots: {', '.join(run_summary['real_roots']) or 'None'}",
            f"Classes: {dataset_summary['num_classes']}",
            f"Total images: {dataset_summary['total_images']}",
            f"Train / val / test: {run_summary['train_count']} / {run_summary['val_count']} / {run_summary['test_count']}",
            f"Min / max images per class: {dataset_summary['min_images_per_class']} / {dataset_summary['max_images_per_class']}",
            "",
            "Model Summary",
            f"Variant: MobileNetV3-{run_summary['variant']}",
            f"Input size: {run_summary['image_size']}x{run_summary['image_size']}",
            f"Output classes: {dataset_summary['num_classes']}",
            "Optimizer: AdamW",
            f"Learning rate: {run_summary['learning_rate']}",
            f"Batch size: {run_summary['batch_size']}",
            f"Epochs trained: {run_summary['epochs_trained']}",
            f"Best validation accuracy: {run_summary['best_val_accuracy']:.4f}",
            "",
            "Test Metrics",
            f"Accuracy: {metrics['test_accuracy']:.4f}",
            f"Top-3 accuracy: {metrics['top3_accuracy']:.4f}",
            f"Top-5 accuracy: {metrics['top5_accuracy']:.4f}",
            f"Macro precision: {metrics['macro_precision']:.4f}",
            f"Macro recall: {metrics['macro_recall']:.4f}",
            f"Macro F1: {metrics['macro_f1']:.4f}",
            "",
            "Next Steps",
            "Add more real phone-camera crops for classes with high confusion.",
            "Keep synthetic data but validate using real cropped images.",
            "Use confidence thresholding in production.",
            "Use FAISS/type detection as fallback when MobileNetV3 confidence is low.",
        ]
        axis.text(0.03, 0.98, "\n".join(lines), va="top", ha="left", fontsize=11)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        for image_path, title in [
            (history_plot_path, "Training History"),
            (confusion_matrix_path, "Confusion Matrix"),
            (sample_predictions_path, "Sample Test Predictions"),
        ]:
            fig, axis = plt.subplots(figsize=(11, 8.5))
            axis.axis("off")
            axis.set_title(title, fontsize=16)
            axis.imshow(plt.imread(image_path))
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def save_checkpoint(
    output_path: Path,
    model: torch.nn.Module,
    class_to_idx: dict[str, int],
    run_summary: dict[str, Any],
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_to_idx": class_to_idx,
        "variant": run_summary["variant"],
        "image_size": run_summary["image_size"],
        "generated_at": run_summary["generated_at"],
        "training_summary": run_summary,
    }
    torch.save(checkpoint, output_path)


def initialize_from_checkpoint(model, checkpoint_path, class_to_idx, variant):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("variant", "small") != variant:
        raise ValueError("Initial checkpoint variant does not match the requested model")
    old_mapping = checkpoint["class_to_idx"]
    old_state = checkpoint["model_state_dict"]
    state = model.state_dict()
    head_keys = {"classifier.3.weight", "classifier.3.bias"}
    for key, tensor in old_state.items():
        if key not in head_keys:
            if key not in state or state[key].shape != tensor.shape:
                raise ValueError(f"Incompatible initial checkpoint tensor: {key}")
            state[key] = tensor
    copied = 0
    for label, index in class_to_idx.items():
        if label not in old_mapping:
            continue
        old_index = old_mapping[label]
        for key in head_keys:
            state[key][index].copy_(old_state[key][old_index])
        copied += 1
    model.load_state_dict(state)
    return copied


def main() -> None:
    args = parse_args()
    if args.evaluate_only:
        if args.init_checkpoint is None:
            raise ValueError("--evaluate-only requires --init-checkpoint")
        args.epochs = 0
    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    synthetic_roots = resolve_dataset_roots(args.synthetic_root, [DEFAULT_SYNTHETIC_ROOT])
    real_roots = resolve_dataset_roots(args.real_root, discover_default_real_roots())
    dataset_roots = synthetic_roots + [root for root in real_roots if root not in synthetic_roots]

    if not dataset_roots:
        raise FileNotFoundError("No dataset roots were found. Pass --synthetic-root or --real-root to valid folders.")

    print("[train] Collecting labeled Pokemon crops...")
    synthetic_samples, synthetic_summary = collect_image_samples(
        dataset_roots=synthetic_roots,
        min_images_per_class=args.min_images_per_class,
        source_kind="synthetic",
    )
    real_samples, real_summary = collect_image_samples(
        dataset_roots=real_roots,
        min_images_per_class=1,
        source_kind="real",
    )

    if args.target_images_per_class:
        samples, sampling_summary = balance_samples_with_real_priority(
            real_samples=real_samples,
            synthetic_samples=synthetic_samples,
            target_images_per_class=args.target_images_per_class,
            seed=args.seed,
        )
    else:
        samples = synthetic_samples + real_samples
        samples.sort(key=lambda sample: (sample.label, str(sample.image_path)))
        sampling_summary = {
            "strategy": "all_samples",
            "target_images_per_class": None,
            "total_selected_images": len(samples),
            "total_real_selected": len(real_samples),
            "total_synthetic_selected": len(synthetic_samples),
        }

    samples, skipped_selected_classes = filter_samples_by_min_count(
        samples,
        min_images_per_class=args.min_images_per_class,
    )
    combined_class_counts = Counter(sample.label for sample in samples)
    dataset_summary = {
        "roots": [str(path) for path in dataset_roots],
        "synthetic_summary": synthetic_summary,
        "real_summary": real_summary,
        "sampling_summary": sampling_summary,
        "class_counts": dict(sorted(combined_class_counts.items())),
        "skipped_classes": {
            "synthetic_collection": synthetic_summary.get("skipped_classes", {}),
            "post_sampling": skipped_selected_classes,
        },
        "total_images": len(samples),
        "num_classes": len(combined_class_counts),
        "min_images_per_class": min(combined_class_counts.values()) if combined_class_counts else 0,
        "max_images_per_class": max(combined_class_counts.values()) if combined_class_counts else 0,
    }
    del synthetic_samples, real_samples
    gc.collect()

    if not samples:
        raise ValueError("No labeled images were found after filtering. Check the dataset roots and class folders.")

    print(
        f"[train] Found {dataset_summary['total_images']} images across "
        f"{dataset_summary['num_classes']} classes."
    )
    if args.target_images_per_class:
        print(
            f"[train] Sampling strategy: all real images plus synthetic fill up to "
            f"{args.target_images_per_class} per class."
        )
    if synthetic_summary["skipped_classes"]:
        print(
            f"[train] Skipped {len(synthetic_summary['skipped_classes'])} synthetic classes below "
            f"the min-images-per-class threshold."
        )
    if skipped_selected_classes:
        preview = ", ".join(
            f"{label} ({count})"
            for label, count in list(skipped_selected_classes.items())[:10]
        )
        print(
            f"[train] Dropped {len(skipped_selected_classes)} classes after sampling because they still "
            f"had fewer than {args.min_images_per_class} images. Example: {preview}"
        )

    class_to_idx, idx_to_class = create_class_mappings(samples)
    train_samples, val_samples, test_samples = split_samples(samples, seed=args.seed)

    print(
        f"[train] Split sizes -> train: {len(train_samples)}, "
        f"val: {len(val_samples)}, test: {len(test_samples)}"
    )

    output_dir = resolve_user_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = make_data_loader(
        train_samples,
        class_to_idx,
        build_train_transform(),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
    eval_transform = get_eval_transform()
    val_loader = make_data_loader(
        val_samples,
        class_to_idx,
        eval_transform,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )
    test_loader = make_data_loader(
        test_samples,
        class_to_idx,
        eval_transform,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
    )

    device = resolve_training_device(args.device)
    use_amp = device.type == "cuda" and not args.disable_amp
    if device.type == "cuda":
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(device_index)
        print(f"[train] Using GPU: cuda:{device_index} ({gpu_name})")
        print(f"[train] AMP enabled: {use_amp}")
    else:
        print(f"[train] Using device: {device}")
    print(f"[train] Data loader workers: {args.num_workers}")
    if args.num_workers > 0:
        print(f"[train] Data loader prefetch factor: {args.prefetch_factor}")

    model = build_mobilenetv3_model(
        num_classes=len(class_to_idx),
        variant=args.variant,
        use_pretrained=args.init_checkpoint is None,
    )
    if args.init_checkpoint is not None:
        retained = initialize_from_checkpoint(model, resolve_user_path(args.init_checkpoint), class_to_idx, args.variant)
        print(f"[train] Initialized from Pokemon checkpoint: {retained} retained classes, {len(class_to_idx) - retained} new classes.")
    model = model.to(device)
    if device.type == "cuda" and args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_state_dict = None
    epochs_without_improvement = 0
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "learning_rate": [],
    }
    if args.evaluate_only:
        history_path = output_dir / "training_history.json"
        if history_path.exists():
            history = json.loads(history_path.read_text(encoding="utf-8"))
        best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if history["val_accuracy"]:
            best_index = max(range(len(history["val_accuracy"])),
                             key=lambda index: (history["val_accuracy"][index], -history["val_loss"][index]))
            best_val_accuracy = history["val_accuracy"][best_index]
            best_val_loss = history["val_loss"][best_index]

    action = "Evaluating" if args.evaluate_only else "Training"
    print(f"[train] {action} MobileNetV3-{args.variant} on {device}...")
    for epoch_index in range(args.epochs):
        current_learning_rate = float(optimizer.param_groups[0]["lr"])
        epoch_label = f"epoch {epoch_index + 1:02d}/{args.epochs:02d}"
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            log_interval=args.log_interval,
            epoch_label=epoch_label,
            channels_last=args.channels_last,
        )
        val_loss, val_accuracy = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
            use_amp=use_amp,
            log_interval=max(1, args.log_interval // 2),
            epoch_label=epoch_label,
            channels_last=args.channels_last,
        )
        scheduler.step(val_loss)

        history["train_loss"].append(round(train_loss, 6))
        history["train_accuracy"].append(round(train_accuracy, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_accuracy"].append(round(val_accuracy, 6))
        history["learning_rate"].append(round(current_learning_rate, 10))
        save_json(output_dir / "training_history.json", history)

        print(
            f"[epoch {epoch_index + 1:02d}/{args.epochs:02d}] "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        improved = False
        if val_accuracy > best_val_accuracy:
            improved = True
        elif math.isclose(val_accuracy, best_val_accuracy, rel_tol=0.0, abs_tol=1e-9) and val_loss < best_val_loss:
            improved = True

        if improved:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            best_state_dict = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
            save_checkpoint(output_dir / "best_checkpoint.pt", model, class_to_idx, {
                "variant": args.variant, "image_size": DEFAULT_IMAGE_SIZE,
                "generated_at": datetime.now().astimezone().isoformat(),
                "epochs_trained": epoch_index + 1,
                "best_val_accuracy": best_val_accuracy, "best_val_loss": best_val_loss,
                "initial_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
            })
            save_json(output_dir / "class_to_idx.json", class_to_idx)
            save_json(output_dir / "idx_to_class.json", {str(index): label for index, label in idx_to_class.items()})
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"[train] Early stopping triggered after {epoch_index + 1} epochs.")
            break

    if best_state_dict is None:
        raise RuntimeError("Training finished without producing a best checkpoint.")

    model.load_state_dict(best_state_dict)

    metrics, predictions = evaluate_model(model, test_loader, idx_to_class, device)
    print(
        "[test] "
        f"accuracy={metrics['test_accuracy']:.4f} "
        f"top3={metrics['top3_accuracy']:.4f} "
        f"top5={metrics['top5_accuracy']:.4f} "
        f"macro_precision={metrics['macro_precision']:.4f} "
        f"macro_recall={metrics['macro_recall']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )

    print("[test] Sample predictions:")
    for prediction in predictions[: min(8, len(predictions))]:
        print(
            f"  {prediction['true_label']} -> {prediction['predicted_label']} "
            f"({prediction['confidence']:.3f}) :: {prediction['image_path']}"
        )

    incorrect_predictions = [
        prediction for prediction in predictions
        if prediction["true_label"] != prediction["predicted_label"]
    ]

    generated_at = datetime.now().astimezone().isoformat()
    run_summary = {
        "generated_at": generated_at,
        "synthetic_roots": [str(path) for path in synthetic_roots],
        "real_roots": [str(path) for path in real_roots],
        "train_count": len(train_samples),
        "val_count": len(val_samples),
        "test_count": len(test_samples),
        "variant": args.variant,
        "image_size": DEFAULT_IMAGE_SIZE,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs_trained": len(history["train_loss"]),
        "best_val_accuracy": round(best_val_accuracy, 6),
        "best_val_loss": round(best_val_loss, 6),
        "device": str(device),
        "min_images_per_class": args.min_images_per_class,
        "initial_checkpoint": str(args.init_checkpoint) if args.init_checkpoint else None,
        "channels_last": args.channels_last,
        "amp_enabled": use_amp,
        "evaluation_only": args.evaluate_only,
    }

    model_path = output_dir / "pokemon_mobilenetv3.pt"
    class_to_idx_path = output_dir / "class_to_idx.json"
    idx_to_class_path = output_dir / "idx_to_class.json"
    history_path = output_dir / "training_history.json"
    metrics_path = output_dir / "evaluation_metrics.json"
    dataset_summary_path = output_dir / "dataset_summary.json"
    classification_report_path = output_dir / "classification_report.json"
    confusion_matrix_path = output_dir / "confusion_matrix.png"
    sample_predictions_json_path = output_dir / "sample_predictions.json"
    sample_predictions_png_path = output_dir / "sample_predictions.png"
    incorrect_predictions_path = output_dir / "incorrect_predictions.json"
    history_plot_path = output_dir / "training_history.png"
    report_path = output_dir / "pokemon_mobilenetv3_report.pdf"

    plot_training_history(history, history_plot_path)
    plot_confusion_matrix(metrics, idx_to_class, confusion_matrix_path)
    sample_predictions = build_sample_prediction_grid(predictions, sample_predictions_png_path)

    save_checkpoint(model_path, model, class_to_idx, run_summary)
    save_json(class_to_idx_path, class_to_idx)
    save_json(idx_to_class_path, {str(index): label for index, label in idx_to_class.items()})
    save_json(history_path, history)
    save_json(
        metrics_path,
        {
            key: value
            for key, value in metrics.items()
            if key not in {"classification_report", "y_true", "y_pred"}
        },
    )
    save_json(dataset_summary_path, {**dataset_summary, **run_summary})
    save_json(classification_report_path, metrics["classification_report"])
    save_json(sample_predictions_json_path, sample_predictions)
    save_json(incorrect_predictions_path, incorrect_predictions)

    create_pdf_report(
        report_path,
        run_summary=run_summary,
        dataset_summary=dataset_summary,
        history=history,
        metrics=metrics,
        history_plot_path=history_plot_path,
        confusion_matrix_path=confusion_matrix_path,
        sample_predictions_path=sample_predictions_png_path,
    )

    print(f"[save] Model checkpoint: {model_path}")
    print(f"[save] PDF report: {report_path}")


if __name__ == "__main__":
    main()
