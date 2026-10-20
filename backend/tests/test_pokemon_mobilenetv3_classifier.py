"""Tests for the supervised MobileNetV3 Pokemon crop classifier helpers."""

from pathlib import Path

from PIL import Image
import torch

from scripts.train_mobilenetv3_pokemon_classifier import (
    PokemonImageDataset,
    balance_samples_with_real_priority,
    collect_image_samples,
    create_class_mappings,
    ImageSample,
    initialize_from_checkpoint,
    evaluate_model,
)
from services.pokemon_mobilenetv3_classifier_service import (
    clear_model_cache,
    get_eval_transform,
    predict_pokemon_crop,
)


def _write_test_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (32, 32), color=color)
    image.save(path)


def test_expanding_classifier_preserves_reordered_existing_labels(upload_dir):
    def model_for_classes(count):
        model = torch.nn.Module()
        model.backbone = torch.nn.Linear(2, 2)
        model.classifier = torch.nn.Sequential(
            torch.nn.Identity(), torch.nn.Identity(), torch.nn.Identity(), torch.nn.Linear(2, count)
        )
        return model

    original = model_for_classes(2)
    expanded = model_for_classes(3)
    new_row = expanded.classifier[3].weight[0].detach().clone()
    path = upload_dir / "original.pt"
    torch.save({"model_state_dict": original.state_dict(), "class_to_idx": {"old_a": 0, "old_b": 1},
                "variant": "small"}, path)
    copied = initialize_from_checkpoint(expanded, path, {"new": 0, "old_b": 1, "old_a": 2}, "small")
    assert copied == 2
    assert torch.equal(expanded.backbone.weight, original.backbone.weight)
    assert torch.equal(expanded.classifier[3].weight[2], original.classifier[3].weight[0])
    assert torch.equal(expanded.classifier[3].bias[2], original.classifier[3].bias[0])
    assert torch.equal(expanded.classifier[3].weight[0], new_row)


def test_evaluation_keeps_predictions_aligned_with_paths_and_labels():
    loader = [(torch.tensor([[8.0, 0.0], [0.0, 9.0]]), torch.tensor([0, 1]), ["old.jpg", "new.jpg"])]
    metrics, predictions = evaluate_model(torch.nn.Identity(), loader, {0: "old", 1: "new"}, torch.device("cpu"))
    assert metrics["test_accuracy"] == 1.0
    assert [(entry["image_path"], entry["true_label"], entry["predicted_label"])
            for entry in predictions] == [("old.jpg", "old", "old"), ("new.jpg", "new", "new")]
    assert predictions[1]["top5"][0][0] == "new"


def test_collect_image_samples_finds_labeled_images(upload_dir):
    dataset_root = upload_dir / "slot_pokemon"
    _write_test_image(dataset_root / "garchomp" / "one.png", (255, 0, 0))
    _write_test_image(dataset_root / "garchomp" / "two.png", (220, 0, 0))
    _write_test_image(dataset_root / "sneasler" / "one.png", (0, 255, 0))
    _write_test_image(dataset_root / "sneasler" / "two.png", (0, 220, 0))
    _write_test_image(dataset_root / "sneasler" / "three.png", (0, 200, 0))
    (dataset_root / "notes.txt").write_text("ignore me", encoding="utf-8")

    samples, summary = collect_image_samples([dataset_root], min_images_per_class=2)

    assert len(samples) == 5
    assert summary["num_classes"] == 2
    assert summary["class_counts"] == {"garchomp": 2, "sneasler": 3}
    assert all(sample.image_path.suffix.lower() == ".png" for sample in samples)


def test_create_class_mappings_sorts_labels(upload_dir):
    dataset_root = upload_dir / "slot_pokemon"
    _write_test_image(dataset_root / "zoroark" / "one.png", (255, 255, 255))
    _write_test_image(dataset_root / "garchomp" / "one.png", (10, 10, 10))

    samples, _summary = collect_image_samples([dataset_root], min_images_per_class=1)
    class_to_idx, idx_to_class = create_class_mappings(samples)

    assert class_to_idx == {"garchomp": 0, "zoroark": 1}
    assert idx_to_class == {0: "garchomp", 1: "zoroark"}


def test_predict_pokemon_crop_returns_structured_error_when_model_missing(upload_dir):
    clear_model_cache()
    missing_model_dir = upload_dir / "missing_model"
    image_path = upload_dir / "crop.png"
    _write_test_image(image_path, (80, 120, 200))

    prediction = predict_pokemon_crop(image_path, model_dir=missing_model_dir)

    assert prediction["pokemon"] is None
    assert prediction["model"] == "mobilenetv3"
    assert prediction["top5"] == []
    assert "Train it with backend/scripts/train_mobilenetv3_pokemon_classifier.py first." in prediction["error"]


def test_dataset_skips_unreadable_images(upload_dir):
    dataset_root = upload_dir / "slot_pokemon"
    broken_path = dataset_root / "garchomp" / "broken.png"
    broken_path.parent.mkdir(parents=True, exist_ok=True)
    broken_path.write_bytes(b"not a real image")
    _write_test_image(dataset_root / "garchomp" / "valid.png", (255, 0, 0))

    samples, _summary = collect_image_samples([dataset_root], min_images_per_class=1)
    class_to_idx, _idx_to_class = create_class_mappings(samples)
    dataset = PokemonImageDataset(samples, class_to_idx, get_eval_transform())

    image_tensor, target, image_path = dataset[0]

    assert image_tensor.shape[0] == 3
    assert target == 0
    assert image_path.endswith("valid.png")


def test_balance_samples_with_real_priority_fills_with_synthetic(upload_dir):
    real_samples = [
        ImageSample(upload_dir / "real" / "garchomp" / "real_1.png", "garchomp", "slot_pokemon_real", "real"),
        ImageSample(upload_dir / "real" / "garchomp" / "real_2.png", "garchomp", "slot_pokemon_real", "real"),
    ]
    synthetic_samples = [
        ImageSample(upload_dir / "syn" / "garchomp" / "a_shiny.png", "garchomp", "slot_pokemon_synthetic", "synthetic"),
        ImageSample(upload_dir / "syn" / "garchomp" / "b_shiny.png", "garchomp", "slot_pokemon_synthetic", "synthetic"),
        ImageSample(upload_dir / "syn" / "garchomp" / "c.png", "garchomp", "slot_pokemon_synthetic", "synthetic"),
        ImageSample(upload_dir / "syn" / "garchomp" / "d.png", "garchomp", "slot_pokemon_synthetic", "synthetic"),
        ImageSample(upload_dir / "syn" / "garchomp" / "e.png", "garchomp", "slot_pokemon_synthetic", "synthetic"),
    ]

    selected_samples, summary = balance_samples_with_real_priority(
        real_samples=real_samples,
        synthetic_samples=synthetic_samples,
        target_images_per_class=4,
        seed=7,
    )

    assert len(selected_samples) == 4
    assert sum(1 for sample in selected_samples if sample.source_kind == "real") == 2
    assert sum(1 for sample in selected_samples if sample.source_kind == "synthetic") == 2
    assert summary["per_class"]["garchomp"]["total_selected"] == 4
