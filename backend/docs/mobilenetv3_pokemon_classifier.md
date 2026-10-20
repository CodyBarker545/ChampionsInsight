# MobileNetV3 Pokemon Classifier

This classifier predicts the Pokemon name from an already-cropped Pokemon image. It is meant to complement the current embedding and FAISS pipeline, not replace it yet.

## What It Does

- Trains a supervised `MobileNetV3` image classifier on labeled Pokemon crop folders.
- Supports mixed datasets from synthetic images and real/debug phone-camera crops.
- Evaluates on a held-out test split and saves:
  - model checkpoint
  - class mappings
  - training history
  - evaluation metrics
  - confusion matrix
  - incorrect predictions
  - sample test prediction grid
  - PDF report
- Exposes a lazy-loaded inference service for backend use.

## Expected Dataset Structure

The trainer treats the immediate parent folder as the class label.

```text
backend/data/training_dataset/slot_pokemon_synthetic/
  garchomp/
    img_001.png
    img_002.png
  sneasler/
    img_001.png

backend/data/training_dataset/slot_pokemon_real/
  garchomp/
    crop_001.jpg
  sneasler/
    crop_002.jpg
```

If an image sits directly under a root without a Pokemon folder, the trainer skips it.

## Training Command

```powershell
python backend/scripts/train_mobilenetv3_pokemon_classifier.py `
  --synthetic-root backend/data/training_dataset/slot_pokemon_synthetic `
  --real-root backend/data/training_dataset/slot_pokemon_real `
  --output-dir backend/data/cv/models/pokemon_mobilenetv3 `
  --epochs 20 `
  --batch-size 32 `
  --learning-rate 0.0003
```

The script also works with only the synthetic root if no real crop dataset is available yet.

## Output Files

The default output folder is `backend/data/cv/models/pokemon_mobilenetv3/`.

- `pokemon_mobilenetv3.pt`: saved PyTorch checkpoint
- `class_to_idx.json`: label to class index mapping
- `idx_to_class.json`: class index to label mapping
- `training_history.json`: epoch-by-epoch loss and accuracy
- `training_history.png`: visual history chart used in the report
- `evaluation_metrics.json`: summary test metrics
- `classification_report.json`: per-class precision, recall, and F1
- `dataset_summary.json`: dataset roots, split counts, and class count summary
- `confusion_matrix.png`: saved confusion matrix
- `sample_predictions.json`: sample test predictions included in the grid
- `sample_predictions.png`: grid of sample test crops with true and predicted labels
- `incorrect_predictions.json`: all incorrect test predictions
- `pokemon_mobilenetv3_report.pdf`: final training and evaluation report

## Reading The PDF Report

The PDF includes:

- dataset roots and dataset size summary
- split counts for train, validation, and test
- model configuration and best validation accuracy
- test accuracy, top-3 accuracy, top-5 accuracy, macro precision, macro recall, and macro F1
- training history chart
- confusion matrix
- sample test prediction grid
- next-step recommendations for collecting more real data and using confidence thresholds

## Backend Inference

Use [pokemon_mobilenetv3_classifier_service.py](/D:/SoftwareEng/ChampionsInsight/backend/services/pokemon_mobilenetv3_classifier_service.py) through:

```python
from services.pokemon_mobilenetv3_classifier_service import predict_pokemon_crop

result = predict_pokemon_crop("path/to/crop.png", top_k=5)
```

The service:

- loads the trained model lazily
- preprocesses crops with the same evaluation transform used during training
- accepts a file path, PIL image, or OpenCV/numpy image
- returns a structured error if the model checkpoint is missing

## Suggested Opponent Workflow

Keep the current embedding and FAISS path as the primary safety net while MobileNetV3 is being validated on real crops.

Suggested decision rule:

```python
if mobilenet_prediction["confidence"] >= 0.85:
    accept_mobilenetv3()
elif mobilenet_prediction["confidence"] >= 0.60:
    compare_with_faiss_and_type_prediction()
else:
    mark_as_needs_review()
```

This lets the supervised classifier help when it is confident, while still using FAISS and type detection as fallback when the crop is ambiguous or out-of-distribution.
