"""Regression checks for uncertain sprite crops and recovery provenance."""
import numpy as np
import pytest
from services import opponent_detection_service as detection


@pytest.mark.parametrize('original_confidence, retry_confidence, accepted', [
    (0.28, 0.99, True),  # clipped tail recovers
    (0.57, 0.94, True),  # clipped flower recovers
    (0.69, 0.68, False),  # Maushold must keep its original crop
    (0.20, 0.80, False),  # still uncertain
    (0.80, 0.86, False),  # improvement too small
])
def test_crop_retry_only_replaces_strong_improvements(monkeypatch, original_confidence, retry_confidence, accepted):
    card = np.zeros((100, 300, 3), dtype=np.uint8)
    original_crop = card[:, 90:120]
    record = {'card_crop': {'image': card}, 'pokemon_crop': original_crop}
    original = {'pokemon': 'original', 'confidence': original_confidence}
    predictions = [original]
    def retry(crops, top_k):
        assert crops[0].shape == (100, 129, 3)
        return [{'pokemon': 'recovered', 'confidence': retry_confidence}]
    monkeypatch.setattr(detection, 'predict_pokemon_crops', retry)
    detection.recover_uncertain_pokemon_crops([record], predictions)
    assert record['crop_recovery']['accepted'] is accepted
    assert predictions[0]['pokemon'] == ('recovered' if accepted else 'original')
    assert (record['pokemon_crop'] is original_crop) is not accepted
    assert record['crop_recovery']['originalPrediction'] is original


def test_confident_predictions_and_model_errors_are_not_retried(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail('Confident predictions and model errors must not trigger retry')
    monkeypatch.setattr(detection, 'predict_pokemon_crops', unexpected)
    predictions = [{'pokemon': 'raichu', 'confidence': 0.96}, {'error': 'missing checkpoint', 'confidence': 0}]
    detection.recover_uncertain_pokemon_crops([{}, {}], predictions)


@pytest.mark.parametrize('confidence, error, expected_name', [
    (0.28, None, 'unknown'),
    (0.84999, None, 'unknown'),  # acceptance uses unrounded confidence
    (0.85, None, 'toxtricity_low_key'),
    (0.99, None, 'toxtricity_low_key'),
    (0.99, 'inference failed', 'unknown'),
])
def test_unaccepted_mobilenet_candidates_do_not_become_roster_identity(
    monkeypatch, confidence, error, expected_name,
):
    card = np.zeros((100, 300, 3), dtype=np.uint8)
    prediction = {
        'pokemon': 'toxtricity_low_key', 'confidence': confidence,
        'top5': [{'name': 'toxtricity_low_key', 'confidence': confidence}],
    }
    if error:
        prediction['error'] = error
    monkeypatch.setattr(detection.cv_service, 'assess_opponent_image_quality_from_image', lambda *args, **kwargs: {})
    monkeypatch.setattr(detection.cv_service, 'crop_opponent_team_slots_from_image',
                        lambda *args, **kwargs: [{'position': 1, 'image': card}])
    monkeypatch.setattr(detection.slot_object_detection_service, 'detect_slot_objects',
                        lambda _: {'pokemon_sprite': {'image': card}})
    monkeypatch.setattr(detection, 'predict_pokemon_crops', lambda *args, **kwargs: [prediction])

    result = detection.detect_opponent_team_mobilenetv3_from_image(card)
    slot = result['detectedTeam'][0]

    assert slot['name'] == slot['pokemonName'] == expected_name
    assert slot['needsReview'] is (expected_name == 'unknown')
    assert slot['matchReason'] == ('needs_review' if expected_name == 'unknown' else 'accept_mobilenetv3')
    assert slot['pokemonTopCandidates'][0]['label'] == 'toxtricity_low_key'
    assert slot['mobileNetV3Prediction'] == prediction
