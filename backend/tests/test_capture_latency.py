"""Keep expensive or inactive work off the photo capture critical path."""
from io import BytesIO

from PIL import Image


def test_warmup_prepares_active_upload_models(monkeypatch):
    import app
    from services import cv_service, pokemon_mobilenetv3_classifier_service as classifier
    calls = []
    class Model:
        def predict(self, **kwargs):
            calls.append(('yolo', kwargs['source'].shape))
    monkeypatch.setattr(cv_service, 'load_yolo_slot_object_model', lambda: Model())
    monkeypatch.setattr(cv_service, 'get_yolo_inference_device', lambda: 'cpu')
    monkeypatch.setattr(classifier, 'predict_pokemon_crops', lambda images: calls.append(('mobilenet', images[0].shape)))
    app.warm_computer_vision_services()
    assert calls == [('yolo', (220, 640, 3)), ('mobilenet', (224, 224, 3))]


def test_background_upload_queues_without_duplicate_quality_work(client, monkeypatch):
    from api import routes, opponent_routes
    def unexpected(*args, **kwargs):
        raise AssertionError('Quality analysis belongs to the detection worker')
    monkeypatch.setattr(routes, 'assess_opponent_image_quality_from_image', unexpected)
    class Thread:
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
    monkeypatch.setattr(opponent_routes, 'Thread', Thread)
    data = BytesIO()
    Image.effect_noise((256, 256), 80).convert('RGB').save(data, format='JPEG')
    data.seek(0)
    response = client.post('/api/opponent/image', data={'backgroundDetection': '1', 'image': (data, 'phone.jpg')})
    assert response.status_code == 200
    assert response.json['status'] == 'queued'
