"""Tests opponent image detection workflow, stat enrichment, and sprite lookup."""

import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from api import routes

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "opponent_expected_team.json"


def load_fixture():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def test_partial_detection_preserves_unresolved_positions_through_flask(client, upload_dir, monkeypatch):
    image_path = upload_dir / 'partial.jpg'
    image_path.write_bytes(b'\xff\xd8\xff\xe0' + b'0' * 4096)
    monkeypatch.setattr(routes, 'UPLOAD_DIR', upload_dir)
    monkeypatch.setattr(routes, 'detect_opponent_team_mobilenetv3', lambda *args, **kwargs: {
        'detectedTeam': [{'position': 2, 'name': 'Pikachu', 'pokemonName': 'Pikachu', 'confidence': .95}],
        'unresolvedPositions': [1, 3, 4, 5, 6], 'positionsAreRelative': True,
        'cardDetection': {'validatedCount': 1, 'retried': True},
    })
    response = client.post('/api/opponent/detect-mobilenetv3', json={'filename': image_path.name})
    assert response.status_code == 200
    assert len(response.json['detectedTeam']) == 1
    assert response.json['detectedTeam'][0]['position'] == 2
    assert response.json['unresolvedPositions'] == [1, 3, 4, 5, 6]
    latest = client.get('/api/opponent/prediction/latest').json
    assert latest['unresolvedPositions'] == [1, 3, 4, 5, 6]
    assert latest['positionsAreRelative'] is True


def fake_detected_team_from_fixture():
    expected_team = load_fixture()["expectedTeam"]

    return {
        "image": load_fixture()["imageFilename"],
        "referenceCount": 6,
        "detectedTeam": [
            {
                "slot": pokemon["slot"],
                "pokemonName": pokemon["name"],
                "confidence": 0.95,
                "types": pokemon["types"],
                "image": f"fake-{pokemon['name']}.png"
            }
            for pokemon in expected_team
        ],
    }


def build_png_bytes(color=(120, 40, 220)):
    buffer = BytesIO()
    image = Image.new("RGB", (256, 256), color=color)
    image.putdata(
        [
            ((x * 13) % 256, (y * 17) % 256, ((x + y) * 19) % 256)
            for y in range(256)
            for x in range(256)
        ]
    )
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_opponent_mobilenetv3_detect_route_enriches_detected_team_with_level_50_stats(
    client,
    upload_dir,
    monkeypatch,
):
    fixture = load_fixture()
    image_path = upload_dir / fixture["imageFilename"]
    image_path.write_bytes(b"\xff\xd8\xff\xe0" + (b"0" * 4096))

    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(
        routes,
        "detect_opponent_team_mobilenetv3",
        lambda _path, **_kwargs: fake_detected_team_from_fixture(),
    )

    response = client.post(
        "/api/opponent/detect-mobilenetv3",
        json={"filename": fixture["imageFilename"]},
    )

    assert response.status_code == 200

    detected_team = response.json["detectedTeam"]
    expected_team = fixture["expectedTeam"]

    assert len(detected_team) == 6

    for actual, expected in zip(detected_team, expected_team):
        assert actual["name"] == expected["name"]
        assert actual["pokemonName"] == expected["name"]
        assert actual["types"] == expected["types"]
        assert actual["stats"] is not None
        assert actual["stats"]["level"] == 50
        assert actual["stats"]["finalStats"] == expected["neutralStats"]


def test_opponent_mobilenetv3_detect_route_stores_latest_prediction_for_refresh(
    client,
    upload_dir,
    monkeypatch,
):
    fixture = load_fixture()
    image_path = upload_dir / fixture["imageFilename"]
    image_path.write_bytes(b"\xff\xd8\xff\xe0" + (b"0" * 4096))

    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(
        routes,
        "detect_opponent_team_mobilenetv3",
        lambda _path, **_kwargs: fake_detected_team_from_fixture(),
    )

    response = client.post(
        "/api/opponent/detect-mobilenetv3",
        json={"filename": fixture["imageFilename"]},
    )

    assert response.status_code == 200

    latest_response = client.get("/api/opponent/prediction/latest")

    assert latest_response.status_code == 200
    assert latest_response.json["hasPrediction"] is True
    assert latest_response.json["filename"] == fixture["imageFilename"]
    assert latest_response.json["detectedTeam"][0]["name"] == fixture["expectedTeam"][0]["name"]
    assert not (upload_dir / "latest_opponent_prediction.json").exists()


def test_opponent_image_upload_does_not_write_latest_prediction_file(
    client,
    upload_dir,
    monkeypatch,
):
    prediction_path = upload_dir / "latest_opponent_prediction.json"

    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(
        routes,
        "save_opponent_image",
        lambda _image, run_detection=True, persist=True: {
            "status": "received",
            "filename": "new-opponent.jpg",
            "originalFilename": "new-opponent.jpg",
            "contentType": "image/jpeg",
            "sizeBytes": 2048,
            "message": "Image received.",
            "quality": {"canAnalyze": True},
            "detectedTeam": [],
            "detectionError": "",
        },
    )

    response = client.post(
        "/api/opponent/image",
        data={
            "image": (build_png_bytes(), "new-opponent.png"),
            "skipDetection": "1",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert not prediction_path.exists()
    latest_response = client.get("/api/opponent/prediction/latest")
    assert latest_response.status_code == 200
    assert latest_response.json["hasPrediction"] is False
    assert latest_response.json["detectedTeam"] == []


def test_opponent_mobilenetv3_detect_route_supports_name_field_from_cv_service(
    client,
    upload_dir,
    monkeypatch,
):
    fixture = load_fixture()
    image_path = upload_dir / fixture["imageFilename"]
    image_path.write_bytes(b"\xff\xd8\xff\xe0" + (b"0" * 4096))

    expected_team = fixture["expectedTeam"]

    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(
        routes,
        "detect_opponent_team_mobilenetv3",
        lambda _path, **_kwargs: {
            "image": fixture["imageFilename"],
            "referenceCount": 6,
            "detectedTeam": [
                {
                    "slot": pokemon["slot"],
                    "name": pokemon["name"],
                    "confidence": 0.95,
                    "types": pokemon["types"],
                }
                for pokemon in expected_team
            ],
        },
    )

    response = client.post(
        "/api/opponent/detect-mobilenetv3",
        json={"filename": fixture["imageFilename"]},
    )

    assert response.status_code == 200
    assert len(response.json["detectedTeam"]) == 6

    for actual, expected in zip(response.json["detectedTeam"], expected_team):
        assert actual["name"] == expected["name"]
        assert actual["stats"]["finalStats"] == expected["neutralStats"]


def test_pokemon_routes_return_images_for_expected_opponent_team(client):
    fixture = load_fixture()

    for pokemon in fixture["expectedTeam"]:
        response = client.get(f"/api/pokemon/{pokemon['name']}")

        assert response.status_code == 200
        assert response.json["name"]
        assert response.json["baseStats"]
        actual_types = [pokemon_type.lower() for pokemon_type in response.json["types"]]
        expected_types = [pokemon_type.lower() for pokemon_type in pokemon["types"]]

        assert actual_types == expected_types
        assert response.json["image"]
        assert response.json["image"].endswith(".png")


def test_mobilenet_predictions_include_roster_sprites_without_reference_images(client):
    from api.opponent_routes import enrich_detected_opponent_team
    names = ['raichu', 'espathra', 'archaludon', 'politoed', 'charizard', 'whimsicott']
    with client.application.app_context():
        team = enrich_detected_opponent_team([
            {'name': name, 'referenceImage': '', 'position': i + 1}
            for i, name in enumerate(names)
        ])
    for slot in team:
        assert slot['image'].startswith('/api/pokemon/sprite/normal/')
        assert slot['abilities']
        assert slot['pokemon']['types']
        assert client.get(slot['image']).status_code == 200


def test_pokemon_stats_route_returns_expected_stats_for_each_detected_pokemon(client):
    fixture = load_fixture()

    for pokemon in fixture["expectedTeam"]:
        response = client.get(f"/api/pokemon/{pokemon['name']}/stats?nature=hardy")

        assert response.status_code == 200
        assert response.json["level"] == 50
        assert response.json["finalStats"] == pokemon["neutralStats"]


def test_pokemon_stats_route_returns_nature_boosted_stats(client):
    fixture = load_fixture()

    for pokemon in fixture["expectedTeam"]:
        nature = pokemon["natureTest"]["nature"]
        expected_stats = pokemon["natureTest"]["expectedStats"]

        response = client.get(f"/api/pokemon/{pokemon['name']}/stats?nature={nature}")

        assert response.status_code == 200
        assert response.json["nature"] == nature
        assert response.json["finalStats"] == expected_stats
