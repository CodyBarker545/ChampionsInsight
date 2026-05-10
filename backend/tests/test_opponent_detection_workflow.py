"""Tests opponent image detection workflow, stat enrichment, and sprite lookup."""

import json
from io import BytesIO
from pathlib import Path

from api import routes

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "opponent_expected_team.json"


def load_fixture():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


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


def test_opponent_detect_route_enriches_detected_team_with_level_50_stats(
    client,
    upload_dir,
    monkeypatch,
):
    fixture = load_fixture()
    image_path = upload_dir / fixture["imageFilename"]
    image_path.write_bytes(b"\xff\xd8\xff\xe0" + (b"0" * 4096))

    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(routes, "detect_opponent_team", lambda _path: fake_detected_team_from_fixture())

    response = client.post(
        "/api/opponent/detect",
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


def test_opponent_detect_route_stores_latest_prediction_for_refresh(
    client,
    upload_dir,
    monkeypatch,
):
    fixture = load_fixture()
    image_path = upload_dir / fixture["imageFilename"]
    image_path.write_bytes(b"\xff\xd8\xff\xe0" + (b"0" * 4096))

    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(routes, "detect_opponent_team", lambda _path: fake_detected_team_from_fixture())

    response = client.post(
        "/api/opponent/detect",
        json={"filename": fixture["imageFilename"]},
    )

    assert response.status_code == 200

    prediction_path = upload_dir / "latest_opponent_prediction.json"
    assert prediction_path.exists()

    saved_prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert saved_prediction["filename"] == fixture["imageFilename"]
    assert saved_prediction["detectedTeam"][0]["name"] == fixture["expectedTeam"][0]["name"]

    latest_response = client.get("/api/opponent/prediction/latest")

    assert latest_response.status_code == 200
    assert latest_response.json["hasPrediction"] is True
    assert latest_response.json["filename"] == fixture["imageFilename"]


def test_opponent_image_upload_clears_stale_latest_prediction(
    client,
    upload_dir,
    monkeypatch,
):
    prediction_path = upload_dir / "latest_opponent_prediction.json"
    prediction_path.write_text(json.dumps({"detectedTeam": [{"name": "Oldmon"}]}), encoding="utf-8")

    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(
        routes,
        "save_opponent_image",
        lambda _image, run_detection=True: {
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
            "image": (BytesIO(b"\xff\xd8\xff\xe0" + (b"0" * 2048)), "new-opponent.jpg"),
            "skipDetection": "1",
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert prediction_path.exists()
    saved_prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    assert saved_prediction["hasPrediction"] is False
    assert saved_prediction["detectedTeam"] == []


def test_opponent_detect_route_supports_name_field_from_cv_service(
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
        "detect_opponent_team",
        lambda _path: {
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
        "/api/opponent/detect",
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
