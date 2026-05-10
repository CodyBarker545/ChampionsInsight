"""Tests Flask API routes used by the frontend."""

from io import BytesIO

from api import routes
from services import image_service
from services import cv_service


VALID_JPEG = b"\xff\xd8\xff\xe0" + (b"0" * 2048)


# Tests the health check route.
def test_health_route(client):
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json == {"status": "ok", "app": "Champions Insight"}


# Tests the locally stored user team route.
def test_get_user_team_route(client):
    response = client.get("/api/user/team")

    assert response.status_code == 200
    assert response.json["userId"] == "local-demo-user"
    assert len(response.json["team"]) == 6
    assert response.json["team"][0]["moves"]


# Tests returning a clear error when the user team file is missing.
def test_get_user_team_route_handles_missing_file(client, monkeypatch):
    monkeypatch.setattr(routes, "load_user_team", lambda: (_ for _ in ()).throw(FileNotFoundError("User team file not found.")))

    response = client.get("/api/user/team")

    assert response.status_code == 500
    assert response.json["error"] == "User team file not found."


def test_put_user_team_route_saves_team(client, upload_dir, monkeypatch):
    saved_path = upload_dir / "user_team.json"
    monkeypatch.setattr("services.user_team_service.USER_TEAM_PATH", saved_path)
    monkeypatch.setattr(routes, "save_user_team", __import__("services.user_team_service", fromlist=["save_user_team"]).save_user_team)

    team = [{"id": f"user-{index}", "name": f"Pokemon {index}"} for index in range(1, 7)]
    response = client.put(
        "/api/user/team",
        json={"userId": "local-demo-user", "teamName": "My Team", "team": team},
    )

    assert response.status_code == 200
    assert response.json["teamName"] == "My Team"
    assert response.json["team"][0]["name"] == "Pokemon 1"
    assert saved_path.exists()


def test_put_user_team_route_requires_six_pokemon(client):
    response = client.put("/api/user/team", json={"team": []})

    assert response.status_code == 400
    assert response.json["error"] == "A saved team must include exactly six Pokemon."


def test_pokemon_route_returns_available_forms(client):
    response = client.get("/api/pokemon/Dragonite")

    assert response.status_code == 200
    assert {"name": "Dragonite Mega", "label": "Dragonite Mega", "isDefault": False} in response.json["formOptions"]


def test_pokemon_stats_route_uses_form_stats(client):
    response = client.get("/api/pokemon/Dragonite%20Mega/stats?nature=modest")

    assert response.status_code == 200
    assert response.json["name"] == "Dragonite Mega"
    assert response.json["finalStats"]["special_attack"] == 181


# Tests the team analysis route.
def test_analyze_team_route(client):
    response = client.post(
        "/api/team/analyze",
        json={"team": [{"name": "Aegis"}], "opponent": "Volt"},
    )

    assert response.status_code == 200
    assert response.json["summary"] == "Aegis has a favorable opening into Volt."


# Tests that the team analysis route validates empty teams.
def test_analyze_team_route_rejects_empty_team(client):
    response = client.post("/api/team/analyze", json={"team": []})

    assert response.status_code == 400
    assert response.json["error"] == "At least one team member is required."


def test_calculate_damage_route(client):
    response = client.post(
        "/api/damage/calculate",
        json={
            "attacker": {
                "name": "Charizard",
                "types": ["Fire", "Flying"],
                "stats": {
                    "hp": 153,
                    "attack": 93,
                    "defense": 98,
                    "specialAttack": 161,
                    "specialDefense": 105,
                    "speed": 167,
                },
                "ability": "Blaze",
            },
            "defender": {
                "name": "Venusaur",
                "types": ["Grass", "Poison"],
                "stats": {
                    "hp": 155,
                    "attack": 91,
                    "defense": 103,
                    "specialAttack": 120,
                    "specialDefense": 120,
                    "speed": 100,
                },
            },
            "move": {"name": "Flamethrower"},
        },
    )

    assert response.status_code == 200
    assert response.json["attacker"] == "Charizard"
    assert response.json["defender"] == "Venusaur"
    assert response.json["damage"]["category"] == "special"
    assert response.json["damage"]["attackStatUsed"] == 161
    assert response.json["damage"]["defenseStatUsed"] == 120
    assert response.json["damage"]["percentRange"].endswith("%")


# Tests the opponent image upload route.
def test_upload_opponent_image_route(client, upload_dir, monkeypatch):
    monkeypatch.setattr(image_service, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(
        image_service,
        "assess_opponent_image_quality",
        lambda _path: {"canAnalyze": True, "qualityLevel": "good", "issues": [], "warnings": [], "metrics": {}},
    )
    monkeypatch.setattr(
        image_service,
        "detect_opponent_team",
        lambda path: {
            "image": str(path),
            "referenceCount": 1,
            "detectedTeam": [],
        },
    )

    response = client.post(
        "/api/opponent/image",
        data={"image": (BytesIO(VALID_JPEG), "team.jpg")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.json["status"] == "received"
    assert (upload_dir / response.json["filename"]).exists()


# Tests that the opponent image upload route requires a file.
def test_upload_opponent_image_route_requires_file(client):
    response = client.post("/api/opponent/image", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert response.json["error"] == "Image file is required."


# Tests detecting an already uploaded opponent image.
def test_detect_uploaded_opponent_image_route(client, upload_dir, monkeypatch):
    image_path = upload_dir / "team.jpg"
    image_path.write_bytes(VALID_JPEG)
    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(
        routes,
        "detect_opponent_team",
        lambda path: {
            "image": str(path),
            "referenceCount": 1,
            "detectedTeam": [{"position": 1, "pokemonName": "Aerodactyl", "confidence": 0.88}],
        },
    )

    response = client.post("/api/opponent/detect", json={"filename": "team.jpg"})

    assert response.status_code == 200
    assert response.json["detectedTeam"][0]["pokemonName"] == "Aerodactyl"
    assert response.json["detectedTeam"][0]["baseStats"]["speed"] == 130


# Tests returning local Pokemon data for manual correction.
def test_get_pokemon_route(client):
    response = client.get("/api/pokemon/Charizard")

    assert response.status_code == 200
    assert response.json["name"] == "Charizard"
    assert response.json["baseStats"]["speed"] == 100
    assert response.json["image"].endswith("Menu_CP_0006.png")
    assert response.json["abilities"] == ["Blaze", "Drought", "Solar Power", "Tough Claws"]


def test_get_pokemon_route_uses_champions_lab_aliases(client):
    response = client.get("/api/pokemon/Alolan%20Ninetales")

    assert response.status_code == 200
    assert response.json["name"] == "Ninetales Alola"
    assert response.json["abilities"] == ["Snow Cloak", "Snow Warning"]


def test_search_pokemon_route(client):
    response = client.get("/api/pokemon/search?q=Aero")

    assert response.status_code == 200
    names = [pokemon["name"] for pokemon in response.json["results"]]
    assert "Aerodactyl" in names


def test_get_pokemon_top_tournament_moves_route(client):
    response = client.get("/api/pokemon/Aerodactyl/moves/top")

    assert response.status_code == 200
    assert response.json["matchedName"] == "Aerodactyl"
    assert len(response.json["moves"]) == 4
    assert response.json["moves"][0]["move"] == "Rock Slide"


# Tests that detection rejects unsafe filenames.
def test_detect_uploaded_opponent_image_route_rejects_unsafe_filename(client, upload_dir, monkeypatch):
    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_dir)

    response = client.post("/api/opponent/detect", json={"filename": "../team.jpg"})

    assert response.status_code == 400
    assert response.json["error"] == "Valid uploaded image filename is required."


# Tests that detection handles missing uploaded files.
def test_detect_uploaded_opponent_image_route_handles_missing_file(client, upload_dir, monkeypatch):
    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_dir)

    response = client.post("/api/opponent/detect", json={"filename": "missing.jpg"})

    assert response.status_code == 404
    assert response.json["error"] == "Uploaded image was not found."


# Tests returning OpenCV errors from the detection route.
def test_detect_uploaded_opponent_image_route_handles_cv_errors(client, upload_dir, monkeypatch):
    image_path = upload_dir / "team.jpg"
    image_path.write_bytes(VALID_JPEG)

    def raise_cv_error(_path):
        raise cv_service.ComputerVisionError("Could not read image.")

    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(routes, "detect_opponent_team", raise_cv_error)

    response = client.post("/api/opponent/detect", json={"filename": "team.jpg"})

    assert response.status_code == 400
    assert response.json["error"] == "Could not read image."


# Tests checking image quality for an uploaded opponent image.
def test_check_uploaded_opponent_image_quality_route(client, upload_dir, monkeypatch):
    image_path = upload_dir / "team.jpg"
    image_path.write_bytes(VALID_JPEG)
    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(
        routes,
        "assess_opponent_image_quality",
        lambda _path: {"canAnalyze": True, "qualityLevel": "good", "issues": [], "warnings": [], "metrics": {}},
    )

    response = client.post("/api/opponent/quality", json={"filename": "team.jpg"})

    assert response.status_code == 200
    assert response.json["qualityLevel"] == "good"


# Tests detecting only type icons for an uploaded opponent image.
def test_detect_uploaded_opponent_types_route(client, upload_dir, monkeypatch):
    image_path = upload_dir / "team.jpg"
    image_path.write_bytes(VALID_JPEG)
    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(
        routes,
        "detect_opponent_team_types",
        lambda _path: {
            "image": str(_path),
            "mode": "opponent-type-only",
            "teamTypes": [{"position": 1, "types": ["rock", "flying"]}],
        },
    )

    response = client.post("/api/opponent/types", json={"filename": "team.jpg"})

    assert response.status_code == 200
    assert response.json["teamTypes"][0]["types"] == ["rock", "flying"]


# Tests the RAG question route.
def test_ask_rag_route(client, monkeypatch):
    monkeypatch.setattr(
        routes,
        "answer_rag_question",
        lambda question: {
            "question": question,
            "answer": "Grounded answer.",
            "source": "Local RAG Knowledge Base",
        },
    )

    response = client.post("/api/rag/ask", json={"question": "What does this ability do?"})

    assert response.status_code == 200
    assert response.json["answer"] == "Grounded answer."


# Tests that the RAG question route validates blank questions.
def test_ask_rag_route_requires_question(client):
    response = client.post("/api/rag/ask", json={"question": ""})

    assert response.status_code == 400
    assert response.json["error"] == "Question is required."
