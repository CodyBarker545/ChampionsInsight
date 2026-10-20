"""Tests Flask API routes used by the frontend."""

from io import BytesIO

from PIL import Image

from api import routes
from services import image_service
from services import cv_service


VALID_JPEG = b"\xff\xd8\xff\xe0" + (b"0" * 2048)


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
    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(image_service, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(
        routes,
        "assess_opponent_image_quality_from_image",
        lambda _image: {"canAnalyze": True, "qualityLevel": "good", "issues": [], "warnings": [], "metrics": {}},
    )
    monkeypatch.setattr(
        routes,
        "detect_opponent_team_mobilenetv3_from_image",
        lambda image, **_kwargs: {
            "image": "in-memory",
            "predictionSource": "mobilenetv3_yolo",
            "detectedTeam": [],
        },
    )

    response = client.post(
        "/api/opponent/image",
        data={"image": (build_png_bytes(), "team.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.json["predictionSource"] == "mobilenetv3_yolo"
    assert not (upload_dir / response.json["filename"]).exists()


# Tests that the opponent image upload route requires a file.
def test_upload_opponent_image_route_requires_file(client):
    response = client.post("/api/opponent/image", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert response.json["error"] == "Image file is required."


# Tests that the old embedding/FAISS route is disabled while MobileNetV3 is active.
def test_detect_uploaded_opponent_image_route(client, upload_dir, monkeypatch):
    image_path = upload_dir / "team.jpg"
    image_path.write_bytes(VALID_JPEG)
    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_dir)

    response = client.post("/api/opponent/detect", json={"filename": "team.jpg"})

    assert response.status_code == 410
    assert response.json["activeRoute"] == "/api/opponent/detect-mobilenetv3"


def test_detect_uploaded_opponent_image_mobilenetv3_route(client, upload_dir, monkeypatch):
    image_path = upload_dir / "team.jpg"
    image_path.write_bytes(VALID_JPEG)
    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(
        routes,
        "detect_opponent_team_mobilenetv3",
        lambda path, **_kwargs: {
            "image": str(path),
            "predictionSource": "mobilenetv3_yolo",
            "detectedTeam": [
                {
                    "position": 1,
                    "pokemonName": "Garchomp",
                    "name": "Garchomp",
                    "confidence": 0.94,
                    "predictionSource": "mobilenetv3_pokemon",
                    "mobileNetV3Prediction": {
                        "pokemon": "Garchomp",
                        "confidence": 0.94,
                        "model": "mobilenetv3",
                    },
                }
            ],
        },
    )

    response = client.post("/api/opponent/detect-mobilenetv3", json={"filename": "team.jpg"})

    assert response.status_code == 200
    assert response.json["predictionSource"] == "mobilenetv3_yolo"
    assert response.json["detectedTeam"][0]["pokemonName"] == "Garchomp"
    assert response.json["detectedTeam"][0]["predictionSource"] == "mobilenetv3_pokemon"
    assert response.json["detectedTeam"][0]["baseStats"]["speed"] == 102


# Tests returning local Pokemon data for manual correction.
def test_get_pokemon_route(client):
    response = client.get("/api/pokemon/Charizard")

    assert response.status_code == 200
    assert response.json["name"] == "Charizard"
    assert response.json["baseStats"]["speed"] == 100
    assert response.json["image"].endswith("Menu_CP_0006.png")
    assert response.json["abilities"] == ["Blaze", "Drought", "Solar Power", "Tough Claws"]


def test_get_pokedex_recommendations_route(client, monkeypatch):
    monkeypatch.setattr(
        routes,
        "get_pokemon_recommended_teams_by_name",
        lambda name: {
            "pokemon": name,
            "recommendedTeams": [
                {
                    "label": "Most Proven Tournament Shell",
                    "pokemon": [name, "Pelipper", "Archaludon"],
                    "sampleSize": 2,
                    "winRate": 0.667,
                    "recommendedSet": {
                        "item": "Venusaurite",
                        "ability": "Overgrow",
                        "nature": "Modest",
                        "moves": ["Leaf Storm", "Protect"],
                    },
                    "summary": "Built around common partners.",
                    "signals": [],
                }
            ],
            "dataQualityNotes": [],
        },
    )

    response = client.get("/api/pokedex/recommendations/Venusaur")

    assert response.status_code == 200
    assert response.json["recommendedTeams"][0]["label"] == "Most Proven Tournament Shell"
    assert response.json["recommendedTeams"][0]["recommendedSet"]["item"] == "Venusaurite"


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


def test_get_tournament_archetype_rules_route(client):
    response = client.get("/api/tournament/archetypes/rules")

    assert response.status_code == 200
    assert "totalTeamsAnalyzed" in response.json
    assert "archetypeSummaries" in response.json
    assert isinstance(response.json["archetypeSummaries"], list)


def test_get_tournament_archetype_clusters_route(client):
    response = client.get("/api/tournament/archetypes/clusters")

    assert response.status_code == 200
    assert "totalTeamsAnalyzed" in response.json
    assert "totalClusters" in response.json
    assert "clustersByParentArchetype" in response.json
    assert isinstance(response.json["clustersByParentArchetype"], dict)


def test_post_tournament_archetype_classify_route(client):
    response = client.post(
        "/api/tournament/archetypes/classify",
        json={
            "pokemon": [
                "Tyranitar",
                "Garchomp",
                "Sneasler",
                "Meganium",
                "Aerodactyl",
                "Basculegion",
            ],
            "moves": [],
            "items": [],
            "abilities": [],
        },
    )

    assert response.status_code == 200
    assert response.json["primaryArchetype"]
    assert response.json["displayLabel"]
    assert response.json["shortExplanation"]
    assert response.json["confidenceLabel"] in {"High", "Medium", "Low"}
    assert "detailedReasons" in response.json
    assert "scores" in response.json
    assert "reasons" in response.json
    assert "dataQualityNotes" in response.json


# Tests that the disabled embedding route does not run validation or detection.
def test_detect_uploaded_opponent_image_route_rejects_unsafe_filename(client, upload_dir, monkeypatch):
    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_dir)

    response = client.post("/api/opponent/detect", json={"filename": "../team.jpg"})

    assert response.status_code == 410
    assert response.json["activeRoute"] == "/api/opponent/detect-mobilenetv3"


# Tests that the disabled embedding route does not check for uploaded files.
def test_detect_uploaded_opponent_image_route_handles_missing_file(client, upload_dir, monkeypatch):
    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_dir)

    response = client.post("/api/opponent/detect", json={"filename": "missing.jpg"})

    assert response.status_code == 410
    assert response.json["activeRoute"] == "/api/opponent/detect-mobilenetv3"


# Tests that the disabled embedding route does not run the old CV detector.
def test_detect_uploaded_opponent_image_route_handles_cv_errors(client, upload_dir, monkeypatch):
    image_path = upload_dir / "team.jpg"
    image_path.write_bytes(VALID_JPEG)

    def raise_cv_error(_path):
        raise cv_service.ComputerVisionError("Could not read image.")

    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(routes, "detect_opponent_team_mobilenetv3", raise_cv_error)

    response = client.post("/api/opponent/detect", json={"filename": "team.jpg"})

    assert response.status_code == 410
    assert response.json["activeRoute"] == "/api/opponent/detect-mobilenetv3"


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


# Tests that standalone type-icon detection is disabled during MobileNetV3-only validation.
def test_detect_uploaded_opponent_types_route(client, upload_dir, monkeypatch):
    image_path = upload_dir / "team.jpg"
    image_path.write_bytes(VALID_JPEG)
    monkeypatch.setattr(routes, "UPLOAD_DIR", upload_dir)

    response = client.post("/api/opponent/types", json={"filename": "team.jpg"})

    assert response.status_code == 410
    assert response.json["activeRoute"] == "/api/opponent/detect-mobilenetv3"


def test_predict_pokemon_crop_route(client, monkeypatch):
    monkeypatch.setattr(
        routes,
        "predict_pokemon_crop",
        lambda _image: {
            "pokemon": "Garchomp",
            "confidence": 0.94,
            "top5": [
                {"name": "Garchomp", "confidence": 0.94},
                {"name": "Gabite", "confidence": 0.03},
            ],
            "model": "mobilenetv3",
            "productionDecision": "accept_mobilenetv3",
        },
    )

    response = client.post(
        "/api/opponent/predict-pokemon-crop",
        data={"image": (build_png_bytes(), "crop.png")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.json["pokemon"] == "Garchomp"
    assert response.json["model"] == "mobilenetv3"


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


def test_battle_advice_chat_route(client, monkeypatch):
    monkeypatch.setattr(
        routes,
        "answer_battle_question",
        lambda **kwargs: {
            "answer": f"Advice for {kwargs['question']}",
            "keyPoints": ["Grounded point."],
            "recommendedActions": ["Lead safely."],
            "relevantPokemon": ["Tyranitar"],
            "likelySetsUsed": [],
            "mechanicsUsed": [],
            "tournamentFactsUsed": [],
            "damageOrSpeedNotes": [],
            "uncertaintyNotes": [],
            "questionType": ["counterplay"],
            "source": "rule_based",
        },
    )

    response = client.post(
        "/api/battle/advice-chat",
        json={
            "question": "How do I beat this team?",
            "opponentTeam": [{"name": "Tyranitar"}, {"name": "Garchomp"}],
            "userTeam": [{"name": "Milotic"}],
            "damageContext": {},
            "speedContext": {},
        },
    )

    assert response.status_code == 200
    assert response.json["source"] == "rule_based"
    assert response.json["questionType"] == ["counterplay"]


def test_battle_advice_chat_route_requires_question(client):
    response = client.post("/api/battle/advice-chat", json={"question": "   "})

    assert response.status_code == 400
    assert response.json["error"] == "Question is required."


def test_battle_chat_failure_does_not_switch_to_team_strategy(client, monkeypatch):
    def fail(**kwargs):
        raise RuntimeError("Simulated lookup failure")

    def unexpected(**kwargs):
        raise AssertionError("API error handling must not invoke another tool")

    monkeypatch.setattr(routes, "answer_battle_question", fail)
    monkeypatch.setattr(routes, "build_battle_context", unexpected)
    response = client.post("/api/battle/advice-chat", json={
        "question": "sitrus berry", "userTeam": [{"name": "Pelipper"}],
    })
    assert response.status_code == 503
    assert "Please try again" in response.json["error"]
