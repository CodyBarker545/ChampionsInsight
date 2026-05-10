def test_get_pokedex_route_returns_list(client):
    response = client.get("/api/pokedex")

    assert response.status_code == 200

    data = response.get_json()

    assert "count" in data
    assert "pokemon" in data
    assert isinstance(data["pokemon"], list)
    assert data["count"] == len(data["pokemon"])
    assert data["count"] > 0


def test_get_pokedex_route_has_frontend_fields(client):
    response = client.get("/api/pokedex")

    assert response.status_code == 200

    data = response.get_json()
    first = data["pokemon"][0]

    assert "id" in first
    assert "name" in first
    assert "speciesName" in first
    assert "formApiName" in first
    assert "types" in first
    assert "abilities" in first
    assert "baseStats" in first
    assert "spriteUrl" in first
    assert "moves" not in first


def test_get_single_pokemon_route_returns_detail_pokemon(client):
    response = client.get("/api/pokedex/Venusaur")

    assert response.status_code == 200

    data = response.get_json()

    assert data["name"] == "Venusaur"
    assert data["id"] == 3
    assert "baseStats" in data
    assert "spriteUrl" in data

    assert "moves" in data
    assert isinstance(data["moves"], list)
    assert len(data["moves"]) > 0

    assert "usage" in data
    assert "appearances" in data["usage"]
    assert "topMoves" in data["usage"]
    assert "topItems" in data["usage"]
    assert "topTeams" in data["usage"]


def test_get_single_pokemon_route_returns_404_for_missing_pokemon(client):
    response = client.get("/api/pokedex/NotARealPokemonName123")

    assert response.status_code == 404

    data = response.get_json()

    assert data["error"] == "Pokemon not found"