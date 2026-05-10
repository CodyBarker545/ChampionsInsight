from services.pokedex_service import (
    get_all_pokedex_entries,
    get_pokedex_entry_by_name,
    get_pokedex_entry_detail_by_name,
    get_pokedex_grid_entries,
    normalize_pokemon_from_lookup,
    normalize_stats,
)


def test_normalize_stats_handles_common_stat_keys():
    raw_stats = {
        "hp": 80,
        "attack": 82,
        "defense": 83,
        "special_attack": 100,
        "special_defense": 100,
        "speed": 80,
    }

    stats = normalize_stats(raw_stats)

    assert stats["hp"] == 80
    assert stats["attack"] == 82
    assert stats["defense"] == 83
    assert stats["specialAttack"] == 100
    assert stats["specialDefense"] == 100
    assert stats["speed"] == 80


def test_normalize_pokemon_from_lookup_returns_frontend_shape():
    form_entry = {
        "species_display_name": "Venusaur",
        "form_display_name": "Venusaur",
        "form_api_name": "venusaur",
        "is_default": True,
        "types": ["grass", "poison"],
        "base_stats": {
            "hp": 80,
            "attack": 82,
            "defense": 83,
            "special_attack": 100,
            "special_defense": 100,
            "speed": 80,
        },
        "abilities": [
            {
                "name": "overgrow",
                "display_name": "Overgrow",
                "is_hidden": False,
            },
            {
                "name": "chlorophyll",
                "display_name": "Chlorophyll",
                "is_hidden": True,
            },
        ],
    }

    pokemon = normalize_pokemon_from_lookup(
        form_entry=form_entry,
        sprite_filename="Menu_CP_0003.png",
        dex_number=3,
    )

    assert pokemon["id"] == 3
    assert pokemon["name"] == "Venusaur"
    assert pokemon["speciesName"] == "Venusaur"
    assert pokemon["formApiName"] == "venusaur"
    assert pokemon["types"] == ["grass", "poison"]
    assert pokemon["baseStats"]["hp"] == 80
    assert pokemon["baseStats"]["specialAttack"] == 100
    assert pokemon["spriteUrl"] == "/api/pokemon/sprite/normal/Menu_CP_0003.png"
    assert pokemon["isMega"] is False

    assert pokemon["moves"] == []
    assert len(pokemon["abilities"]) == 2
    assert pokemon["abilities"][0]["displayName"] == "Overgrow"
    assert pokemon["abilities"][1]["isHidden"] is True


def test_get_all_pokedex_entries_returns_list():
    entries = get_all_pokedex_entries()

    assert isinstance(entries, tuple)
    assert len(entries) > 0

    first = entries[0]

    assert "id" in first
    assert "name" in first
    assert "speciesName" in first
    assert "formApiName" in first
    assert "types" in first
    assert "abilities" in first
    assert "moves" in first
    assert "baseStats" in first
    assert "spriteUrl" in first


def test_get_pokedex_grid_entries_returns_lightweight_list():
    entries = get_pokedex_grid_entries()

    assert isinstance(entries, list)
    assert len(entries) > 0

    first = entries[0]

    assert "id" in first
    assert "name" in first
    assert "spriteUrl" in first
    assert "types" in first
    assert "baseStats" in first
    assert "moves" not in first


def test_get_all_pokedex_entries_uses_real_names_and_stats():
    entries = get_all_pokedex_entries()

    venusaur = next(
        pokemon for pokemon in entries
        if pokemon["name"].lower() == "venusaur"
    )

    assert venusaur["id"] == 3
    assert venusaur["types"] == ["grass", "poison"]
    assert venusaur["baseStats"]["hp"] == 80
    assert venusaur["baseStats"]["attack"] == 82
    assert venusaur["baseStats"]["specialAttack"] == 100
    assert len(venusaur["abilities"]) > 0


def test_get_pokedex_entry_by_name_finds_known_pokemon():
    pokemon = get_pokedex_entry_by_name("Venusaur")

    assert pokemon is not None
    assert pokemon["name"] == "Venusaur"
    assert pokemon["id"] == 3


def test_get_pokedex_entry_by_name_finds_form_api_name():
    pokemon = get_pokedex_entry_by_name("venusaur")

    assert pokemon is not None
    assert pokemon["name"] == "Venusaur"


def test_get_pokedex_entry_by_name_returns_none_for_missing_pokemon():
    pokemon = get_pokedex_entry_by_name("NotARealPokemonName123")

    assert pokemon is None


def test_get_pokedex_entry_detail_by_name_includes_moves_and_usage():
    pokemon = get_pokedex_entry_detail_by_name("Venusaur")

    assert pokemon is not None
    assert pokemon["name"] == "Venusaur"

    assert "moves" in pokemon
    assert isinstance(pokemon["moves"], list)
    assert len(pokemon["moves"]) > 0

    assert "usage" in pokemon

    usage = pokemon["usage"]

    assert "appearances" in usage
    assert "wins" in usage
    assert "losses" in usage
    assert "games" in usage
    assert "winRate" in usage
    assert "topMoves" in usage
    assert "topItems" in usage
    assert "topTeams" in usage