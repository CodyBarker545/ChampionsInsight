"""Tests level 50 Pokemon stat calculation and nature modifiers."""

import json
from pathlib import Path

import pytest

from services.pokemon_stats_service import get_level_50_stats


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "opponent_expected_team.json"


def load_expected_team():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["expectedTeam"]


@pytest.mark.parametrize("expected_pokemon", load_expected_team())
def test_get_level_50_stats_returns_expected_neutral_stats(expected_pokemon):
    result = get_level_50_stats(expected_pokemon["name"], nature="hardy")

    assert result is not None
    assert result["level"] == 50
    assert result["nature"] == "hardy"
    assert result["ivs"] == {
        "hp": 31,
        "attack": 31,
        "defense": 31,
        "special_attack": 31,
        "special_defense": 31,
        "speed": 31,
    }

    assert result["finalStats"] == expected_pokemon["neutralStats"]


@pytest.mark.parametrize("expected_pokemon", load_expected_team())
def test_get_level_50_stats_applies_nature_boosts_correctly(expected_pokemon):
    nature_name = expected_pokemon["natureTest"]["nature"]
    expected_stats = expected_pokemon["natureTest"]["expectedStats"]

    result = get_level_50_stats(expected_pokemon["name"], nature=nature_name)

    assert result is not None
    assert result["nature"] == nature_name
    assert result["finalStats"] == expected_stats


@pytest.mark.parametrize("expected_pokemon", load_expected_team())
def test_get_level_50_stats_returns_expected_types(expected_pokemon):
    result = get_level_50_stats(expected_pokemon["name"], nature="hardy")

    assert result is not None
    assert result["types"] == expected_pokemon["types"]


def test_get_level_50_stats_returns_none_for_unknown_pokemon():
    result = get_level_50_stats("NotAPokemon", nature="hardy")

    assert result is None
    
def test_print_expected_team_stats_report():
    for pokemon in load_expected_team():
        result = get_level_50_stats(pokemon["name"], nature="hardy")

        print("\n==============================")
        print(f"{pokemon['name']} - Level 50")
        print(f"Types: {result['types']}")
        print(f"Image: {result.get('image')}")
        print(f"Base Stats: {result['baseStats']}")
        print(f"Final Stats: {result['finalStats']}")

    assert True