"""Integration test for real opponent image detection, type detection, stat enrichment, and expected team comparison."""

import json
from pathlib import Path

import pytest

from services.cv_detection_service import detect_opponent_team
from services.pokemon_stats_service import get_level_50_stats


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "opponent_expected_team.json"

TEST_IMAGE_PATH = (
    Path(__file__).parent
    / "test_assets"
    / "2026-05-0318.09.556603716296042018193-48279c5b.jpg"
)


def load_expected_fixture():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def normalize_name(name):
    return "".join(char.lower() for char in str(name) if char.isalnum())


def normalize_types(types):
    return [str(pokemon_type).lower() for pokemon_type in types]


def get_detected_name(pokemon):
    return (
        pokemon.get("name")
        or pokemon.get("pokemonName")
        or pokemon.get("species")
        or pokemon.get("label")
    )


def get_detected_types(pokemon):
    return pokemon.get("types") or pokemon.get("detectedTypes") or []


@pytest.mark.integration
def test_real_picture_detects_expected_team_types_and_level_50_stats():
    if not TEST_IMAGE_PATH.exists():
        pytest.skip(f"Test image is missing: {TEST_IMAGE_PATH}")

    fixture = load_expected_fixture()
    expected_team = fixture["expectedTeam"]

    detection_result = detect_opponent_team(TEST_IMAGE_PATH)
    detected_team = detection_result.get("detectedTeam", [])

    print("\n================ REAL OPPONENT IMAGE TEST ================")
    print(f"Image: {TEST_IMAGE_PATH}")
    print(f"Detected count: {len(detected_team)}")
    print("Raw detection result:")
    print(json.dumps(detection_result, indent=2))

    assert len(detected_team) == 6

    for index, expected in enumerate(expected_team):
        actual = detected_team[index]

        detected_name = get_detected_name(actual)
        detected_types = get_detected_types(actual)

        stats = get_level_50_stats(detected_name, nature="hardy")

        print("\n--------------------------------------------------")
        print(f"Slot {index + 1}")
        print(f"Expected Pokemon: {expected['name']}")
        print(f"Detected Pokemon: {detected_name}")
        print(f"Expected Types: {expected['types']}")
        print(f"Detected Types: {detected_types}")
        print(f"Expected Level 50 Stats: {expected['neutralStats']}")

        if stats:
            print(f"Base Stats: {stats['baseStats']}")
            print(f"Actual Level 50 Stats: {stats['finalStats']}")
            print(f"Image: {stats.get('image')}")
        else:
            print("Stats: None")

        assert detected_name is not None
        assert normalize_name(detected_name) == normalize_name(expected["name"])

        assert normalize_types(detected_types) == normalize_types(expected["types"])

        assert stats is not None
        assert stats["level"] == 50
        assert stats["ivs"] == {
            "hp": 31,
            "attack": 31,
            "defense": 31,
            "special_attack": 31,
            "special_defense": 31,
            "speed": 31,
        }

        assert stats["types"] == expected["types"]
        assert stats["finalStats"] == expected["neutralStats"]


@pytest.mark.integration
def test_real_picture_detected_team_matches_expected_json_exact_order():
    if not TEST_IMAGE_PATH.exists():
        pytest.skip(f"Test image is missing: {TEST_IMAGE_PATH}")

    fixture = load_expected_fixture()
    expected_team = fixture["expectedTeam"]

    detection_result = detect_opponent_team(TEST_IMAGE_PATH)
    detected_team = detection_result.get("detectedTeam", [])

    detected_summary = []

    for actual in detected_team:
        detected_name = get_detected_name(actual)
        stats = get_level_50_stats(detected_name, nature="hardy")

        detected_summary.append({
            "name": detected_name,
            "types": normalize_types(get_detected_types(actual)),
            "neutralStats": stats["finalStats"] if stats else None,
        })

    expected_summary = [
        {
            "name": pokemon["name"],
            "types": normalize_types(pokemon["types"]),
            "neutralStats": pokemon["neutralStats"],
        }
        for pokemon in expected_team
    ]

    print("\n================ EXPECTED VS ACTUAL SUMMARY ================")
    print("Expected:")
    print(json.dumps(expected_summary, indent=2))
    print("\nActual:")
    print(json.dumps(detected_summary, indent=2))

    assert detected_summary == expected_summary