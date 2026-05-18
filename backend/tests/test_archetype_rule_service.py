from services.archetype_cluster_service import find_matching_archetype_cluster
from services.archetype_feature_service import extract_team_features
from services.archetype_rule_service import (
    classify_team_archetype,
    get_archetype_rule_summary,
)


def test_extract_team_features_builds_expected_shape():
    features = extract_team_features(
        {
            "pokemon": [
                "Pelipper",
                "Basculegion",
                "Archaludon",
                "Amoonguss",
                "Tornadus",
                "Urshifu",
            ],
            "moves": [],
            "items": [],
            "abilities": [],
        }
    )

    assert features["teamSize"] == 6 if "teamSize" in features else len(features["pokemon_names"]) == 6
    assert "average_speed" in features
    assert "weather_setter_count" in features
    assert isinstance(features["moves"], list)
    assert isinstance(features["type_distribution"], dict)


def test_classify_team_archetype_detects_rain_shell():
    classification = classify_team_archetype(
        {
            "pokemon": [
                "Pelipper",
                "Basculegion",
                "Archaludon",
                "Amoonguss",
                "Tornadus",
                "Urshifu",
            ],
            "moves": [
                ["Hurricane", "Protect", "Tailwind", "Weather Ball"],
                ["Wave Crash", "Last Respects", "Protect", "Aqua Jet"],
                ["Electro Shot", "Draco Meteor", "Protect", "Flash Cannon"],
                ["Spore", "Rage Powder", "Pollen Puff", "Protect"],
                ["Bleakwind Storm", "Tailwind", "Taunt", "Protect"],
                ["Surging Strikes", "Close Combat", "Protect", "Aqua Jet"],
            ],
            "items": [
                "Focus Sash",
                "Mystic Water",
                "Assault Vest",
                "Sitrus Berry",
                "Covert Cloak",
                "Choice Scarf",
            ],
            "abilities": [
                "Drizzle",
                "Swift Swim",
                "Stamina",
                "Regenerator",
                "Prankster",
                "Unseen Fist",
            ],
        }
    )

    assert classification["primaryArchetype"] in {"Rain", "Weather Offense"}
    assert 0 <= classification["scores"]["Rain"]["score"] <= 1
    assert classification["reasons"]
    assert isinstance(classification["dataQualityNotes"], list)


def test_rule_summary_builds_and_contains_archetypes():
    summary = get_archetype_rule_summary()

    assert "archetypes" in summary
    assert isinstance(summary["archetypes"], (dict, list))
    assert len(summary["archetypes"]) > 0


def test_find_matching_archetype_cluster_returns_matching_parent_or_none():
    team = {
        "pokemon": [
            "Torkoal",
            "Venusaur",
            "Farigiraf",
            "Kingambit",
            "Flutter Mane",
            "Urshifu",
        ],
        "moves": [],
        "items": [],
        "abilities": [],
    }
    classification = classify_team_archetype(team)
    match = find_matching_archetype_cluster(team)

    assert match is None or match["primaryArchetype"] == classification["primaryArchetype"]
