import json

from services.archetype_cluster_service import find_matching_archetype_cluster
from services.archetype_feature_service import extract_team_features
from services.archetype_rule_service import classify_team_archetype


SAMPLE_TEAMS = {
    "Rain": {
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
    },
    "Sun": {
        "pokemon": [
            "Torkoal",
            "Venusaur",
            "Lilligant",
            "Charizard",
            "Flutter Mane",
            "Kingambit",
        ],
        "moves": [
            ["Eruption", "Heat Wave", "Protect", "Yawn"],
            ["Sleep Powder", "Leaf Storm", "Sludge Bomb", "Protect"],
            ["After You", "Solar Blade", "Sleep Powder", "Protect"],
            ["Heat Wave", "Air Slash", "Solar Beam", "Protect"],
            ["Moonblast", "Shadow Ball", "Protect", "Icy Wind"],
            ["Kowtow Cleave", "Sucker Punch", "Iron Head", "Protect"],
        ],
        "items": [
            "Charcoal",
            "Focus Sash",
            "Focus Sash",
            "Charizardite Y",
            "Booster Energy",
            "Black Glasses",
        ],
        "abilities": [
            "Drought",
            "Chlorophyll",
            "Chlorophyll",
            "Blaze",
            "Protosynthesis",
            "Defiant",
        ],
    },
    "Sand": {
        "pokemon": [
            "Tyranitar",
            "Garchomp",
            "Excadrill",
            "Kingambit",
            "Rotom Wash",
            "Amoonguss",
        ],
        "moves": [
            ["Rock Slide", "Crunch", "Protect", "Low Kick"],
            ["Earthquake", "Rock Slide", "Protect", "Dragon Claw"],
            ["Iron Head", "High Horsepower", "Protect", "Swords Dance"],
            ["Kowtow Cleave", "Sucker Punch", "Protect", "Iron Head"],
            ["Thunderbolt", "Hydro Pump", "Will O Wisp", "Protect"],
            ["Spore", "Rage Powder", "Pollen Puff", "Protect"],
        ],
        "items": [
            "Chople Berry",
            "Clear Amulet",
            "Focus Sash",
            "Black Glasses",
            "Sitrus Berry",
            "Rocky Helmet",
        ],
        "abilities": [
            "Sand Stream",
            "Rough Skin",
            "Sand Rush",
            "Defiant",
            "Levitate",
            "Regenerator",
        ],
    },
    "Trick Room": {
        "pokemon": [
            "Cresselia",
            "Farigiraf",
            "Torkoal",
            "Rhyperior",
            "Iron Hands",
            "Amoonguss",
        ],
        "moves": [
            ["Trick Room", "Helping Hand", "Lunar Blessing", "Protect"],
            ["Trick Room", "Hyper Voice", "Helping Hand", "Protect"],
            ["Eruption", "Heat Wave", "Protect", "Earth Power"],
            ["Rock Slide", "High Horsepower", "Protect", "Swords Dance"],
            ["Fake Out", "Drain Punch", "Wild Charge", "Protect"],
            ["Spore", "Rage Powder", "Pollen Puff", "Protect"],
        ],
        "items": [
            "Mental Herb",
            "Safety Goggles",
            "Charcoal",
            "Weakness Policy",
            "Assault Vest",
            "Sitrus Berry",
        ],
        "abilities": [
            "Levitate",
            "Armor Tail",
            "Drought",
            "Solid Rock",
            "Quark Drive",
            "Regenerator",
        ],
    },
    "Hyper Offense": {
        "pokemon": [
            "Dragonite",
            "Garchomp",
            "Weavile",
            "Gholdengo",
            "Baxcalibur",
            "Talonflame",
        ],
        "moves": [
            ["Dragon Dance", "Extreme Speed", "Earthquake", "Protect"],
            ["Swords Dance", "Earthquake", "Rock Slide", "Protect"],
            ["Icicle Crash", "Knock Off", "Ice Shard", "Protect"],
            ["Nasty Plot", "Make It Rain", "Shadow Ball", "Protect"],
            ["Dragon Dance", "Icicle Spear", "Glaive Rush", "Protect"],
            ["Tailwind", "Brave Bird", "Flare Blitz", "Taunt"],
        ],
        "items": [
            "Lum Berry",
            "Clear Amulet",
            "Focus Sash",
            "Choice Specs",
            "Loaded Dice",
            "Covert Cloak",
        ],
        "abilities": [
            "Inner Focus",
            "Rough Skin",
            "Pressure",
            "Good as Gold",
            "Thermal Exchange",
            "Gale Wings",
        ],
    },
    "Balance": {
        "pokemon": [
            "Incineroar",
            "Rotom Wash",
            "Amoonguss",
            "Garchomp",
            "Flutter Mane",
            "Kingambit",
        ],
        "moves": [
            ["Fake Out", "Flare Blitz", "Parting Shot", "Protect"],
            ["Thunderbolt", "Hydro Pump", "Will O Wisp", "Protect"],
            ["Spore", "Rage Powder", "Pollen Puff", "Protect"],
            ["Earthquake", "Rock Slide", "Swords Dance", "Protect"],
            ["Moonblast", "Shadow Ball", "Icy Wind", "Protect"],
            ["Kowtow Cleave", "Sucker Punch", "Iron Head", "Protect"],
        ],
        "items": [
            "Sitrus Berry",
            "Leftovers",
            "Rocky Helmet",
            "Clear Amulet",
            "Booster Energy",
            "Black Glasses",
        ],
        "abilities": [
            "Intimidate",
            "Levitate",
            "Regenerator",
            "Rough Skin",
            "Protosynthesis",
            "Defiant",
        ],
    },
}


EXPECTED_PRIMARY = {
    "Rain": {"Rain", "Weather Offense"},
    "Sun": {"Sun", "Weather Offense"},
    "Sand": {"Sand", "Weather Offense", "Balance"},
    "Trick Room": {"Trick Room", "Semi-Room"},
    "Hyper Offense": {"Hyper Offense", "Tailwind Offense"},
    "Balance": {"Balance", "Bulky Setup"},
}

EXPECTED_DISPLAY_LABELS = {
    "Rain": {"Rain Offense", "Weather Offense", "Rain"},
    "Sun": {"Sun", "Sun Trick Room", "Weather Offense"},
    "Sand": {"Sand Balance", "Sand Offense", "Sand"},
    "Trick Room": {"Hard Trick Room", "Semi-Room", "Trick Room", "Sun Trick Room"},
    "Hyper Offense": {"Hyper Offense", "Tailwind Offense"},
    "Balance": {"Balance", "Bulky Setup", "Control Balance"},
}


def _debug_report(team_name, features, classification):
    report = {
        "teamName": team_name,
        "pokemon": features.get("pokemon_names"),
        "primaryArchetype": classification.get("primaryArchetype"),
        "secondaryArchetype": classification.get("secondaryArchetype"),
        "confidence": classification.get("confidence"),
        "topScores": {
            archetype: payload["score"]
            for archetype, payload in sorted(
                classification.get("scores", {}).items(),
                key=lambda item: item[1]["score"],
                reverse=True,
            )[:4]
        },
        "reasons": classification.get("reasons", []),
        "dataQualityNotes": classification.get("dataQualityNotes", []),
    }
    return json.dumps(report, indent=2)


def test_feature_extraction_handles_known_and_unknown_pokemon_without_crashing():
    features = extract_team_features(
        {
            "pokemon": ["Pelipper", "UnknownMon", "Archaludon", "MysteryMon"],
            "moves": [[], [], [], []],
            "items": ["", "", "", ""],
            "abilities": ["", "", "", ""],
        }
    )

    assert features["pokemon_names"] == [
        "pelipper",
        "unknownmon",
        "archaludon",
        "mysterymon",
    ]
    assert 0 <= features["average_speed"]
    assert "missing Pokémon stats" in features["data_quality_notes"]


def test_rule_scores_stay_between_zero_and_one_for_sample_teams():
    for team_name, team in SAMPLE_TEAMS.items():
        classification = classify_team_archetype(team)

        for archetype_name, payload in classification["scores"].items():
            assert 0 <= payload["score"] <= 1, (
                f"Score out of bounds for {team_name} -> {archetype_name}\n"
                f"{json.dumps(payload, indent=2)}"
            )


def test_sample_teams_produce_reasonable_primary_archetypes():
    for team_name, team in SAMPLE_TEAMS.items():
        features = extract_team_features(team)
        classification = classify_team_archetype(features)

        assert classification["primaryArchetype"] in EXPECTED_PRIMARY[team_name], (
            f"Unexpected archetype classification.\n{_debug_report(team_name, features, classification)}"
        )
        assert classification["reasons"], (
            f"Expected reasons for {team_name} classification.\n"
            f"{_debug_report(team_name, features, classification)}"
        )
        assert classification["displayLabel"] in EXPECTED_DISPLAY_LABELS[team_name], (
            f"Unexpected display label.\n{_debug_report(team_name, features, classification)}"
        )
        assert classification["shortExplanation"]
        assert classification["confidenceLabel"] in {"High", "Medium", "Low"}
        assert classification["detailedReasons"]["primary"]["summary"]


def test_missing_moves_and_items_do_not_crash_and_return_notes():
    classification = classify_team_archetype(
        {
            "pokemon": SAMPLE_TEAMS["Balance"]["pokemon"],
            "moves": [],
            "items": [],
            "abilities": SAMPLE_TEAMS["Balance"]["abilities"],
        }
    )

    assert classification["primaryArchetype"]
    assert "missing moves" in classification["dataQualityNotes"]
    assert "missing items" in classification["dataQualityNotes"]
    assert classification["reasons"]


def test_cluster_matching_returns_parent_consistent_with_classification(monkeypatch):
    team = SAMPLE_TEAMS["Rain"]

    monkeypatch.setattr(
        "services.archetype_cluster_service.get_archetype_cluster_summary",
        lambda force_rebuild=False: {
            "clusters": [
                {
                    "clusterId": "rain-1",
                    "parentArchetype": "Rain",
                    "suggestedLabel": "Pelipper Rain",
                    "commonPokemon": [
                        {"name": "Pelipper", "count": 20},
                        {"name": "Basculegion", "count": 15},
                        {"name": "Archaludon", "count": 14},
                    ],
                    "commonMoves": [{"name": "Hurricane", "count": 22}],
                    "commonItems": [{"name": "Mystic Water", "count": 16}],
                    "averageTeamStats": {
                        "averageSpeed": 86,
                        "averageAttack": 101,
                        "averageSpecialAttack": 98,
                        "averageBulk": 255,
                    },
                },
                {
                    "clusterId": "balance-1",
                    "parentArchetype": "Balance",
                    "suggestedLabel": "Balance",
                    "commonPokemon": [{"name": "Incineroar", "count": 20}],
                    "commonMoves": [{"name": "Fake Out", "count": 20}],
                    "commonItems": [{"name": "Sitrus Berry", "count": 20}],
                    "averageTeamStats": {
                        "averageSpeed": 80,
                        "averageAttack": 90,
                        "averageSpecialAttack": 90,
                        "averageBulk": 275,
                    },
                },
            ]
        },
    )

    classification = classify_team_archetype(team)
    match = find_matching_archetype_cluster(team)

    assert match is not None, _debug_report("Rain", extract_team_features(team), classification)
    assert match["primaryArchetype"] == classification["primaryArchetype"]
    assert 0 <= match["similarityScore"] <= 1


def test_rules_and_clusters_routes_return_useful_shapes(client):
    rules_response = client.get("/api/tournament/archetypes/rules")
    clusters_response = client.get("/api/tournament/archetypes/clusters")

    assert rules_response.status_code == 200
    assert "totalTeamsAnalyzed" in rules_response.json
    assert "archetypeSummaries" in rules_response.json
    assert isinstance(rules_response.json["archetypeSummaries"], list)

    assert clusters_response.status_code == 200
    assert "totalTeamsAnalyzed" in clusters_response.json
    assert "totalClusters" in clusters_response.json
    assert "clustersByParentArchetype" in clusters_response.json
    assert isinstance(clusters_response.json["clustersByParentArchetype"], dict)


def test_classify_route_returns_helpful_error_messages(client):
    missing_body_response = client.post(
        "/api/tournament/archetypes/classify",
        data="not-json",
        content_type="text/plain",
    )
    unknown_response = client.post(
        "/api/tournament/archetypes/classify",
        json={"pokemon": ["UnknownMon", "FakeMon"]},
    )

    assert missing_body_response.status_code == 400
    assert missing_body_response.json["error"] == "A JSON request body is required."

    assert unknown_response.status_code == 400
    assert unknown_response.json["error"] == "None of the provided Pokemon names were recognized."


def test_classify_route_handles_partial_unknowns_without_crashing(client):
    response = client.post(
        "/api/tournament/archetypes/classify",
        json={
            "pokemon": ["Pelipper", "UnknownMon", "Archaludon", "Amoonguss"],
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
    assert isinstance(response.json["reasons"], list)
    assert "missing Pokémon stats" in response.json["dataQualityNotes"]
