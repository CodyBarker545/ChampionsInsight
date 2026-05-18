"""Quick local smoke-test for archetype detection and summaries."""

from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.archetype_cluster_service import (
    find_matching_archetype_cluster,
    get_archetype_cluster_summary,
)
from services.archetype_feature_service import extract_team_features
from services.archetype_rule_service import classify_team_archetype, get_archetype_rule_summary


SAMPLE_TEAMS = {
    "rain": {
        "pokemon": ["Pelipper", "Basculegion", "Archaludon", "Amoonguss", "Tornadus", "Urshifu"],
        "moves": [
            ["Hurricane", "Protect", "Tailwind", "Weather Ball"],
            ["Wave Crash", "Last Respects", "Protect", "Aqua Jet"],
            ["Electro Shot", "Draco Meteor", "Protect", "Flash Cannon"],
            ["Spore", "Rage Powder", "Pollen Puff", "Protect"],
            ["Bleakwind Storm", "Tailwind", "Taunt", "Protect"],
            ["Surging Strikes", "Close Combat", "Protect", "Aqua Jet"],
        ],
        "items": ["Focus Sash", "Mystic Water", "Assault Vest", "Sitrus Berry", "Covert Cloak", "Choice Scarf"],
        "abilities": ["Drizzle", "Swift Swim", "Stamina", "Regenerator", "Prankster", "Unseen Fist"],
    },
    "sun": {
        "pokemon": ["Torkoal", "Venusaur", "Farigiraf", "Flutter Mane", "Kingambit", "Urshifu"],
        "moves": [
            ["Eruption", "Heat Wave", "Protect", "Yawn"],
            ["Sleep Powder", "Leaf Storm", "Sludge Bomb", "Protect"],
            ["Trick Room", "Hyper Voice", "Helping Hand", "Protect"],
            ["Moonblast", "Shadow Ball", "Protect", "Icy Wind"],
            ["Kowtow Cleave", "Sucker Punch", "Iron Head", "Protect"],
            ["Surging Strikes", "Close Combat", "Protect", "Aqua Jet"],
        ],
        "items": ["Charcoal", "Focus Sash", "Mental Herb", "Booster Energy", "Black Glasses", "Mystic Water"],
        "abilities": ["Drought", "Chlorophyll", "Armor Tail", "Protosynthesis", "Defiant", "Unseen Fist"],
    },
}


def main() -> None:
    cluster_summary = get_archetype_cluster_summary()
    rule_summary = get_archetype_rule_summary()

    for label, team in SAMPLE_TEAMS.items():
        features = extract_team_features(team)
        classification = classify_team_archetype(features)
        matching_cluster = find_matching_archetype_cluster(team)

        print(f"== {label.upper()} sample ==")
        print(json.dumps(
            {
                "pokemon": team["pokemon"],
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
                "reasons": classification.get("reasons"),
                "dataQualityNotes": classification.get("dataQualityNotes"),
                "matchingCluster": matching_cluster,
            },
            indent=2,
        ))
        print()

    print("== Summary counts ==")
    print(json.dumps(
        {
            "ruleArchetypeCount": len(rule_summary.get("archetypes", {})),
            "clusterCount": len(cluster_summary.get("clusters", [])),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
