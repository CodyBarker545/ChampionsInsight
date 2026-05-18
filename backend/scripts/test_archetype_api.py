"""Simple smoke test for archetype API routes."""

from __future__ import annotations

import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
for path in (PROJECT_ROOT, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from backend.app import app


def main() -> None:
    client = app.test_client()

    rules_response = client.get("/api/tournament/archetypes/rules")
    print("GET /api/tournament/archetypes/rules")
    print(rules_response.status_code)
    print(json.dumps(rules_response.get_json(), indent=2)[:800])
    print()

    clusters_response = client.get("/api/tournament/archetypes/clusters")
    print("GET /api/tournament/archetypes/clusters")
    print(clusters_response.status_code)
    print(json.dumps(clusters_response.get_json(), indent=2)[:800])
    print()

    classify_response = client.post(
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
    print("POST /api/tournament/archetypes/classify")
    print(classify_response.status_code)
    print(json.dumps(classify_response.get_json(), indent=2)[:1600])


if __name__ == "__main__":
    main()
