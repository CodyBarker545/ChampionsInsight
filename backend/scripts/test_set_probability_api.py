"""Smoke test the Flask set probability API routes."""

from __future__ import annotations

import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
for path in (PROJECT_ROOT, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app import app


def print_response(label: str, response) -> None:
    print(label)
    print(f"status={response.status_code}")
    try:
        print(json.dumps(response.get_json(), indent=2))
    except Exception:
        print(response.get_data(as_text=True))
    print()


def main() -> None:
    client = app.test_client()

    print_response(
        "GET /api/tournament/set-probabilities/Sneasler",
        client.get("/api/tournament/set-probabilities/Sneasler"),
    )
    print_response(
        "GET /api/tournament/set-probabilities/DefinitelyNotAPokemon",
        client.get("/api/tournament/set-probabilities/DefinitelyNotAPokemon"),
    )
    print_response(
        "POST /api/tournament/set-probabilities/context",
        client.post(
            "/api/tournament/set-probabilities/context",
            json={
                "pokemon": "Sneasler",
                "teammates": ["Indeedee"],
                "archetype": "Psychic Terrain Offense",
            },
        ),
    )
    print_response(
        "POST /api/tournament/set-probabilities/context missing pokemon",
        client.post(
            "/api/tournament/set-probabilities/context",
            json={"teammates": ["Indeedee"]},
        ),
    )
    print_response(
        "POST /api/tournament/set-probabilities/team",
        client.post(
            "/api/tournament/set-probabilities/team",
            json={
                "opponentTeam": ["Sneasler", "Indeedee", "Garchomp", "Tyranitar"],
            },
        ),
    )
    print_response(
        "POST /api/tournament/set-probabilities/team empty",
        client.post(
            "/api/tournament/set-probabilities/team",
            json={"opponentTeam": []},
        ),
    )
    print_response(
        "POST /api/tournament/set-probabilities/team too large",
        client.post(
            "/api/tournament/set-probabilities/team",
            json={
                "opponentTeam": [
                    "Sneasler",
                    "Indeedee",
                    "Garchomp",
                    "Tyranitar",
                    "Rillaboom",
                    "Amoonguss",
                    "Incineroar",
                ],
            },
        ),
    )


if __name__ == "__main__":
    main()
