"""Print team-level set predictions for sample opponent teams."""

from __future__ import annotations

import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
for path in (PROJECT_ROOT, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from services.set_probability_service import predict_team_sets


TEST_CASES = [
    {
        "label": "full_team",
        "team": [
            "Tyranitar",
            "Garchomp",
            "Amoonguss",
            "Incineroar",
            "Rillaboom",
            "Gholdengo",
        ],
    },
    {
        "label": "partial_team",
        "team": [
            "Sneasler",
            "Rillaboom",
            "Incineroar",
        ],
    },
    {
        "label": "unknown_member",
        "team": [
            "Sneasler",
            "Indeedee",
            "DefinitelyNotAPokemon",
        ],
    },
    {
        "label": "empty_team",
        "team": [],
    },
]


def main() -> None:
    for case in TEST_CASES:
        print(f"== {case['label']} ==")
        print(json.dumps(predict_team_sets(case["team"]), indent=2))
        print()


if __name__ == "__main__":
    main()
