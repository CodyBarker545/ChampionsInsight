"""Smoke-test script for the Battle Advice Chat service."""

from __future__ import annotations

import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.battle_advice_chat_service import answer_battle_question


TEST_QUESTIONS = [
    "What is this opponent team trying to do?",
    "What item is Sneasler likely holding?",
    "How do I beat Trick Room?",
    "What does Tailwind do?",
    "What should I lead into this team?",
    "Is Garchomp likely to run Earthquake?",
    "What are the biggest threats?",
    "How do I counter Sand teams?",
]


def main() -> int:
    opponent_team = [
        {"name": "Tyranitar"},
        {"name": "Garchomp"},
        {"name": "Sneasler"},
        {"name": "Indeedee"},
    ]
    user_team = [
        {"name": "Milotic"},
        {"name": "Dragonite"},
        {"name": "Garchomp"},
        {"name": "Amoonguss"},
    ]

    for question in TEST_QUESTIONS:
        print(f"\n=== {question} ===")
        result = answer_battle_question(
            question=question,
            opponent_team=opponent_team,
            user_team=user_team,
            damage_context={},
            speed_context={},
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
