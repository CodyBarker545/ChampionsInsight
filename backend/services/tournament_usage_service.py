import json

from paths import POKEDEX_USAGE_SUMMARY_PATH


EMPTY_USAGE_SUMMARY = {
    "appearances": 0,
    "wins": 0,
    "losses": 0,
    "games": 0,
    "winRate": 0,
    "topMoves": [],
    "topItems": [],
    "topTeams": [],
}


def load_usage_summary_data():
    if not POKEDEX_USAGE_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Missing pokedex usage summary JSON. Run this first from backend:\n"
            "python scripts\\build_pokedex_usage_summary.py"
        )

    with open(POKEDEX_USAGE_SUMMARY_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_name(value):
    return str(value or "").strip().lower()


def get_pokemon_usage_summary(pokemon_name: str):
    data = load_usage_summary_data()
    target = normalize_name(pokemon_name)

    for name, summary in data.items():
        if normalize_name(name) == target:
            return summary

    return dict(EMPTY_USAGE_SUMMARY)
