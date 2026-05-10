"""Loads the local demo user team used by the main page."""

import json

from paths import USER_TEAM_PATH


# Loads the locally stored demo team for the current user.
def load_user_team():
    if not USER_TEAM_PATH.exists():
        raise FileNotFoundError("User team file not found.")

    with USER_TEAM_PATH.open("r", encoding="utf-8") as team_file:
        return json.load(team_file)


def save_user_team(team, user_id="local-demo-user", team_name="Saved Battle Team"):
    if not isinstance(team, list) or len(team) != 6:
        raise ValueError("A saved team must include exactly six Pokemon.")

    saved_team = {
        "userId": user_id or "local-demo-user",
        "teamName": team_name or "Saved Battle Team",
        "team": team,
    }

    USER_TEAM_PATH.parent.mkdir(parents=True, exist_ok=True)
    USER_TEAM_PATH.write_text(
        json.dumps(saved_team, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return saved_team
