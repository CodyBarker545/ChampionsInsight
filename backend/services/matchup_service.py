"""Provides matchup analysis responses for the battle prep workflow."""

# Returns mock matchup analysis until the real calculator is implemented.
def analyze_matchup(payload):
    team = payload.get("team", [])
    opponent = payload.get("opponent", "")

    if not team:
        raise ValueError("At least one team member is required.")

    lead = team[0]
    lead_name = lead.get("name", "Your lead")
    opponent_name = opponent or "the opponent"

    return {
        "summary": f"{lead_name} has a favorable opening into {opponent_name}.",
        "speed": {
            "result": "Your lead is faster",
            "yourSpeed": 152,
            "opponentSpeed": 128,
        },
        "damage": {
            "range": "42% - 51%",
            "koChance": "Guaranteed 3HKO",
        },
        "recommendations": [
            "Lead with your fastest attacker while preserving defensive switch-ins.",
            "Use manual correction if image detection cannot identify the opponent team.",
            "Replace this mock response with the real calculator module next.",
        ],
    }
