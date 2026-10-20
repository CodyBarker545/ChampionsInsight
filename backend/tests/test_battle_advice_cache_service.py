"""Tests JSON-backed set prediction helpers for battle advice."""

from __future__ import annotations

from services import battle_advice_cache_service


def test_predict_team_sets_from_cache_uses_cached_pair_shifts(monkeypatch):
    monkeypatch.setattr(
        battle_advice_cache_service,
        "load_set_probability_summary_cache",
        lambda: {
            "garchomp": {
                "pokemon": "Garchomp",
                "sampleSize": 100,
                "moves": [
                    {"name": "Earthquake", "count": 90, "percentage": 90.0},
                    {"name": "Dragon Claw", "count": 75, "percentage": 75.0},
                ],
                "items": [
                    {"name": "Clear Amulet", "count": 20, "percentage": 20.0},
                    {"name": "Life Orb", "count": 18, "percentage": 18.0},
                ],
                "abilities": [
                    {"name": "Rough Skin", "count": 100, "percentage": 100.0},
                ],
                "natures": [
                    {"name": "Jolly", "count": 60, "percentage": 60.0},
                ],
                "commonTeammates": [],
                "dataQualityNotes": [],
            },
            "sneasler": {
                "pokemon": "Sneasler",
                "sampleSize": 120,
                "moves": [],
                "items": [],
                "abilities": [],
                "natures": [],
                "commonTeammates": [],
                "dataQualityNotes": [],
            },
        },
    )
    monkeypatch.setattr(
        battle_advice_cache_service,
        "load_conditional_pair_summary_cache",
        lambda: {
            ("garchomp", "sneasler"): {
                "pokemon": "Garchomp",
                "teammate": "Sneasler",
                "matchedSampleSize": 30,
                "conditionalMoves": [
                    {
                        "name": "Earthquake",
                        "basePercentage": 90.0,
                        "conditionalPercentage": 92.0,
                        "change": 2.0,
                        "count": 92,
                    }
                ],
                "conditionalItems": [
                    {
                        "name": "Clear Amulet",
                        "basePercentage": 20.0,
                        "conditionalPercentage": 42.0,
                        "change": 22.0,
                        "count": 42,
                    }
                ],
                "biggestProbabilityIncreases": [
                    {
                        "category": "item",
                        "name": "Clear Amulet",
                        "change": 22.0,
                    }
                ],
            }
        },
    )

    result = battle_advice_cache_service.predict_team_sets_from_cache(
        ["Garchomp", "Sneasler"]
    )

    garchomp = result["predictions"]["Garchomp"]
    assert garchomp["predictedSet"]["item"] == "Clear Amulet"
    assert garchomp["matchedSampleSize"] == 30
    assert garchomp["likelyMoves"][0]["name"] == "Earthquake"
    assert any("Sneasler" in explanation for explanation in garchomp["explanations"])


def test_predict_team_sets_from_cache_falls_back_to_live_when_summary_missing(monkeypatch):
    monkeypatch.setattr(
        battle_advice_cache_service,
        "load_set_probability_summary_cache",
        lambda: {},
    )
    monkeypatch.setattr(
        battle_advice_cache_service,
        "predict_team_sets",
        lambda team: {"team": team, "predictions": {}, "teamExplanations": [], "dataQualityNotes": []},
    )

    result = battle_advice_cache_service.predict_team_sets_from_cache(["Garchomp"])

    assert result["team"] == ["Garchomp"]
