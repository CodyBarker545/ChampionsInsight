"""Tests mock matchup analysis behavior."""

import pytest

from services.matchup_service import analyze_matchup


# Tests that matchup analysis returns the expected mock shape.
def test_analyze_matchup_returns_mock_analysis():
    result = analyze_matchup({"team": [{"name": "Aegis"}], "opponent": "Volt"})

    assert result["summary"] == "Aegis has a favorable opening into Volt."
    assert result["speed"]["result"] == "Your lead is faster"
    assert result["damage"]["range"] == "42% - 51%"
    assert len(result["recommendations"]) == 3


# Tests that matchup analysis rejects empty teams.
def test_analyze_matchup_requires_team_member():
    with pytest.raises(ValueError, match="At least one team member is required."):
        analyze_matchup({"team": []})
