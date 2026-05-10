from services.tournament_usage_service import (
    get_pokemon_usage_summary,
    load_usage_summary_data,
)


def test_load_usage_summary_data_returns_dictionary():
    data = load_usage_summary_data()

    assert isinstance(data, dict)
    assert len(data) > 0


def test_usage_summary_entries_have_frontend_shape():
    data = load_usage_summary_data()

    first_name = next(iter(data))
    summary = data[first_name]

    assert "appearances" in summary
    assert "wins" in summary
    assert "losses" in summary
    assert "games" in summary
    assert "winRate" in summary
    assert "topMoves" in summary
    assert "topItems" in summary
    assert "topTeams" in summary


def test_get_pokemon_usage_summary_returns_known_pokemon():
    data = load_usage_summary_data()

    first_name = next(iter(data))
    summary = get_pokemon_usage_summary(first_name)

    assert summary["appearances"] > 0
    assert isinstance(summary["topMoves"], list)
    assert isinstance(summary["topItems"], list)
    assert isinstance(summary["topTeams"], list)


def test_get_pokemon_usage_summary_handles_missing_pokemon():
    summary = get_pokemon_usage_summary("NotARealPokemonName123")

    assert summary["appearances"] == 0
    assert summary["wins"] == 0
    assert summary["losses"] == 0
    assert summary["games"] == 0
    assert summary["winRate"] == 0
    assert summary["topMoves"] == []
    assert summary["topItems"] == []
    assert summary["topTeams"] == []