from scripts.data_build.tournament_scraper import (
    pairing_result,
    select_new_tournaments,
)


def test_select_new_tournaments_stops_on_first_existing_date():
    tournaments = [
        {"id": "new-1", "date": "2026-05-17T10:00:00.000Z", "format": "M-A"},
        {"id": "new-2", "date": "2026-05-16T10:00:00.000Z", "format": "M-B"},
        {"id": "old-1", "date": "2026-05-15T10:00:00.000Z", "format": "M-A"},
        {"id": "older", "date": "2026-05-14T10:00:00.000Z", "format": "M-A"},
    ]

    selected, stopped = select_new_tournaments(
        tournaments=tournaments,
        existing_dates={"2026-05-15"},
        existing_ids={"old-1"},
    )

    assert [tournament["id"] for tournament in selected] == ["new-1", "new-2"]
    assert stopped is True


def test_select_new_tournaments_skips_duplicate_ids_before_stop_date():
    tournaments = [
        {"id": "already-there", "date": "2026-05-17T10:00:00.000Z", "format": "M-A"},
        {"id": "new-1", "date": "2026-05-17T09:00:00.000Z", "format": "M-B"},
    ]

    selected, stopped = select_new_tournaments(
        tournaments=tournaments,
        existing_dates=set(),
        existing_ids={"already-there"},
    )

    assert [tournament["id"] for tournament in selected] == ["new-1"]
    assert stopped is False


def test_select_new_tournaments_only_keeps_regulation_m_formats():
    tournaments = [
        {"id": "m-a", "date": "2026-05-17T10:00:00.000Z", "format": "M-A"},
        {"id": "m-b", "date": "2026-05-17T09:00:00.000Z", "format": "M-B"},
        {"id": "reg-i", "date": "2026-05-17T08:00:00.000Z", "format": "I"},
    ]

    selected, stopped = select_new_tournaments(
        tournaments=tournaments,
        existing_dates=set(),
        existing_ids=set(),
    )

    assert [tournament["id"] for tournament in selected] == ["m-a", "m-b"]
    assert stopped is False


def test_pairing_result_handles_wins_losses_and_byes():
    assert pairing_result("alice", "alice") == "win"
    assert pairing_result("alice", "bob") == "loss"
    assert pairing_result("alice", -1) == "tie/other"
    assert pairing_result(None, "alice") == "tie/other"
