from pathlib import Path

from scripts.data_build.tournament_scraper import (
    load_latest_tournament_timestamp,
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
        latest_existing_timestamp="2026-05-15T10:00:00.000Z",
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
        latest_existing_timestamp=None,
        existing_ids={"already-there"},
    )

    assert [tournament["id"] for tournament in selected] == ["new-1"]
    assert stopped is False


def test_select_new_tournaments_only_keeps_regulation_m_formats():
    tournaments = [
        {"id": "m-a", "date": "2026-05-17T10:00:00.000Z", "format": "M-A"},
        {"id": "m-b", "date": "2026-05-17T09:00:00.000Z", "format": "M-B"},
        {"id": "m-c", "date": "2026-10-19T09:00:00.000Z", "format": "M-C"},
        {"id": "reg-i", "date": "2026-05-17T08:00:00.000Z", "format": "I"},
    ]

    selected, stopped = select_new_tournaments(
        tournaments=tournaments,
        latest_existing_timestamp=None,
        existing_ids=set(),
    )

    assert [tournament["id"] for tournament in selected] == ["m-a", "m-b", "m-c"]
    assert stopped is False


def test_pairing_result_handles_wins_losses_and_byes():
    assert pairing_result("alice", "alice") == "win"
    assert pairing_result("alice", "bob") == "loss"
    assert pairing_result("alice", -1) == "loss"
    assert pairing_result("alice", 0) == "tie"
    assert pairing_result(None, "alice") == "tie/other"


def test_load_latest_timestamp_only_counts_allowed_formats():
    assert load_latest_tournament_timestamp(
        Path("backend/tests/fixtures/tournaments_mixed_formats.csv")
    ) == "2026-05-17T10:00:00.000Z"


def test_pagination_uses_limitless_page_parameter(monkeypatch):
    from scripts.data_build import tournament_scraper as scraper
    calls = []
    monkeypatch.setattr(scraper, 'api_get', lambda path, params: calls.append(params) or [])
    scraper.fetch_tournament_page(limit=100, offset=0)
    scraper.fetch_tournament_page(limit=100, offset=100)
    assert [call['page'] for call in calls] == [1, 2]
    assert all('offset' not in call for call in calls)


def test_nested_standing_records_preserve_zero_wins_and_drop_round():
    from scripts.data_build.tournament_scraper import standing_rows, teamlist_rows
    record = {'player': 'alice', 'placing': 2, 'drop': 3,
              'record': {'wins': 0, 'losses': 3, 'ties': 1},
              'decklist': [{'name': 'Raichu', 'attacks': ['Protect']}]}
    tournament = {'id': 'event', 'name': 'Example', 'players': 8}
    standing = standing_rows([record], tournament)[0]
    pokemon = teamlist_rows([record], tournament)[0]
    assert (standing['wins'], standing['losses'], standing['ties'], standing['drop_round']) == (0, 3, 1, 3)
    assert (pokemon['wins'], pokemon['losses'], pokemon['ties']) == (0, 3, 1)


def test_repair_replaces_affected_rows_and_is_resumable(monkeypatch, upload_dir):
    import csv
    import json
    from scripts.data_build import tournament_scraper as scraper
    raw = upload_dir / 'raw'
    raw.mkdir()
    scraper.append_rows(raw / 'tournaments.csv', scraper.TOURNAMENT_FIELDS,
                        [{'tournament_id': 'event', 'date': '2026-10-19', 'players': 8, 'name': 'Example'}])
    scraper.append_rows(raw / 'standings.csv', scraper.STANDINGS_FIELDS,
                        [{'tournament_id': 'event', 'player_id': 'alice'}])
    scraper.append_rows(raw / 'pokemon_teamlists.csv', scraper.TEAMLIST_FIELDS,
                        [{'tournament_id': 'event', 'player_id': 'alice', 'pokemon': 'Raichu'}])
    scraper.append_rows(raw / 'matches.csv', scraper.MATCH_FIELDS,
                        [{'tournament_id': 'event', 'raw': json.dumps({'player1': 'alice', 'player2': 'bob', 'winner': -1})}])
    calls = []
    def standings(identifier):
        calls.append(identifier)
        return [{'player': 'alice', 'record': {'wins': 2, 'losses': 1, 'ties': 0},
                 'decklist': [{'name': 'Raichu', 'attacks': ['Protect']}]}]
    monkeypatch.setattr(scraper, 'fetch_tournament_standings', standings)
    assert scraper.repair_standings(raw, workers=1) == 1
    for filename in ['standings.csv', 'pokemon_teamlists.csv']:
        with (raw / filename).open(encoding='utf-8', newline='') as file:
            rows = list(csv.DictReader(file))
        assert len(rows) == 1
        assert (rows[0]['wins'], rows[0]['losses'], rows[0]['ties']) == ('2', '1', '0')
    with (raw / 'matches.csv').open(encoding='utf-8', newline='') as file:
        match = next(csv.DictReader(file))
    assert match['player1_result'] == match['player2_result'] == 'loss'
    assert scraper.repair_standings(raw, workers=1) == 0
    assert calls == ['event']


def test_incremental_import_is_oldest_first_and_excludes_future_dates(monkeypatch, upload_dir):
    from scripts.data_build import tournament_scraper as scraper
    scraper.append_rows(upload_dir / 'tournaments.csv', scraper.TOURNAMENT_FIELDS,
                        [{'date': '2026-05-17T10:00:00.000Z', 'format': 'M-A', 'tournament_id': 'saved'}])
    page = [
        {'id': 'future', 'date': '2026-10-20T10:00:00.000Z', 'format': 'M-B'},
        {'id': 'today', 'date': '2026-10-19T10:00:00.000Z', 'format': 'M-B'},
        {'id': 'older-new', 'date': '2026-05-18T10:00:00.000Z', 'format': 'M-A'},
        {'id': 'old', 'date': '2026-05-16T10:00:00.000Z', 'format': 'M-A'},
    ]
    fetched = []
    monkeypatch.setattr(scraper, 'fetch_tournament_page', lambda **kwargs: page)
    monkeypatch.setattr(scraper, 'fetch_tournament_details', lambda identifier: fetched.append(identifier) or {})
    monkeypatch.setattr(scraper, 'fetch_tournament_standings', lambda identifier: [])
    monkeypatch.setattr(scraper, 'fetch_tournament_pairings', lambda identifier: [])
    assert scraper.scrape_incremental(upload_dir, sleep_seconds=0, through_date='2026-10-19', workers=1) == 2
    assert fetched == ['older-new', 'today']
