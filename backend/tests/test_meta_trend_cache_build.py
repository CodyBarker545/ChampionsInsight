"""Batch meta caching must preserve individual computation results."""
from services import meta_trend_service as meta


def test_batch_context_matches_individual_meta_results(monkeypatch):
    teams = [
        {'team_id': str(i), 'date': date, 'pokemon': ['raichu', 'politoed', 'venusaur'],
         'wins': i % 3, 'losses': 1}
        for i, date in enumerate(['2026-07-01', '2026-08-15', '2026-09-01', '2026-09-20', '2026-10-01', '2026-10-19'])
    ]
    records = [
        {'team_id': team['team_id'], 'pokemon': name, 'tournament_date': team['date'],
         'moves': ['protect'], 'item': 'sitrus berry', 'archetype': 'Rain', 'wins': 2, 'losses': 1}
        for team in teams for name in team['pokemon']
    ]
    monkeypatch.setattr(meta, 'load_meta_team_records', lambda: teams)
    monkeypatch.setattr(meta, 'load_pokemon_set_records', lambda: records)
    monkeypatch.setattr(meta, 'get_pokedex_entry_by_name', lambda name: {'name': name, 'speciesName': name})
    monkeypatch.setattr(meta, 'get_pokedex_grid_entries', lambda: [
        {'speciesName': name} for name in ['Raichu', 'Politoed', 'Missing', 'Raichu']
    ])
    expected = {name: meta._compute_pokemon_meta_intelligence(name) for name in ['Raichu', 'Politoed', 'Missing']}
    assert meta.build_pokedex_meta_intelligence_cache() == expected


def test_repeated_event_timestamp_is_parsed_once(monkeypatch):
    meta._parse_timestamp_value.cache_clear()
    original = meta.pd.to_datetime
    calls = []
    def parse(value, **kwargs):
        calls.append(value)
        return original(value, **kwargs)
    monkeypatch.setattr(meta.pd, 'to_datetime', parse)
    first = meta._parse_timestamp({'date': '2026-10-19T01:00:00Z'})
    second = meta._parse_timestamp({'tournament_date': '2026-10-19T01:00:00Z'})
    assert first == second
    assert len(calls) == 1
    meta._parse_timestamp_value.cache_clear()
