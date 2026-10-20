"""Rebuild usage, common moves and common items from the saved teamlists."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
from paths import COMPETITIVE_RAW_DIR, COMPETITIVE_ANALYSIS_DIR


def build_usage(rows):
    teams = defaultdict(list)
    members = defaultdict(list)
    for row in rows:
        if row.get('pokemon'):
            teams[(row['tournament_id'], row['player_id'])].append(row)
            members[row['pokemon']].append(row)

    def number(value):
        try:
            return int(float(value or 0))
        except (ValueError, TypeError):
            return 0

    summary = {}
    for name, appearances in sorted(members.items()):
        moves = Counter(move for row in appearances for i in range(1, 5)
                        if (move := row.get(f'move_{i}', '').strip()))
        items = Counter(row['item'] for row in appearances if row.get('item'))
        wins = sum(number(row.get('wins')) for row in appearances)
        losses = sum(number(row.get('losses')) for row in appearances)
        top_teams = []
        seen = set()
        for row in sorted(appearances, key=lambda row: (
            -number(row.get('wins')), number(row.get('losses')),
            number(row.get('placing')) or 99999)):
            key = (row['tournament_id'], row['player_id'])
            if key in seen:
                continue
            seen.add(key)
            team = sorted(teams[key], key=lambda member: number(member.get('team_slot')))
            top_teams.append({'player': row.get('player_name', ''),
                              'tournament': row.get('tournament_name', ''),
                              'wins': number(row.get('wins')), 'losses': number(row.get('losses')),
                              'pokemon': [member['pokemon'] for member in team]})
            if len(top_teams) == 5:
                break
        summary[name] = {
            'appearances': len(appearances), 'wins': wins, 'losses': losses,
            'games': wins + losses, 'winRate': round(wins / (wins + losses), 3) if wins + losses else 0,
            'topMoves': [{'move': move, 'count': count} for move, count in moves.most_common(8)],
            'topItems': [{'item': item, 'count': count} for item, count in items.most_common(5)],
            'topTeams': top_teams,
        }
    return summary


def main():
    with (COMPETITIVE_RAW_DIR / 'pokemon_teamlists.csv').open(newline='', encoding='utf-8') as file:
        summary = build_usage(csv.DictReader(file))
    outputs = {
        'pokedex_usage_summary.json': summary,
        'pokemon_top_6_moves.json': {name: row['topMoves'][:6] for name, row in summary.items()},
        'pokemon_top_3_items.json': {name: row['topItems'][:3] for name, row in summary.items()},
    }
    for filename, payload in outputs.items():
        (COMPETITIVE_ANALYSIS_DIR / filename).write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(f'Rebuilt usage, moves and items for {len(summary)} Pokemon.')


if __name__ == '__main__':
    main()
