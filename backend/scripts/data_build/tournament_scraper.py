"""Incrementally download newer Limitless VGC tournament results into CSV files.

The scraper reads the already-saved tournament dates, walks the Limitless
tournament list newest-first, and stops as soon as it reaches a date that has
already been ingested. That keeps repeat runs cheap while preserving the raw
CSV contract used by the rest of the app.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from paths import COMPETITIVE_RAW_DIR


API_ROOT = "https://play.limitlesstcg.com/api"
GAME = "VGC"
ALLOWED_FORMATS = {"M-A", "M-B"}

TOURNAMENT_FIELDS = [
    "date",
    "decklists",
    "format",
    "game",
    "is_online",
    "name",
    "organizer",
    "players",
    "tournament_id",
    "url",
]

STANDINGS_FIELDS = [
    "country",
    "drop_round",
    "is_winner",
    "losses",
    "placing",
    "player_id",
    "player_name",
    "players_in_tournament",
    "ties",
    "tournament_id",
    "tournament_name",
    "weighted_score",
    "wins",
]

TEAMLIST_FIELDS = [
    "ability",
    "item",
    "losses",
    "move_1",
    "move_2",
    "move_3",
    "move_4",
    "placing",
    "player_id",
    "player_name",
    "players_in_tournament",
    "pokemon",
    "raw",
    "team_slot",
    "tera_type",
    "ties",
    "tournament_id",
    "tournament_name",
    "wins",
]

MATCH_FIELDS = [
    "match",
    "phase",
    "player1",
    "player1_result",
    "player2",
    "player2_result",
    "raw",
    "round",
    "table",
    "tournament_id",
    "tournament_name",
    "winner",
]


def parse_iso_date(value: str | None) -> str:
    """Return YYYY-MM-DD from an API ISO timestamp."""
    if not value:
        return ""
    return str(value)[:10]


def load_latest_tournament_timestamp(tournaments_path: Path) -> str | None:
    if not tournaments_path.exists():
        return None

    with tournaments_path.open("r", newline="", encoding="utf-8") as csv_file:
        timestamps = [
            row.get("date", "")
            for row in csv.DictReader(csv_file)
            if row.get("date") and row.get("format") in ALLOWED_FORMATS
        ]
    return max(timestamps, default=None)


def load_existing_tournament_ids(tournaments_path: Path) -> set[str]:
    if not tournaments_path.exists():
        return set()

    with tournaments_path.open("r", newline="", encoding="utf-8") as csv_file:
        return {
            row["tournament_id"]
            for row in csv.DictReader(csv_file)
            if row.get("tournament_id")
        }


def select_new_tournaments(
    tournaments: Iterable[dict[str, Any]],
    latest_existing_timestamp: str | None,
    existing_ids: set[str],
) -> tuple[list[dict[str, Any]], bool]:
    """Return unseen tournaments until we move older than the newest saved row."""
    selected: list[dict[str, Any]] = []
    stopped_on_existing_date = False

    for tournament in tournaments:
        tournament_timestamp = str(tournament.get("date") or "")
        tournament_id = str(tournament.get("id") or "")

        if latest_existing_timestamp and tournament_timestamp < latest_existing_timestamp:
            stopped_on_existing_date = True
            break

        if (
            tournament_id
            and tournament_id not in existing_ids
            and tournament.get("format") in ALLOWED_FORMATS
        ):
            selected.append(tournament)

    return selected, stopped_on_existing_date


def api_get(
    path: str,
    params: dict[str, Any] | None = None,
    max_retries: int = 5,
) -> Any:
    query = f"?{urlencode(params)}" if params else ""
    request = Request(
        f"{API_ROOT}{path}{query}",
        headers={"User-Agent": "ChampionsInsight tournament scraper"},
    )

    for attempt in range(max_retries + 1):
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            if error.code != 429 or attempt == max_retries:
                raise

            retry_after = error.headers.get("Retry-After")
            try:
                wait_seconds = float(retry_after) if retry_after else 2 ** attempt
            except ValueError:
                wait_seconds = 2 ** attempt

            wait_seconds = max(wait_seconds, 1.0)
            print(
                f"Rate limited on {path}; waiting {wait_seconds:.1f}s before retry "
                f"{attempt + 1}/{max_retries}.",
                flush=True,
            )
            time.sleep(wait_seconds)

    raise RuntimeError(f"Failed to fetch {path}")


def fetch_tournament_page(limit: int, offset: int) -> list[dict[str, Any]]:
    return api_get("/tournaments", {"game": GAME, "limit": limit, "offset": offset})


def fetch_tournament_details(tournament_id: str) -> dict[str, Any]:
    return api_get(f"/tournaments/{tournament_id}/details")


def fetch_tournament_standings(tournament_id: str) -> list[dict[str, Any]]:
    return api_get(f"/tournaments/{tournament_id}/standings")


def fetch_tournament_pairings(tournament_id: str) -> list[dict[str, Any]]:
    return api_get(f"/tournaments/{tournament_id}/pairings")


def append_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def overwrite_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_organizer_name(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("name") or value.get("id") or "")
    if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, dict):
            return str(parsed.get("name") or parsed.get("id") or value)
    if value is None:
        return ""
    return str(value)


def build_tournament_url(tournament: dict[str, Any]) -> str:
    url = tournament.get("url")
    if url:
        return str(url)

    tournament_id = str(tournament.get("id") or tournament.get("tournament_id") or "")
    if tournament_id:
        return f"https://play.limitlesstcg.com/tournament/{tournament_id}"
    return ""


def tournament_row(tournament: dict[str, Any]) -> dict[str, Any]:
    tournament_id = str(tournament.get("id") or "")
    return {
        "date": tournament.get("date"),
        "decklists": tournament.get("decklists"),
        "format": tournament.get("format"),
        "game": tournament.get("game"),
        "is_online": tournament.get("isOnline"),
        "name": tournament.get("name"),
        "organizer": normalize_organizer_name(tournament.get("organizer")),
        "players": tournament.get("players"),
        "tournament_id": tournament_id,
        "url": build_tournament_url(tournament),
    }


def sort_tournament_rows_desc(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (str(row.get("date") or ""), str(row.get("tournament_id") or "")),
        reverse=True,
    )


def normalize_existing_tournament_rows(tournaments_path: Path) -> None:
    if not tournaments_path.exists():
        return

    with tournaments_path.open("r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {
                "date": row.get("date"),
                "decklists": row.get("decklists"),
                "format": row.get("format"),
                "game": row.get("game"),
                "is_online": row.get("is_online"),
                "name": row.get("name"),
                "organizer": normalize_organizer_name(row.get("organizer")),
                "players": row.get("players"),
                "tournament_id": row.get("tournament_id"),
                "url": build_tournament_url(row),
            }
        )

    overwrite_rows(
        tournaments_path,
        TOURNAMENT_FIELDS,
        sort_tournament_rows_desc(normalized_rows),
    )


def standing_rows(
    standings: list[dict[str, Any]],
    tournament: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for standing in standings:
        rows.append(
            {
                "country": standing.get("country"),
                "drop_round": standing.get("dropRound"),
                "is_winner": standing.get("placing") == 1,
                "losses": standing.get("losses"),
                "placing": standing.get("placing"),
                "player_id": standing.get("player"),
                "player_name": standing.get("name"),
                "players_in_tournament": tournament.get("players"),
                "ties": standing.get("ties"),
                "tournament_id": tournament.get("id"),
                "tournament_name": tournament.get("name"),
                "weighted_score": standing.get("score"),
                "wins": standing.get("wins"),
            }
        )
    return rows


def teamlist_rows(
    standings: list[dict[str, Any]],
    tournament: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for standing in standings:
        decklist = standing.get("decklist") or []
        if not isinstance(decklist, list):
            continue

        for slot, pokemon in enumerate(decklist, start=1):
            attacks = pokemon.get("attacks") or []
            rows.append(
                {
                    "ability": pokemon.get("ability"),
                    "item": pokemon.get("item"),
                    "losses": standing.get("losses"),
                    "move_1": attacks[0] if len(attacks) > 0 else None,
                    "move_2": attacks[1] if len(attacks) > 1 else None,
                    "move_3": attacks[2] if len(attacks) > 2 else None,
                    "move_4": attacks[3] if len(attacks) > 3 else None,
                    "placing": standing.get("placing"),
                    "player_id": standing.get("player"),
                    "player_name": standing.get("name"),
                    "players_in_tournament": tournament.get("players"),
                    "pokemon": pokemon.get("name"),
                    "raw": json.dumps(pokemon, ensure_ascii=False),
                    "team_slot": slot,
                    "tera_type": pokemon.get("tera"),
                    "ties": standing.get("ties"),
                    "tournament_id": tournament.get("id"),
                    "tournament_name": tournament.get("name"),
                    "wins": standing.get("wins"),
                }
            )
    return rows


def pairing_result(player_id: str | None, winner: Any) -> str:
    if not player_id or winner in (None, -1, ""):
        return "tie/other"
    return "win" if player_id == winner else "loss"


def match_rows(pairings: list[dict[str, Any]], tournament: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for pairing in pairings:
        winner = pairing.get("winner", -1)
        player1 = pairing.get("player1")
        player2 = pairing.get("player2")
        rows.append(
            {
                "match": pairing.get("match"),
                "phase": pairing.get("phase"),
                "player1": player1,
                "player1_result": pairing_result(player1, winner),
                "player2": player2,
                "player2_result": pairing_result(player2, winner),
                "raw": json.dumps(pairing, ensure_ascii=False),
                "round": pairing.get("round"),
                "table": pairing.get("table"),
                "tournament_id": tournament.get("id"),
                "tournament_name": tournament.get("name"),
                "winner": winner,
            }
        )
    return rows


def update_summary(raw_dir: Path) -> None:
    counts = {}
    for filename, key in [
        ("tournaments.csv", "tournaments_count"),
        ("standings.csv", "standings_count"),
        ("pokemon_teamlists.csv", "pokemon_rows_count"),
        ("matches.csv", "matches_count"),
    ]:
        path = raw_dir / filename
        if not path.exists():
            counts[key] = 0
            continue
        with path.open("r", newline="", encoding="utf-8") as csv_file:
            counts[key] = max(sum(1 for _ in csv_file) - 1, 0)

    (raw_dir / "raw_summary.json").write_text(
        json.dumps(counts, indent=2) + "\n",
        encoding="utf-8",
    )


def scrape_incremental(
    raw_dir: Path = COMPETITIVE_RAW_DIR,
    page_size: int = 100,
    sleep_seconds: float = 1.0,
) -> int:
    tournaments_path = raw_dir / "tournaments.csv"
    latest_existing_timestamp = load_latest_tournament_timestamp(tournaments_path)
    existing_ids = load_existing_tournament_ids(tournaments_path)

    offset = 0
    new_tournament_count = 0

    while True:
        page = fetch_tournament_page(limit=page_size, offset=offset)
        if not page:
            break

        selected, stopped = select_new_tournaments(page, latest_existing_timestamp, existing_ids)
        for listed_tournament in selected:
            tournament_id = str(listed_tournament["id"])
            details = fetch_tournament_details(tournament_id)
            tournament = {**listed_tournament, **details}
            standings = fetch_tournament_standings(tournament_id)
            pairings = fetch_tournament_pairings(tournament_id)

            append_rows(raw_dir / "tournaments.csv", TOURNAMENT_FIELDS, [tournament_row(tournament)])
            append_rows(raw_dir / "standings.csv", STANDINGS_FIELDS, standing_rows(standings, tournament))
            append_rows(raw_dir / "pokemon_teamlists.csv", TEAMLIST_FIELDS, teamlist_rows(standings, tournament))
            append_rows(raw_dir / "matches.csv", MATCH_FIELDS, match_rows(pairings, tournament))

            existing_ids.add(tournament_id)
            new_tournament_count += 1
            if sleep_seconds:
                time.sleep(sleep_seconds)

        if stopped or len(page) < page_size:
            break

        offset += page_size

    normalize_existing_tournament_rows(tournaments_path)
    update_summary(raw_dir)
    return new_tournament_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append only newer Limitless VGC Regulation Set M-A/M-B tournament data, "
            "stopping at known dates."
        )
    )
    parser.add_argument("--page-size", type=int, default=100)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    added = scrape_incremental(page_size=args.page_size, sleep_seconds=args.sleep_seconds)
    print(f"Added {added} new tournament(s).")


if __name__ == "__main__":
    main()
