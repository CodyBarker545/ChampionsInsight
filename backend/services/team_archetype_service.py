"""Prepare tournament team records and feature vectors for archetype clustering."""

from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

from paths import COMPETITIVE_RAW_DIR


TEAMLISTS_PATH = COMPETITIVE_RAW_DIR / "pokemon_teamlists.csv"
TOURNAMENTS_PATH = COMPETITIVE_RAW_DIR / "tournaments.csv"

TEAMLIST_REQUIRED_COLUMNS = ["tournament_id", "player_id", "pokemon"]
TEAMLIST_OPTIONAL_COLUMNS = [
    "player_name",
    "tournament_name",
    "players_in_tournament",
    "placing",
    "wins",
    "losses",
    "ties",
    "team_slot",
    "pokemon",
    "item",
    "ability",
    "move_1",
    "move_2",
    "move_3",
    "move_4",
    "tera_type",
]
TOURNAMENT_OPTIONAL_COLUMNS = [
    "tournament_id",
    "date",
    "format",
    "game",
    "is_online",
    "name",
]
MOVE_COLUMNS = ["move_1", "move_2", "move_3", "move_4"]
METADATA_NUMERIC_COLUMNS = [
    "wins",
    "losses",
    "ties",
    "placing",
    "players_in_tournament",
    "rounds_played",
    "player_scale_weight",
    "round_scale_weight",
    "outcome_weight",
    "team_size",
    "placement_percentile",
    "event_date_ordinal",
]


def normalize_entity_name(value: Any) -> str:
    """Normalize names for Pokemon, items, moves, and abilities."""
    text = str(value or "").strip().lower()
    if not text or text in {"none", "nan", "null"}:
        return ""

    text = text.replace("’", "'")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _ensure_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame


def _safe_read_csv(path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required tournament data file: {path}")
    return pd.read_csv(path, usecols=lambda column: column in columns)


def load_tournament_team_rows(
    teamlists_path=TEAMLISTS_PATH,
    tournaments_path=TOURNAMENTS_PATH,
) -> pd.DataFrame:
    """Load tournament team rows and join available tournament metadata."""
    teamlists = _safe_read_csv(teamlists_path, TEAMLIST_REQUIRED_COLUMNS + TEAMLIST_OPTIONAL_COLUMNS)
    missing_required = [column for column in TEAMLIST_REQUIRED_COLUMNS if column not in teamlists.columns]
    if missing_required:
        raise ValueError(f"Teamlist data is missing required columns: {missing_required}")

    teamlists = _ensure_columns(teamlists, TEAMLIST_OPTIONAL_COLUMNS)

    tournaments = _safe_read_csv(tournaments_path, TOURNAMENT_OPTIONAL_COLUMNS)
    tournaments = _ensure_columns(tournaments, TOURNAMENT_OPTIONAL_COLUMNS)
    tournaments = tournaments.rename(
        columns={
            "name": "tournament_name_meta",
            "date": "tournament_date",
            "format": "tournament_format",
            "game": "tournament_game",
            "is_online": "tournament_is_online",
        }
    )

    merged = teamlists.merge(tournaments, on="tournament_id", how="left")
    if "tournament_name" in merged.columns:
        merged["tournament_name"] = merged["tournament_name"].fillna(merged["tournament_name_meta"])
    else:
        merged["tournament_name"] = merged["tournament_name_meta"]

    return merged


def _normalize_list(values: list[Any]) -> list[str]:
    normalized = [normalize_entity_name(value) for value in values]
    return sorted({value for value in normalized if value})


def _coerce_numeric(value: Any) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    number = _coerce_numeric(value)
    if number is None:
        return None
    return int(number)


def _coerce_bool(value: Any) -> bool | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _estimate_rounds_played(wins: int | None, losses: int | None, ties: int | None) -> int:
    return max((wins or 0) + (losses or 0) + (ties or 0), 0)


def _player_scale_weight(players_in_tournament: int | None) -> float:
    if not players_in_tournament or players_in_tournament < 2:
        return 1.0
    return max(math.log2(players_in_tournament), 1.0)


def _round_scale_weight(rounds_played: int) -> float:
    if rounds_played < 1:
        return 1.0
    return float(rounds_played)


def _outcome_weight(players_in_tournament: int | None, rounds_played: int) -> float:
    return _player_scale_weight(players_in_tournament) * _round_scale_weight(rounds_played)


def _team_record_from_group(team_key: tuple[Any, Any], group: pd.DataFrame) -> dict[str, Any] | None:
    first_row = group.iloc[0]
    pokemon = _normalize_list(group["pokemon"].tolist()) if "pokemon" in group else []
    if not pokemon:
        return None

    moves: list[Any] = []
    for column in MOVE_COLUMNS:
        if column in group.columns:
            moves.extend(group[column].tolist())

    items = _normalize_list(group["item"].tolist()) if "item" in group else []
    abilities = _normalize_list(group["ability"].tolist()) if "ability" in group else []
    tera_types = _normalize_list(group["tera_type"].tolist()) if "tera_type" in group else []

    players_in_tournament = _coerce_int(first_row.get("players_in_tournament"))
    placing = _coerce_int(first_row.get("placing"))
    wins = _coerce_int(first_row.get("wins"))
    losses = _coerce_int(first_row.get("losses"))
    ties = _coerce_int(first_row.get("ties"))
    rounds_played = _estimate_rounds_played(wins, losses, ties)
    player_scale_weight = _player_scale_weight(players_in_tournament)
    round_scale_weight = _round_scale_weight(rounds_played)
    outcome_weight = _outcome_weight(players_in_tournament, rounds_played)
    event_date = pd.to_datetime(first_row.get("tournament_date"), errors="coerce", utc=True)
    if pd.isna(event_date):
        event_date_iso = None
        event_date_ordinal = None
    else:
        event_date_iso = event_date.isoformat()
        event_date_ordinal = int(event_date.normalize().timestamp() // 86400)

    placement_percentile = None
    if placing and players_in_tournament and players_in_tournament > 0:
        placement_percentile = placing / players_in_tournament

    return {
        "team_id": f"{team_key[0]}::{team_key[1]}",
        "tournament_id": str(team_key[0] or ""),
        "player_id": str(team_key[1] or ""),
        "player_name": first_row.get("player_name"),
        "tournament_name": first_row.get("tournament_name"),
        "format": normalize_entity_name(first_row.get("tournament_format")),
        "date": event_date_iso,
        "is_online": _coerce_bool(first_row.get("tournament_is_online")),
        "players_in_tournament": players_in_tournament,
        "placing": placing,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "rounds_played": rounds_played,
        "player_scale_weight": player_scale_weight,
        "round_scale_weight": round_scale_weight,
        "outcome_weight": outcome_weight,
        "team_size": len(pokemon),
        "placement_percentile": placement_percentile,
        "event_date_ordinal": event_date_ordinal,
        "pokemon": pokemon,
        "moves": _normalize_list(moves),
        "items": items,
        "abilities": abilities,
        "tera_types": tera_types,
    }


def create_team_records(team_rows: pd.DataFrame) -> list[dict[str, Any]]:
    """Aggregate one record per tournament team."""
    if team_rows.empty:
        return []

    team_rows = team_rows.copy()
    for required in TEAMLIST_REQUIRED_COLUMNS:
        if required not in team_rows.columns:
            team_rows[required] = pd.NA

    team_rows = team_rows.dropna(subset=["tournament_id", "player_id"])
    records: list[dict[str, Any]] = []
    grouped = team_rows.groupby(["tournament_id", "player_id"], dropna=True, sort=False)
    for team_key, group in grouped:
        record = _team_record_from_group(team_key, group)
        if record is not None:
            records.append(record)
    return records


def _multi_hot_frame(values: list[list[str]], prefix: str) -> pd.DataFrame:
    binarizer = MultiLabelBinarizer()
    matrix = binarizer.fit_transform(values)
    if len(binarizer.classes_) == 0:
        return pd.DataFrame(index=range(len(values)))
    columns = [f"{prefix}:{name}" for name in binarizer.classes_]
    return pd.DataFrame(matrix, columns=columns)


def create_team_feature_matrix(team_records: list[dict[str, Any]]) -> pd.DataFrame:
    """Create a machine-learning-friendly feature matrix from team records."""
    if not team_records:
        return pd.DataFrame()

    records_frame = pd.DataFrame(team_records)

    pokemon_features = _multi_hot_frame(records_frame["pokemon"].tolist(), "pokemon")
    move_features = _multi_hot_frame(records_frame["moves"].tolist(), "move")
    item_features = _multi_hot_frame(records_frame["items"].tolist(), "item")
    ability_features = _multi_hot_frame(records_frame["abilities"].tolist(), "ability")

    format_dummies = pd.get_dummies(
        records_frame.get("format", pd.Series([""] * len(records_frame))).fillna(""),
        prefix="format",
    )

    numeric_frame = pd.DataFrame(index=records_frame.index)
    for column in METADATA_NUMERIC_COLUMNS:
        if column in records_frame.columns:
            numeric_frame[column] = pd.to_numeric(records_frame[column], errors="coerce").fillna(0)
        else:
            numeric_frame[column] = 0

    feature_matrix = pd.concat(
        [
            numeric_frame,
            format_dummies,
            pokemon_features,
            move_features,
            item_features,
            ability_features,
        ],
        axis=1,
    )
    return feature_matrix.astype("float32")


def load_team_archetype_features() -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Load raw team records and their feature matrix."""
    team_rows = load_tournament_team_rows()
    team_records = create_team_records(team_rows)
    feature_matrix = create_team_feature_matrix(team_records)
    return team_records, feature_matrix
