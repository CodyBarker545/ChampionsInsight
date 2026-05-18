"""Cluster tournament teams into archetype groups and write readable summaries."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from scipy import sparse

from paths import COMPETITIVE_ANALYSIS_DIR
from services.team_archetype_service import load_team_archetype_features


TEAM_CLUSTER_ASSIGNMENTS_PATH = COMPETITIVE_ANALYSIS_DIR / "team_archetype_cluster_assignments.csv"
TEAM_CLUSTER_SUMMARY_PATH = COMPETITIVE_ANALYSIS_DIR / "team_archetype_cluster_summary.json"
FEATURE_PREFIXES = ("pokemon:", "move:", "item:", "ability:", "format_")
METADATA_COLUMNS = [
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


def _split_feature_columns(feature_matrix: pd.DataFrame) -> tuple[list[str], list[str]]:
    categorical_columns = [
        column
        for column in feature_matrix.columns
        if column.startswith(FEATURE_PREFIXES)
    ]
    numeric_columns = [
        column
        for column in feature_matrix.columns
        if column in METADATA_COLUMNS
    ]
    return categorical_columns, numeric_columns


def _build_clustering_matrix(feature_matrix: pd.DataFrame) -> np.ndarray:
    categorical_columns, numeric_columns = _split_feature_columns(feature_matrix)

    if not categorical_columns:
        raise ValueError("No categorical feature columns were found for clustering.")

    categorical_matrix = sparse.csr_matrix(
        feature_matrix[categorical_columns].astype(np.float32).to_numpy()
    )
    categorical_matrix = TfidfTransformer().fit_transform(categorical_matrix)

    blocks: list[sparse.csr_matrix] = [categorical_matrix]
    if numeric_columns:
        numeric_matrix = feature_matrix[numeric_columns].astype(np.float32).to_numpy()
        numeric_matrix = StandardScaler().fit_transform(numeric_matrix)
        numeric_matrix = numeric_matrix * 0.35
        blocks.append(sparse.csr_matrix(numeric_matrix))

    combined = sparse.hstack(blocks).tocsr()
    max_components = min(64, combined.shape[0] - 1, combined.shape[1] - 1)
    if max_components >= 2:
        reduced = TruncatedSVD(n_components=max_components, random_state=42).fit_transform(combined)
        return reduced.astype(np.float32)

    return combined.toarray().astype(np.float32)


def _compressed_sample_weights(team_records: list[dict[str, Any]]) -> np.ndarray:
    raw_weights = np.array(
        [float(record.get("outcome_weight") or 1.0) for record in team_records],
        dtype=np.float32,
    )
    return np.sqrt(np.maximum(raw_weights, 1.0))


def cluster_team_archetypes(
    cluster_count: int = 12,
    random_state: int = 42,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """Assign each team to a cluster using the prepared feature matrix."""
    team_records, feature_matrix = load_team_archetype_features()
    if not team_records:
        return [], pd.DataFrame()

    if feature_matrix.empty:
        raise ValueError("Feature matrix is empty. Cannot cluster teams.")

    effective_clusters = max(1, min(cluster_count, len(team_records)))
    clustering_matrix = _build_clustering_matrix(feature_matrix)
    sample_weights = _compressed_sample_weights(team_records)

    model = KMeans(
        n_clusters=effective_clusters,
        random_state=random_state,
        n_init=10,
    )
    labels = model.fit_predict(clustering_matrix, sample_weight=sample_weights)

    assignments: list[dict[str, Any]] = []
    for record, cluster_id in zip(team_records, labels):
        assignment = dict(record)
        assignment["cluster_id"] = int(cluster_id)
        assignments.append(assignment)

    assignment_frame = pd.DataFrame(assignments)
    return assignments, assignment_frame


def _top_counter_values(values: list[str], top_n: int) -> list[dict[str, Any]]:
    counter = Counter(value for value in values if value)
    return [
        {"name": name, "count": count}
        for name, count in counter.most_common(top_n)
    ]


def _distinctive_counter_values(
    cluster_values: list[str],
    global_counts: Counter,
    cluster_team_count: int,
    global_team_count: int,
    top_n: int,
    min_count: int = 3,
) -> list[dict[str, Any]]:
    cluster_counts = Counter(value for value in cluster_values if value)
    scored: list[dict[str, Any]] = []
    for name, count in cluster_counts.items():
        if count < min_count:
            continue
        cluster_rate = count / max(cluster_team_count, 1)
        global_rate = global_counts.get(name, 0) / max(global_team_count, 1)
        if global_rate <= 0:
            lift = cluster_rate
        else:
            lift = cluster_rate / global_rate
        scored.append(
            {
                "name": name,
                "count": count,
                "cluster_rate": cluster_rate,
                "global_rate": global_rate,
                "lift": lift,
                "score": lift * count,
            }
        )

    scored.sort(key=lambda item: (item["score"], item["lift"], item["count"]), reverse=True)
    return scored[:top_n]


def _flatten_list_column(frame: pd.DataFrame, column: str) -> list[str]:
    values: list[str] = []
    if column not in frame.columns:
        return values
    for items in frame[column]:
        if isinstance(items, list):
            values.extend(item for item in items if item)
    return values


def summarize_team_clusters(
    assignments: list[dict[str, Any]],
    top_n: int = 12,
    sample_teams_per_cluster: int = 5,
) -> dict[str, Any]:
    """Build a readable summary for each cluster to support manual labeling."""
    if not assignments:
        return {"clusterCount": 0, "clusters": []}

    assignment_frame = pd.DataFrame(assignments)
    global_pokemon_counts = Counter(_flatten_list_column(assignment_frame, "pokemon"))
    global_move_counts = Counter(_flatten_list_column(assignment_frame, "moves"))
    global_item_counts = Counter(_flatten_list_column(assignment_frame, "items"))
    global_ability_counts = Counter(_flatten_list_column(assignment_frame, "abilities"))
    global_team_count = len(assignment_frame)
    clusters: list[dict[str, Any]] = []

    for cluster_id in sorted(assignment_frame["cluster_id"].unique()):
        cluster_frame = assignment_frame[assignment_frame["cluster_id"] == cluster_id].copy()
        cluster_frame = cluster_frame.sort_values(
            by=["outcome_weight", "wins", "players_in_tournament"],
            ascending=[False, False, False],
            na_position="last",
        )

        weighted_size = float(cluster_frame["outcome_weight"].fillna(0).sum())
        cluster_team_count = len(cluster_frame)
        cluster_pokemon_values = _flatten_list_column(cluster_frame, "pokemon")
        cluster_move_values = _flatten_list_column(cluster_frame, "moves")
        cluster_item_values = _flatten_list_column(cluster_frame, "items")
        cluster_ability_values = _flatten_list_column(cluster_frame, "abilities")
        representative_teams = []
        for _, row in cluster_frame.head(sample_teams_per_cluster).iterrows():
            representative_teams.append(
                {
                    "team_id": row.get("team_id"),
                    "player_name": row.get("player_name"),
                    "tournament_name": row.get("tournament_name"),
                    "format": row.get("format"),
                    "placing": row.get("placing"),
                    "wins": row.get("wins"),
                    "losses": row.get("losses"),
                    "outcome_weight": row.get("outcome_weight"),
                    "pokemon": row.get("pokemon"),
                    "moves": row.get("moves"),
                    "items": row.get("items"),
                    "abilities": row.get("abilities"),
                }
            )

        clusters.append(
            {
                "cluster_id": int(cluster_id),
                "team_count": int(len(cluster_frame)),
                "weighted_team_count": weighted_size,
                "avg_players_in_tournament": float(cluster_frame["players_in_tournament"].fillna(0).mean()),
                "avg_rounds_played": float(cluster_frame["rounds_played"].fillna(0).mean()),
                "avg_wins": float(cluster_frame["wins"].fillna(0).mean()),
                "avg_losses": float(cluster_frame["losses"].fillna(0).mean()),
                "avg_placing": float(cluster_frame["placing"].fillna(0).mean()),
                "top_pokemon": _top_counter_values(cluster_pokemon_values, top_n),
                "top_moves": _top_counter_values(cluster_move_values, top_n),
                "top_items": _top_counter_values(cluster_item_values, top_n),
                "top_abilities": _top_counter_values(cluster_ability_values, top_n),
                "distinctive_pokemon": _distinctive_counter_values(
                    cluster_pokemon_values,
                    global_pokemon_counts,
                    cluster_team_count,
                    global_team_count,
                    top_n,
                ),
                "distinctive_moves": _distinctive_counter_values(
                    cluster_move_values,
                    global_move_counts,
                    cluster_team_count,
                    global_team_count,
                    top_n,
                ),
                "distinctive_items": _distinctive_counter_values(
                    cluster_item_values,
                    global_item_counts,
                    cluster_team_count,
                    global_team_count,
                    top_n,
                ),
                "distinctive_abilities": _distinctive_counter_values(
                    cluster_ability_values,
                    global_ability_counts,
                    cluster_team_count,
                    global_team_count,
                    top_n,
                ),
                "sample_teams": representative_teams,
            }
        )

    return {
        "clusterCount": len(clusters),
        "clusters": clusters,
    }


def save_team_cluster_outputs(
    assignments: list[dict[str, Any]],
    summary: dict[str, Any],
    assignments_path: Path = TEAM_CLUSTER_ASSIGNMENTS_PATH,
    summary_path: Path = TEAM_CLUSTER_SUMMARY_PATH,
) -> None:
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(assignments).to_csv(assignments_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
