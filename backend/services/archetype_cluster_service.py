"""Cluster teams inside each rule-based archetype bucket."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler

from paths import ARCHETYPE_CLUSTER_SUMMARY_PATH
from services.archetype_feature_service import extract_team_features, load_tournament_teams, normalize_name
from services.archetype_rule_service import classify_team_archetype


MIN_TEAMS_TO_CLUSTER = 8


def _canonical_display(value: str) -> str:
    return " ".join(part.capitalize() for part in str(value or "").replace("-", " ").split())


def _safe_average(values: list[float]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    if not usable:
        return None
    return round(sum(usable) / len(usable), 3)


def _normalize_sequence(values: list[Any]) -> list[str]:
    return [normalize_name(value) for value in values if normalize_name(value)]


def _choose_cluster_count(team_count: int) -> int:
    if team_count < MIN_TEAMS_TO_CLUSTER:
        return 0
    if team_count < 30:
        return min(2, team_count)
    if team_count <= 100:
        return min(3, team_count)
    if team_count <= 250:
        return min(4, team_count)
    return min(5, team_count)


def _multi_hot_frame(values: list[list[str]], prefix: str) -> pd.DataFrame:
    if not values:
        return pd.DataFrame()
    binarizer = MultiLabelBinarizer()
    matrix = binarizer.fit_transform(values)
    if len(binarizer.classes_) == 0:
        return pd.DataFrame(index=range(len(values)))
    columns = [f"{prefix}:{name}" for name in binarizer.classes_]
    return pd.DataFrame(matrix, columns=columns)


def _numeric_feature_frame(feature_records: list[dict[str, Any]]) -> pd.DataFrame:
    numeric_keys = [
        "average_hp",
        "average_attack",
        "average_defense",
        "average_sp_attack",
        "average_sp_defense",
        "average_speed",
        "average_bulk",
        "physical_attacker_count",
        "special_attacker_count",
        "mixed_attacker_count",
        "fast_pokemon_count",
        "slow_pokemon_count",
        "bulky_pokemon_count",
        "setup_move_count",
        "recovery_move_count",
        "status_move_count",
        "priority_move_count",
        "speed_control_count",
        "trick_room_move_count",
        "tailwind_move_count",
        "weather_move_count",
        "protect_count",
        "pivot_move_count",
        "weather_setter_count",
        "weather_abuser_count",
        "trick_room_setter_count",
        "tailwind_setter_count",
        "focus_sash_count",
        "choice_item_count",
        "defensive_item_count",
        "offensive_item_count",
    ]
    frame = pd.DataFrame([{key: record.get(key, 0) for key in numeric_keys} for record in feature_records])
    return frame.fillna(0.0).astype("float32")


def _build_feature_matrix(group_entries: list[dict[str, Any]]) -> pd.DataFrame:
    teams = [entry["team"] for entry in group_entries]
    features = [entry["features"] for entry in group_entries]

    pokemon_frame = _multi_hot_frame([_normalize_sequence(team.get("pokemon_names", [])) for team in teams], "pokemon")
    move_frame = _multi_hot_frame([_normalize_sequence(team.get("moves", [])) for team in teams], "move")
    item_frame = _multi_hot_frame([_normalize_sequence(team.get("items", [])) for team in teams], "item")
    ability_frame = _multi_hot_frame([_normalize_sequence(team.get("abilities", [])) for team in teams], "ability")
    numeric_frame = _numeric_feature_frame(features)

    matrix = pd.concat(
        [numeric_frame, pokemon_frame, move_frame, item_frame, ability_frame],
        axis=1,
    )
    return matrix.fillna(0.0).astype("float32")


def _top_counter_values(counter: Counter[str], limit: int = 10) -> list[dict[str, Any]]:
    return [
        {"name": _canonical_display(name), "count": count}
        for name, count in counter.most_common(limit)
    ]


def _summarize_offense_shape(cluster_entries: list[dict[str, Any]]) -> str:
    avg_speed = _safe_average([entry["features"].get("average_speed") for entry in cluster_entries]) or 0
    avg_attack = _safe_average([entry["features"].get("average_attack") for entry in cluster_entries]) or 0
    avg_sp_attack = _safe_average([entry["features"].get("average_sp_attack") for entry in cluster_entries]) or 0
    avg_bulk = _safe_average([entry["features"].get("average_bulk") for entry in cluster_entries]) or 0

    offensive_peak = max(avg_attack, avg_sp_attack)
    if avg_speed >= 95 and offensive_peak >= 100:
        return "Hyper Offense"
    if avg_bulk >= 270 and offensive_peak >= 95:
        return "Balance"
    if avg_bulk >= 285:
        return "Bulky"
    if avg_speed <= 75:
        return "Control"
    return "Offense"


def _suggest_cluster_label(parent_archetype: str, cluster_entries: list[dict[str, Any]]) -> str:
    avg_speed = _safe_average([entry["features"].get("average_speed") for entry in cluster_entries]) or 0
    avg_bulk = _safe_average([entry["features"].get("average_bulk") for entry in cluster_entries]) or 0
    avg_attack = _safe_average([entry["features"].get("average_attack") for entry in cluster_entries]) or 0
    avg_sp_attack = _safe_average([entry["features"].get("average_sp_attack") for entry in cluster_entries]) or 0
    average_offense = max(avg_attack, avg_sp_attack)
    trick_room_density = _safe_average([entry["features"].get("trick_room_move_count") for entry in cluster_entries]) or 0
    tailwind_density = _safe_average([entry["features"].get("tailwind_move_count") for entry in cluster_entries]) or 0

    if parent_archetype == "Sand":
        return "Sand Offense" if average_offense >= 100 and avg_speed >= 80 else "Sand Balance"
    if parent_archetype == "Rain":
        return "Rain Offense" if average_offense >= 98 else "Rain Balance"
    if parent_archetype == "Sun":
        return "Sun Offense" if average_offense >= 100 else "Sun Balance"
    if parent_archetype == "Snow":
        return "Snow Offense" if avg_speed >= 80 else "Snow Balance"
    if parent_archetype == "Trick Room":
        return "Hard Trick Room" if avg_speed <= 70 and trick_room_density >= 1 else "Trick Room Balance"
    if parent_archetype == "Semi-Room":
        return "Semi-Room Balance" if avg_bulk >= 265 else "Semi-Room Offense"
    if parent_archetype == "Tailwind Offense":
        return "Tailwind Hyper Offense" if avg_speed >= 95 and average_offense >= 100 else "Tailwind Balance"
    if parent_archetype == "Bulky Setup":
        return "Bulky Setup Balance" if avg_bulk >= 275 else "Bulky Setup Offense"
    if parent_archetype == "Hyper Offense":
        return "Hyper Offense"
    if parent_archetype == "Stall or Control":
        return "Control Balance" if avg_bulk >= 275 else "Control"
    if parent_archetype == "Weather Offense":
        return "Weather Offense"
    if parent_archetype == "Balance":
        if tailwind_density >= 1:
            return "Tailwind Balance"
        if trick_room_density >= 1:
            return "Semi-Room Balance"
        return "Balance"

    shape = _summarize_offense_shape(cluster_entries)
    return f"{parent_archetype} {shape}".strip()


def _summarize_cluster(cluster_id: str, parent_archetype: str, cluster_entries: list[dict[str, Any]]) -> dict[str, Any]:
    pokemon_counter: Counter[str] = Counter()
    move_counter: Counter[str] = Counter()
    item_counter: Counter[str] = Counter()
    ability_counter: Counter[str] = Counter()
    quality_counter: Counter[str] = Counter()

    sample_teams: list[dict[str, Any]] = []
    speeds: list[float] = []
    attacks: list[float] = []
    sp_attacks: list[float] = []
    bulks: list[float] = []
    win_rates: list[float] = []

    for entry in cluster_entries:
        team = entry["team"]
        features = entry["features"]

        pokemon_counter.update(_normalize_sequence(team.get("pokemon_names", [])))
        move_counter.update(_normalize_sequence(team.get("moves", [])))
        item_counter.update(_normalize_sequence(team.get("items", [])))
        ability_counter.update(_normalize_sequence(team.get("abilities", [])))
        quality_counter.update(features.get("data_quality_notes", []))

        if features.get("average_speed") is not None:
            speeds.append(features["average_speed"])
        if features.get("average_attack") is not None:
            attacks.append(features["average_attack"])
        if features.get("average_sp_attack") is not None:
            sp_attacks.append(features["average_sp_attack"])
        if features.get("average_bulk") is not None:
            bulks.append(features["average_bulk"])
        if features.get("team_win_rate") is not None:
            win_rates.append(features["team_win_rate"])

        if len(sample_teams) < 5:
            sample_teams.append(
                {
                    "teamId": team.get("team_id"),
                    "playerName": team.get("player_name"),
                    "tournamentName": team.get("tournament_name"),
                    "pokemon": [_canonical_display(name) for name in team.get("pokemon_names", [])],
                    "wins": team.get("wins"),
                    "losses": team.get("losses"),
                    "placement": team.get("placement"),
                    "confidence": entry["classification"].get("confidence"),
                }
            )

    return {
        "clusterId": cluster_id,
        "parentArchetype": parent_archetype,
        "suggestedLabel": _suggest_cluster_label(parent_archetype, cluster_entries),
        "teamCount": len(cluster_entries),
        "commonPokemon": _top_counter_values(pokemon_counter),
        "commonMoves": _top_counter_values(move_counter),
        "commonItems": _top_counter_values(item_counter),
        "commonAbilities": _top_counter_values(ability_counter),
        "averageTeamStats": {
            "averageSpeed": _safe_average(speeds),
            "averageAttack": _safe_average(attacks),
            "averageSpecialAttack": _safe_average(sp_attacks),
            "averageBulk": _safe_average(bulks),
        },
        "averageWinRate": _safe_average(win_rates),
        "sampleTeams": sample_teams,
        "commonDataQualityNotes": [
            {"note": note, "count": count}
            for note, count in quality_counter.most_common(10)
        ],
    }


def _prepare_team_entries() -> list[dict[str, Any]]:
    team_entries: list[dict[str, Any]] = []
    for team in load_tournament_teams():
        features = extract_team_features(team)
        classification = classify_team_archetype(features)
        team_entries.append(
            {
                "team": team,
                "features": features,
                "classification": classification,
            }
        )
    return team_entries


def build_archetype_cluster_summary() -> dict[str, Any]:
    team_entries = _prepare_team_entries()
    grouped_entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in team_entries:
        grouped_entries[entry["classification"]["primaryArchetype"]].append(entry)

    clusters: list[dict[str, Any]] = []
    group_summaries: list[dict[str, Any]] = []

    for parent_archetype, entries in sorted(grouped_entries.items(), key=lambda item: len(item[1]), reverse=True):
        team_count = len(entries)
        cluster_count = _choose_cluster_count(team_count)
        group_summary = {
            "parentArchetype": parent_archetype,
            "teamCount": team_count,
            "clusterCount": cluster_count if cluster_count > 0 else 0,
            "clustered": cluster_count > 0,
        }

        if cluster_count == 0:
            group_summaries.append(group_summary)
            continue

        matrix = _build_feature_matrix(entries)
        if matrix.empty or len(matrix) < cluster_count:
            group_summaries.append(group_summary)
            continue

        scaled_matrix = StandardScaler(with_mean=False).fit_transform(matrix)
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init="auto")
        labels = model.fit_predict(scaled_matrix)

        label_to_entries: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for index, label in enumerate(labels):
            label_to_entries[int(label)].append(entries[index])

        for label, cluster_entries in sorted(label_to_entries.items()):
            clusters.append(
                _summarize_cluster(
                    cluster_id=f"{normalize_name(parent_archetype)}-{label + 1}",
                    parent_archetype=parent_archetype,
                    cluster_entries=cluster_entries,
                )
            )

        group_summary["clusterCount"] = len(label_to_entries)
        group_summaries.append(group_summary)

    summary = {
        "totalTeamsAnalyzed": len(team_entries),
        "archetypeGroups": group_summaries,
        "clusterCount": len(clusters),
        "clusters": clusters,
    }

    ARCHETYPE_CLUSTER_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARCHETYPE_CLUSTER_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@lru_cache(maxsize=1)
def get_archetype_cluster_summary(force_rebuild: bool = False) -> dict[str, Any]:
    if force_rebuild or not ARCHETYPE_CLUSTER_SUMMARY_PATH.exists():
        return build_archetype_cluster_summary()
    return json.loads(ARCHETYPE_CLUSTER_SUMMARY_PATH.read_text(encoding="utf-8"))


def _cluster_similarity_score(team_features: dict[str, Any], cluster: dict[str, Any]) -> float:
    team_pokemon = {normalize_name(name) for name in team_features.get("pokemon_names", []) if name}
    cluster_pokemon = {normalize_name(entry["name"]) for entry in cluster.get("commonPokemon", [])[:8]}
    pokemon_overlap = len(team_pokemon & cluster_pokemon)

    team_speed = float(team_features.get("average_speed") or 0)
    team_attack = float(team_features.get("average_attack") or 0)
    team_sp_attack = float(team_features.get("average_sp_attack") or 0)
    team_bulk = float(team_features.get("average_bulk") or 0)

    cluster_stats = cluster.get("averageTeamStats", {}) or {}
    stat_distance = abs(team_speed - float(cluster_stats.get("averageSpeed") or 0))
    stat_distance += abs(team_attack - float(cluster_stats.get("averageAttack") or 0))
    stat_distance += abs(team_sp_attack - float(cluster_stats.get("averageSpecialAttack") or 0))
    stat_distance += abs(team_bulk - float(cluster_stats.get("averageBulk") or 0)) / 2

    normalized_stat_similarity = max(0.0, 1.0 - min(stat_distance / 250.0, 1.0))
    overlap_similarity = min(1.0, pokemon_overlap / max(1, min(6, len(team_pokemon))))
    return round(overlap_similarity * 0.65 + normalized_stat_similarity * 0.35, 3)


def find_matching_archetype_cluster(team: dict[str, Any]) -> dict[str, Any] | None:
    features = extract_team_features(team)
    classification = classify_team_archetype(features)
    cluster_summary = get_archetype_cluster_summary()
    parent_archetype = classification.get("primaryArchetype", "")

    candidate_clusters = [
        cluster
        for cluster in cluster_summary.get("clusters", [])
        if cluster.get("parentArchetype") == parent_archetype
    ]
    if not candidate_clusters:
        return None

    ranked = sorted(
        candidate_clusters,
        key=lambda cluster: _cluster_similarity_score(features, cluster),
        reverse=True,
    )
    best_cluster = ranked[0]
    return {
        "primaryArchetype": parent_archetype,
        "clusterId": best_cluster.get("clusterId"),
        "suggestedLabel": best_cluster.get("suggestedLabel"),
        "similarityScore": _cluster_similarity_score(features, best_cluster),
        "cluster": best_cluster,
    }
