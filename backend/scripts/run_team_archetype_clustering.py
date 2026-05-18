"""Run team archetype clustering and save cluster outputs for manual review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.team_cluster_service import (
    TEAM_CLUSTER_ASSIGNMENTS_PATH,
    TEAM_CLUSTER_SUMMARY_PATH,
    cluster_team_archetypes,
    save_team_cluster_outputs,
    summarize_team_clusters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster tournament teams into archetype groups.")
    parser.add_argument("--clusters", type=int, default=12, help="Number of archetype clusters to build.")
    parser.add_argument("--top-n", type=int, default=12, help="How many top features to show per cluster.")
    parser.add_argument(
        "--sample-teams",
        type=int,
        default=5,
        help="How many representative teams to include per cluster.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assignments, _ = cluster_team_archetypes(cluster_count=args.clusters)
    summary = summarize_team_clusters(
        assignments,
        top_n=args.top_n,
        sample_teams_per_cluster=args.sample_teams,
    )
    save_team_cluster_outputs(assignments, summary)

    print(f"Clustered teams: {len(assignments)}")
    print(f"Clusters created: {summary['clusterCount']}")
    print(f"Assignments saved to: {TEAM_CLUSTER_ASSIGNMENTS_PATH}")
    print(f"Summary saved to: {TEAM_CLUSTER_SUMMARY_PATH}")
    if summary["clusters"]:
        first_cluster = summary["clusters"][0]
        print(f"First cluster id: {first_cluster['cluster_id']}")
        print(f"Top pokemon: {[entry['name'] for entry in first_cluster['top_pokemon'][:6]]}")


if __name__ == "__main__":
    main()
