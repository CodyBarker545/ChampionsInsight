"""Run clustering inside rule-based archetype groups."""

from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from paths import ARCHETYPE_CLUSTER_SUMMARY_PATH
from services.archetype_cluster_service import build_archetype_cluster_summary


def main() -> None:
    summary = build_archetype_cluster_summary()
    print(f"total teams analyzed: {summary['totalTeamsAnalyzed']}")
    print("archetype groups:")
    for group in summary.get("archetypeGroups", []):
        print(
            f"  - {group['parentArchetype']}: teams={group['teamCount']}, "
            f"clustered={group['clustered']}, clusters={group['clusterCount']}"
        )
    print(f"total clusters written: {summary['clusterCount']}")
    print(f"saved to: {ARCHETYPE_CLUSTER_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
