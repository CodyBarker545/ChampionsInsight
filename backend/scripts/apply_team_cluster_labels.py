"""Apply manual archetype labels to saved team cluster outputs."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from paths import COMPETITIVE_ANALYSIS_DIR


SUMMARY_PATH = COMPETITIVE_ANALYSIS_DIR / "team_archetype_cluster_summary.json"
ASSIGNMENTS_PATH = COMPETITIVE_ANALYSIS_DIR / "team_archetype_cluster_assignments.csv"
LABELS_PATH = COMPETITIVE_ANALYSIS_DIR / "team_archetype_cluster_labels.json"
LABELED_SUMMARY_PATH = COMPETITIVE_ANALYSIS_DIR / "team_archetype_cluster_summary_labeled.json"
LABELED_ASSIGNMENTS_PATH = COMPETITIVE_ANALYSIS_DIR / "team_archetype_cluster_assignments_labeled.csv"

CLUSTER_LABELS = {
    0: {
        "primary_label": "Tailwind",
        "archetype_tags": ["Tailwind"],
        "notes": "Fast offense cluster. Mostly generic Tailwind-style teams with mixed secondary weather modes.",
    },
    1: {
        "primary_label": "Tailwind",
        "archetype_tags": ["Tailwind", "Rain"],
        "notes": "Tailwind-forward cluster with noticeable Pelipper rain presence.",
    },
    2: {
        "primary_label": "Tailwind",
        "archetype_tags": ["Tailwind", "Trick Room"],
        "notes": "Hybrid speed-control cluster. Tailwind is primary, with a meaningful Trick Room backup mode.",
    },
    3: {
        "primary_label": "Tailwind",
        "archetype_tags": ["Tailwind", "Trick Room"],
        "notes": "Another hybrid control cluster with strong Tailwind presence and frequent Trick Room support.",
    },
    4: {
        "primary_label": "Tailwind",
        "archetype_tags": ["Tailwind"],
        "notes": "Small but fairly clean fast-mode cluster led by Tailwind-style teams.",
    },
    5: {
        "primary_label": "Rain",
        "archetype_tags": ["Rain", "Tailwind"],
        "notes": "Rain-leaning cluster with Pelipper/Politoed style teams and common Tailwind support.",
    },
    6: {
        "primary_label": "Snow",
        "archetype_tags": ["Snow"],
        "notes": "Clear snow cluster built around Snow Warning and Blizzard/Aurora Veil patterns.",
    },
    7: {
        "primary_label": "Tailwind",
        "archetype_tags": ["Tailwind"],
        "notes": "Mixed offensive support cluster. Tailwind is the clearest common shared mode.",
    },
    8: {
        "primary_label": "Sun",
        "archetype_tags": ["Sun", "Tailwind"],
        "notes": "Sun-leaning cluster, often with Charizard or Mega Charizard Y and fast support.",
    },
    9: {
        "primary_label": "Tailwind",
        "archetype_tags": ["Tailwind", "Sun"],
        "notes": "Mostly Tailwind offense with some Charizard/Venusaur-style sun overlap.",
    },
    10: {
        "primary_label": "Rain",
        "archetype_tags": ["Rain", "Tailwind"],
        "notes": "Rain-leaning cluster with Pelipper/Archaludon-style structures and Tailwind support.",
    },
    11: {
        "primary_label": "Trick Room",
        "archetype_tags": ["Trick Room", "Sun"],
        "notes": "Tiny cluster, but the strongest signal is Trick Room with some sun overlap.",
    },
}


def write_labels_file() -> None:
    LABELS_PATH.write_text(
        json.dumps(CLUSTER_LABELS, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_labeled_summary() -> None:
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    for cluster in summary.get("clusters", []):
        label_info = CLUSTER_LABELS.get(cluster["cluster_id"], {})
        cluster.update(label_info)
    LABELED_SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_labeled_assignments() -> None:
    with ASSIGNMENTS_PATH.open("r", encoding="utf-8", newline="") as infile:
        rows = list(csv.DictReader(infile))
        fieldnames = list(rows[0].keys()) + ["primary_label", "archetype_tags"]

    with LABELED_ASSIGNMENTS_PATH.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            cluster_id = int(row["cluster_id"])
            label_info = CLUSTER_LABELS.get(cluster_id, {})
            row["primary_label"] = label_info.get("primary_label", "")
            row["archetype_tags"] = "|".join(label_info.get("archetype_tags", []))
            writer.writerow(row)


def main() -> None:
    write_labels_file()
    write_labeled_summary()
    write_labeled_assignments()
    print(f"Saved labels to: {LABELS_PATH}")
    print(f"Saved labeled summary to: {LABELED_SUMMARY_PATH}")
    print(f"Saved labeled assignments to: {LABELED_ASSIGNMENTS_PATH}")


if __name__ == "__main__":
    main()
