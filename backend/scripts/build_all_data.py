"""Build every derived data artifact needed for a fresh local install."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent


@dataclass(frozen=True)
class BuildStep:
    name: str
    command: list[str]
    outputs: list[Path]
    source_paths: list[Path]
    category: str = "data"
    optional: bool = False


def build_steps(include_debug_report: bool = False) -> list[BuildStep]:
    steps = [
        BuildStep(
            name="Validate and preview RAG document chunks",
            command=[sys.executable, "-m", "rag.preprocess"],
            outputs=[],
            source_paths=[BACKEND_DIR / "RAGDocs"],
            category="ai_artifact",
        ),
        BuildStep(
            name="Build RAG vector index",
            command=[sys.executable, "-m", "rag.build_faiss_index"],
            outputs=[
                BACKEND_DIR / "vector_store" / "pokemon_rag_embeddings.npy",
                BACKEND_DIR / "vector_store" / "chunks.json",
            ],
            source_paths=[BACKEND_DIR / "RAGDocs"],
            category="ai_artifact",
        ),
        BuildStep(
            name="Build type combo reference images",
            command=[sys.executable, "scripts/data_build/build_type_combo_references.py"],
            outputs=[
                BACKEND_DIR
                / "data"
                / "cv"
                / "references"
                / "types"
                / "type_combo_icons"
                / "type_combo_metadata.json",
            ],
            source_paths=[
                BACKEND_DIR / "data" / "pokemon" / "champions_sprites",
                BACKEND_DIR / "data" / "cv" / "references" / "types" / "type_icons",
            ],
            category="ai_artifact",
        ),
        BuildStep(
            name="Build type icon embedding index",
            command=[sys.executable, "scripts/data_build/build_type_embedding_index.py"],
            outputs=[
                BACKEND_DIR / "data" / "cv" / "indexes" / "type_embeddings" / "embeddings.npy",
                BACKEND_DIR / "data" / "cv" / "indexes" / "type_embeddings" / "metadata.json",
            ],
            source_paths=[BACKEND_DIR / "data" / "cv" / "references" / "types" / "type_icons"],
            category="ai_artifact",
        ),
        BuildStep(
            name="Build Pokemon sprite embedding index",
            command=[sys.executable, "scripts/data_build/build_pokemon_embedding_index.py"],
            outputs=[
                BACKEND_DIR / "data" / "cv" / "indexes" / "pokemon_embeddings" / "embeddings.npy",
                BACKEND_DIR / "data" / "cv" / "indexes" / "pokemon_embeddings" / "metadata.json",
            ],
            source_paths=[
                BACKEND_DIR / "data" / "pokemon" / "champions_sprites",
                BACKEND_DIR / "data" / "cv" / "references" / "pokemon",
            ],
            category="ai_artifact",
        ),
        BuildStep(
            name="Prune blocked Pokemon embedding references",
            command=[sys.executable, "scripts/data_build/prune_blocked_embedding_index.py"],
            outputs=[
                BACKEND_DIR / "data" / "cv" / "indexes" / "pokemon_embeddings" / "embeddings.npy",
                BACKEND_DIR / "data" / "cv" / "indexes" / "pokemon_embeddings" / "metadata.json",
            ],
            source_paths=[BACKEND_DIR / "data" / "cv" / "indexes" / "pokemon_embeddings"],
            category="ai_artifact",
        ),
        BuildStep(
            name="Build tournament usage, moves and items",
            command=[sys.executable, "scripts/build_pokedex_usage_summary.py"],
            outputs=[
                BACKEND_DIR / "data/competitive/analysis_json/pokedex_usage_summary.json",
                BACKEND_DIR / "data/competitive/analysis_json/pokemon_top_6_moves.json",
                BACKEND_DIR / "data/competitive/analysis_json/pokemon_top_3_items.json",
            ],
            source_paths=[BACKEND_DIR / "data/competitive/vgc_data/pokemon_teamlists.csv"],
        ),
        BuildStep(
            name="Build set probability summaries",
            command=[sys.executable, "scripts/run_set_probability_analysis.py"],
            outputs=[
                BACKEND_DIR / "data" / "competitive" / "analysis_json" / "set_probability_summary.json",
                BACKEND_DIR / "data" / "competitive" / "analysis_json" / "conditional_set_probability_summary.json",
            ],
            source_paths=[
                BACKEND_DIR / "data" / "competitive" / "vgc_data" / "pokemon_teamlists.csv",
                BACKEND_DIR / "data" / "competitive" / "vgc_data" / "tournaments.csv",
            ],
        ),
        BuildStep(
            name="Build archetype rule summaries",
            command=[sys.executable, "scripts/run_archetype_rule_analysis.py"],
            outputs=[
                BACKEND_DIR / "data" / "competitive" / "analysis_json" / "archetype_rule_summary.json",
                BACKEND_DIR / "data" / "competitive" / "analysis_json" / "archetype_team_classifications.json",
            ],
            source_paths=[
                BACKEND_DIR / "data" / "competitive" / "vgc_data" / "pokemon_teamlists.csv",
                BACKEND_DIR / "data" / "competitive" / "vgc_data" / "tournaments.csv",
            ],
        ),
        BuildStep(
            name="Build archetype cluster summaries",
            command=[sys.executable, "scripts/run_archetype_clustering.py"],
            outputs=[
                BACKEND_DIR / "data" / "competitive" / "analysis_json" / "archetype_cluster_summary.json",
            ],
            source_paths=[
                BACKEND_DIR / "data" / "competitive" / "vgc_data" / "pokemon_teamlists.csv",
                BACKEND_DIR / "data" / "competitive" / "vgc_data" / "tournaments.csv",
            ],
        ),
        BuildStep(
            name="Build Pokedex meta intelligence cache",
            command=[sys.executable, "scripts/build_pokedex_meta_intelligence_cache.py"],
            outputs=[
                BACKEND_DIR / "data" / "competitive" / "analysis_json" / "pokedex_meta_intelligence.json",
            ],
            source_paths=[
                BACKEND_DIR / "data" / "competitive" / "vgc_data" / "pokemon_teamlists.csv",
                BACKEND_DIR / "data" / "competitive" / "vgc_data" / "tournaments.csv",
            ],
        ),
        BuildStep(
            name="Build Pokemon recommended builds cache",
            command=[sys.executable, "scripts/build_pokemon_recommended_builds.py"],
            outputs=[
                BACKEND_DIR / "data" / "competitive" / "analysis_json" / "pokemon_recommended_builds.json",
            ],
            source_paths=[
                BACKEND_DIR / "data" / "competitive" / "vgc_data" / "pokemon_teamlists.csv",
                BACKEND_DIR / "data" / "competitive" / "vgc_data" / "tournaments.csv",
            ],
        ),
    ]

    if include_debug_report:
        steps.append(
            BuildStep(
                name="Run uploaded opponent detection debug report",
                command=[sys.executable, "scripts/cv_runtime/test_uploaded_opponent_detections.py"],
                outputs=[
                    BACKEND_DIR
                    / "data"
                    / "cv"
                    / "debug"
                    / "reports"
                    / "uploaded_opponent_detection_debug_report.json",
                ],
                source_paths=[BACKEND_DIR / "data" / "uploads"],
                optional=True,
            )
        )

    # The app uses supervised YOLO/MobileNet models, not these older indexes.
    legacy_names = {
        "Build type combo reference images",
        "Build type icon embedding index",
        "Build Pokemon sprite embedding index",
        "Prune blocked Pokemon embedding references",
    }
    return [replace(step, category="legacy_cv") if step.name in legacy_names else step for step in steps]


def validate_sources(step: BuildStep) -> None:
    missing = [path for path in step.source_paths if not path.exists()]
    if missing:
        missing_list = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(f"{step.name} is missing required source data:\n{missing_list}")


def run_step(step: BuildStep, dry_run: bool = False) -> None:
    print(flush=True)
    print(f"==> {step.name}", flush=True)
    print("    " + " ".join(str(part) for part in step.command), flush=True)

    missing_sources = [path for path in step.source_paths if not path.exists()]
    if missing_sources and step.outputs and all(path.exists() for path in step.outputs):
        print("    Required source data is not present; using existing prebuilt outputs.", flush=True)
        for path in step.outputs:
            print(f"    Found: {path}", flush=True)
        return

    validate_sources(step)

    if dry_run:
        return

    subprocess.run(step.command, cwd=BACKEND_DIR, check=True)

    missing_outputs = [path for path in step.outputs if not path.exists()]
    if missing_outputs:
        missing_list = "\n".join(f"  - {path}" for path in missing_outputs)
        raise FileNotFoundError(f"{step.name} completed but did not create:\n{missing_list}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build all derived Champions Insight data artifacts."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the steps without running them.",
    )
    parser.add_argument(
        "--skip-rag",
        action="store_true",
        help="Skip RAG preprocessing and vector index generation.",
    )
    parser.add_argument(
        "--with-legacy-cv",
        action="store_true",
        help="Also rebuild obsolete reference/embedding CV artifacts for offline experiments.",
    )
    parser.add_argument(
        "--skip-cv",
        action="store_true",
        help="Skip computer-vision reference and embedding indexes.",
    )
    parser.add_argument(
        "--skip-ai-artifacts",
        action="store_true",
        help="Skip RAG/CV artifact generation because prebuilt artifacts are already shipped.",
    )
    parser.add_argument(
        "--with-debug-report",
        action="store_true",
        help="Also run the upload debug report against backend/data/uploads.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    steps = build_steps(include_debug_report=args.with_debug_report)

    if args.skip_rag:
        steps = [step for step in steps if "RAG" not in step.name]

    if args.skip_cv:
        steps = [
            step
            for step in steps
            if step.category != "legacy_cv" and not step.optional
        ]

    if args.skip_ai_artifacts:
        steps = [step for step in steps if step.category not in {"ai_artifact", "legacy_cv"}]

    if not args.with_legacy_cv:
        steps = [step for step in steps if step.category != "legacy_cv"]

    print(f"Project root: {PROJECT_ROOT}", flush=True)
    print(f"Backend dir:  {BACKEND_DIR}", flush=True)

    for step in steps:
        run_step(step, dry_run=args.dry_run)

    print(flush=True)
    print("All requested data artifacts are ready.", flush=True)


if __name__ == "__main__":
    main()
