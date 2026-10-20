"""Build cached tournament-backed recommended builds for the Pokedex Teams tab."""

from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
for path in (PROJECT_ROOT, BACKEND_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from paths import POKEMON_RECOMMENDED_BUILDS_PATH  # noqa: E402
from services.pokemon_recommended_build_service import write_recommended_builds_cache  # noqa: E402


def main() -> None:
    payload = write_recommended_builds_cache()
    print(
        f"Wrote {len(payload)} recommended build entries to {POKEMON_RECOMMENDED_BUILDS_PATH}"
    )


if __name__ == "__main__":
    main()
