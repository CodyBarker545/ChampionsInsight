"""Build a precomputed JSON cache for Pokedex meta intelligence."""

from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from paths import POKEDEX_META_INTELLIGENCE_PATH
from services.meta_trend_service import write_pokedex_meta_intelligence_cache


def main() -> None:
    payload = write_pokedex_meta_intelligence_cache()
    print(f"Wrote {len(payload)} Pokemon meta entries to {POKEDEX_META_INTELLIGENCE_PATH}")


if __name__ == "__main__":
    main()
