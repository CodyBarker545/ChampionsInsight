"""Central filesystem layout for backend data, artifacts, and scripts."""

from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parent

DATA_DIR = BACKEND_DIR / "data"

# Curated/source data used by services.
POKEMON_DATA_DIR = DATA_DIR / "pokemon"
POKEMON_BATTLE_DATA_PATH = POKEMON_DATA_DIR / "pokemon_battle_data.json"
CHAMPIONS_ROSTER_ABILITIES_PATH = POKEMON_DATA_DIR / "champions_lab_roster_abilities.json"
SPRITE_ROOT = POKEMON_DATA_DIR / "champions_sprites"
SPRITE_METADATA_PATH = SPRITE_ROOT / "champions_sprite_database.json"

COMPETITIVE_DATA_DIR = DATA_DIR / "competitive"
COMPETITIVE_ANALYSIS_DIR = COMPETITIVE_DATA_DIR / "analysis_json"
COMPETITIVE_DIAGRAM_DIR = COMPETITIVE_DATA_DIR / "diagrams"
COMPETITIVE_RAW_DIR = COMPETITIVE_DATA_DIR / "vgc_data"
POKEDEX_USAGE_SUMMARY_PATH = COMPETITIVE_ANALYSIS_DIR / "pokedex_usage_summary.json"
TOP_MOVES_PATH = COMPETITIVE_ANALYSIS_DIR / "pokemon_top_6_moves.json"

USER_DATA_DIR = DATA_DIR / "user"
USER_TEAM_PATH = USER_DATA_DIR / "user_team.json"

RAG_DOCS_DIR = DATA_DIR / "rag" / "docs"

# Runtime uploads and generated computer-vision artifacts.
UPLOAD_DIR = DATA_DIR / "uploads"

CV_DATA_DIR = DATA_DIR / "cv"
CV_REFERENCE_DIR = CV_DATA_DIR / "references"
POKEMON_REFERENCE_DIR = CV_REFERENCE_DIR / "pokemon"
TYPE_REFERENCE_DIR = CV_REFERENCE_DIR / "types"
TYPE_ICON_REFERENCE_DIR = TYPE_REFERENCE_DIR / "type_icons"
TYPE_COMBO_REFERENCE_DIR = TYPE_REFERENCE_DIR / "type_combo_icons"
TYPE_COMBO_REFERENCE_METADATA_PATH = TYPE_COMBO_REFERENCE_DIR / "type_combo_metadata.json"

CV_INDEX_DIR = CV_DATA_DIR / "indexes"
POKEMON_EMBEDDING_INDEX_DIR = CV_INDEX_DIR / "pokemon_embeddings"
TYPE_EMBEDDING_INDEX_DIR = CV_INDEX_DIR / "type_embeddings"

CV_DEBUG_DIR = CV_DATA_DIR / "debug"
OPPONENT_DEBUG_CROP_DIR = CV_DEBUG_DIR / "crops"
CV_DEBUG_REPORT_DIR = CV_DEBUG_DIR / "reports"

# Ensure runtime directories exist.
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OPPONENT_DEBUG_CROP_DIR.mkdir(parents=True, exist_ok=True)
CV_DEBUG_REPORT_DIR.mkdir(parents=True, exist_ok=True)
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)