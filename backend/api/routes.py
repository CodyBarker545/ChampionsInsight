"""Public API blueprint and compatibility exports.

Route handlers live in domain modules under ``backend/api``. This module stays
as the single import point for the Flask app and for tests that monkeypatch
route dependencies.
"""

from flask import Blueprint

from api.battle_routes import battle_bp
from api.common import (
    build_sprite_url_from_reference,
    clear_latest_opponent_prediction,
    get_latest_prediction_path,
    get_latest_uploaded_image_path,
    get_uploaded_image_path_from_request,
    is_valid_uploaded_filename,
    load_latest_opponent_prediction,
    make_json_safe,
    save_latest_opponent_prediction,
)
from api.health_routes import health_bp
from api.opponent_routes import opponent_bp
from api.pokedex_routes import pokedex_bp
from api.pokemon_routes import pokemon_bp
from api.rag_routes import rag_bp
from api.user_team_routes import user_team_bp
from pokemon_damage_calculator.calculator import analyze_battle
from services.cv_detection_service import (
    assess_opponent_image_quality,
    detect_opponent_team,
    detect_opponent_team_types,
)
from services.cv_service import ComputerVisionError
from services.image_service import ImageValidationError, UPLOAD_DIR, save_opponent_image
from services.matchup_service import analyze_matchup
from services.pokedex_service import (
    get_pokedex_entry_detail_by_name,
    get_pokedex_grid_entries,
)
from services.pokemon_data_service import (
    SPRITE_ROOT,
    build_pokemon_summary,
    search_pokemon_summaries,
)
from services.pokemon_moves_service import get_top_tournament_moves
from services.pokemon_stats_service import get_level_50_stats
from services.rag_service import answer_rag_question
from services.user_team_service import load_user_team, save_user_team


api_bp = Blueprint("api", __name__)

api_bp.register_blueprint(health_bp)
api_bp.register_blueprint(user_team_bp)
api_bp.register_blueprint(battle_bp)
api_bp.register_blueprint(opponent_bp)
api_bp.register_blueprint(pokemon_bp)
api_bp.register_blueprint(rag_bp)
api_bp.register_blueprint(pokedex_bp)
