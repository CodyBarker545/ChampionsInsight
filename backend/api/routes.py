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
from api.set_probability_routes import set_probability_bp
from api.tournament_routes import tournament_bp
from api.user_team_routes import user_team_bp
from services.archetype_cluster_service import (
    find_matching_archetype_cluster,
    get_archetype_cluster_summary,
)
from services.archetype_rule_service import (
    classify_team_archetype,
    get_archetype_rule_summary,
)
from services.archetype_feature_service import load_pokemon_stat_lookup, normalize_name
from pokemon_damage_calculator.old.calculator import analyze_battle
from services.cv_service import ComputerVisionError, assess_opponent_image_quality, assess_opponent_image_quality_from_image
from services.opponent_detection_service import (
    detect_opponent_team_mobilenetv3,
    detect_opponent_team_mobilenetv3_from_image,
)
from services.image_service import ImageValidationError, UPLOAD_DIR, prepare_opponent_image_for_prediction, save_opponent_image
from services.matchup_service import analyze_matchup
from services.battle_advice_chat_service import (
    answer_battle_question,
    build_battle_context,
    classify_battle_question,
    generate_rule_based_answer,
)
from services.pokedex_service import (
    get_pokedex_entry_detail_by_name,
    get_pokedex_grid_entries,
    get_pokemon_recommended_teams_by_name,
)
from services.pokemon_mobilenetv3_classifier_service import predict_pokemon_crop
from services.meta_trend_service import get_pokemon_meta_intelligence
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
api_bp.register_blueprint(tournament_bp)
api_bp.register_blueprint(set_probability_bp)
