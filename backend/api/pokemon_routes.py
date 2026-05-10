"""Routes for Pokemon lookup, stats, moves, and local sprites."""

from flask import Blueprint, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from api.common import route_deps


pokemon_bp = Blueprint("pokemon_routes", __name__)


@pokemon_bp.get("/pokemon/<pokemon_name>")
def get_pokemon(pokemon_name):
    summary = route_deps().build_pokemon_summary(pokemon_name)

    if not summary.get("baseStats"):
        return jsonify({"error": "Pokemon was not found in the local dataset."}), 404

    return jsonify(summary)


@pokemon_bp.get("/pokemon/search")
def search_pokemon():
    query = request.args.get("q", "")
    limit = request.args.get("limit", 12, type=int)
    safe_limit = min(max(limit, 1), 24)

    return jsonify({
        "query": query,
        "results": route_deps().search_pokemon_summaries(query, safe_limit),
    })


@pokemon_bp.get("/pokemon/<string:name>/stats")
def pokemon_level_50_stats(name):
    nature = request.args.get("nature", "hardy")
    result = route_deps().get_level_50_stats(name, nature)

    if result is None:
        return jsonify({"error": f"Pokemon not found: {name}"}), 404

    return jsonify(result), 200


@pokemon_bp.get("/pokemon/<string:name>/moves/top")
def pokemon_top_tournament_moves(name):
    limit = request.args.get("limit", 4, type=int)
    safe_limit = min(max(limit, 1), 6)

    return jsonify(route_deps().get_top_tournament_moves(name, safe_limit)), 200


@pokemon_bp.get("/pokemon/sprite/<sprite_type>/<filename>")
def get_pokemon_sprite(sprite_type, filename):
    safe_sprite_type = secure_filename(sprite_type)
    safe_filename = secure_filename(filename)
    sprite_dir = route_deps().SPRITE_ROOT / safe_sprite_type

    if (
        safe_sprite_type != sprite_type
        or safe_filename != filename
        or not sprite_dir.exists()
    ):
        return jsonify({"error": "Sprite was not found."}), 404

    return send_from_directory(sprite_dir, safe_filename)
