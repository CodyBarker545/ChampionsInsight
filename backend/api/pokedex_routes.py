"""Routes for the Pokedex browser."""

from flask import Blueprint, jsonify

from api.common import route_deps


pokedex_bp = Blueprint("pokedex_routes", __name__)


@pokedex_bp.get("/pokedex")
def get_pokedex():
    pokemon = route_deps().get_pokedex_grid_entries()

    return {
        "count": len(pokemon),
        "pokemon": pokemon,
    }


@pokedex_bp.get("/pokedex/<name>")
def get_pokedex_entry(name):
    pokemon = route_deps().get_pokedex_entry_detail_by_name(name)

    if pokemon is None:
        return {
            "error": "Pokemon not found",
            "name": name,
        }, 404

    return pokemon


@pokedex_bp.get("/pokedex/recommendations/<name>")
def get_pokedex_recommendations(name):
    result = route_deps().get_pokemon_recommended_teams_by_name(name)

    if result is None:
        return {
            "error": "Pokemon not found",
            "name": name,
        }, 404

    return jsonify(result)


@pokedex_bp.get("/pokedex/meta/<name>")
def get_pokedex_meta(name):
    try:
        return jsonify(route_deps().get_pokemon_meta_intelligence(name))
    except ValueError:
        return jsonify({
            "error": "Pokemon not found",
            "name": name,
        }), 404
    except FileNotFoundError as error:
        return jsonify({
            "error": str(error),
            "name": name,
        }), 500
