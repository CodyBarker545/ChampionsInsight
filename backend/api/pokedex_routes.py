"""Routes for the Pokedex browser."""

from flask import Blueprint

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
