"""Routes for matchup and damage calculations."""

from flask import Blueprint, jsonify, request

from api.common import route_deps


battle_bp = Blueprint("battle_routes", __name__)


@battle_bp.post("/team/analyze")
def analyze_team():
    payload = request.get_json(silent=True) or {}

    try:
        result = route_deps().analyze_matchup(payload)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify(result)


@battle_bp.post("/damage/calculate")
def calculate_damage():
    payload = request.get_json(silent=True) or {}

    try:
        result = route_deps().analyze_battle(payload)
    except (KeyError, ValueError) as error:
        return jsonify({"error": str(error)}), 400

    return jsonify(result)
