"""Routes for matchup, damage calculations, and battle advice chat."""

from flask import Blueprint, current_app, jsonify, request

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


@battle_bp.post("/battle/advice-chat")
def battle_advice_chat():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "A JSON request body is required."}), 400

    question = str(payload.get("question", "")).strip()
    if not question:
        return jsonify({"error": "Question is required."}), 400

    try:
        result = route_deps().answer_battle_question(
            question=question,
            opponent_team=payload.get("opponentTeam"),
            user_team=payload.get("userTeam"),
            damage_context=payload.get("damageContext"),
            speed_context=payload.get("speedContext"),
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        current_app.logger.exception("Battle question failed")
        return jsonify({"error": "Could not answer that battle question right now. Please try again."}), 503

    return jsonify(result)
