"""Routes for tournament-based move/item set probability predictions."""

from __future__ import annotations

from flask import Blueprint, jsonify, request

from services.set_probability_service import (
    calculate_basic_probabilities,
    calculate_conditional_probabilities,
    load_pokemon_set_records,
    predict_likely_set,
    predict_team_sets,
)


set_probability_bp = Blueprint("set_probability_routes", __name__)


def _json_error(message: str, status_code: int = 400):
    return jsonify({"error": message}), status_code


def _has_tournament_data() -> bool:
    return bool(load_pokemon_set_records())


@set_probability_bp.get("/tournament/set-probabilities/<pokemon_name>")
def get_set_probabilities(pokemon_name: str):
    if not pokemon_name or not str(pokemon_name).strip():
        return _json_error("A Pokemon name is required.", 400)

    if not _has_tournament_data():
        return _json_error("Tournament set probability data is not available.", 503)

    basic_result = calculate_basic_probabilities(pokemon_name, limit=10)
    predicted_set = predict_likely_set(pokemon_name)

    if basic_result.get("sampleSize", 0) == 0:
        return _json_error(f"No tournament records found for {pokemon_name}.", 404)

    return jsonify({
        "pokemon": basic_result.get("pokemon", ""),
        "sampleSize": basic_result.get("sampleSize", 0),
        "moves": basic_result.get("moves", []),
        "items": basic_result.get("items", []),
        "abilities": basic_result.get("abilities", []),
        "natures": basic_result.get("natures", []),
        "predictedSet": {
            "item": predicted_set.get("predictedItem", ""),
            "ability": predicted_set.get("predictedAbility", ""),
            "nature": predicted_set.get("predictedNature", ""),
            "moves": predicted_set.get("predictedMoves", []),
        },
        "confidence": predicted_set.get("confidence", 0.0),
        "confidenceLabel": predicted_set.get("confidenceLabel", "Low"),
        "dataQualityNotes": basic_result.get("dataQualityNotes", []),
    })


@set_probability_bp.post("/tournament/set-probabilities/context")
def get_contextual_set_probabilities():
    if not _has_tournament_data():
        return _json_error("Tournament set probability data is not available.", 503)

    payload = request.get_json(silent=True)
    if payload is None:
        return _json_error("A JSON request body is required.", 400)

    pokemon_name = payload.get("pokemon", "")
    teammates = payload.get("teammates")
    archetype = payload.get("archetype")

    if not pokemon_name or not str(pokemon_name).strip():
        return _json_error("A Pokemon name is required.", 400)
    if teammates is not None and not isinstance(teammates, list):
        return _json_error("Teammates must be provided as a list.", 400)

    conditional_result = calculate_conditional_probabilities(
        pokemon_name,
        teammates=teammates,
        archetype=archetype,
        limit=10,
    )
    predicted_set = predict_likely_set(
        pokemon_name,
        teammates=teammates,
        archetype=archetype,
    )

    if conditional_result.get("baseSampleSize", 0) == 0:
        return _json_error(f"No tournament records found for {pokemon_name}.", 404)

    return jsonify({
        "pokemon": conditional_result.get("pokemon", ""),
        "context": conditional_result.get("context", {}),
        "baseSampleSize": conditional_result.get("baseSampleSize", 0),
        "matchedSampleSize": conditional_result.get("matchedSampleSize", 0),
        "moves": conditional_result.get("moves", []),
        "items": conditional_result.get("items", []),
        "abilities": conditional_result.get("abilities", []),
        "natures": conditional_result.get("natures", []),
        "predictedSet": {
            "item": predicted_set.get("predictedItem", ""),
            "ability": predicted_set.get("predictedAbility", ""),
            "nature": predicted_set.get("predictedNature", ""),
            "moves": predicted_set.get("predictedMoves", []),
        },
        "confidence": predicted_set.get("confidence", 0.0),
        "confidenceLabel": predicted_set.get("confidenceLabel", "Low"),
        "explanations": conditional_result.get("explanations", []),
        "dataQualityNotes": conditional_result.get("dataQualityNotes", []),
    })


@set_probability_bp.post("/tournament/set-probabilities/team")
def get_team_set_probabilities():
    if not _has_tournament_data():
        return _json_error("Tournament set probability data is not available.", 503)

    payload = request.get_json(silent=True)
    if payload is None:
        return _json_error("A JSON request body is required.", 400)

    opponent_team = payload.get("opponentTeam")
    if opponent_team is None:
        return _json_error("An opponentTeam list is required.", 400)
    if not isinstance(opponent_team, list):
        return _json_error("opponentTeam must be provided as a list.", 400)
    if not opponent_team:
        return _json_error("At least one opponent Pokemon is required.", 400)
    if len(opponent_team) > 6:
        return _json_error("No more than 6 opponent Pokemon can be provided.", 400)

    prediction = predict_team_sets(opponent_team)
    if not prediction.get("predictions"):
        return _json_error("No recognizable opponent Pokemon names were provided.", 400)

    return jsonify(prediction)
