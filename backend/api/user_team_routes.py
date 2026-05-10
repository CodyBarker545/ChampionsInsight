"""Routes for locally stored user teams."""

import logging

from flask import Blueprint, jsonify, request

from api.common import route_deps


user_team_bp = Blueprint("user_team_routes", __name__)
logger = logging.getLogger(__name__)


@user_team_bp.get("/user/team")
def get_user_team():
    try:
        return jsonify(route_deps().load_user_team())
    except FileNotFoundError as error:
        logger.error("User team file could not be loaded: %s", error)
        return jsonify({"error": str(error)}), 500


@user_team_bp.put("/user/team")
def put_user_team():
    payload = request.get_json(silent=True) or {}

    try:
        saved_team = route_deps().save_user_team(
            payload.get("team"),
            user_id=payload.get("userId", "local-demo-user"),
            team_name=payload.get("teamName", "Saved Battle Team"),
        )
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify(saved_team)
