"""Health-check routes."""

from flask import Blueprint, jsonify


health_bp = Blueprint("health_routes", __name__)


@health_bp.get("/health")
def health():
    return jsonify({"status": "ok", "app": "Champions Insight"})
