"""Routes for local RAG questions."""

from flask import Blueprint, jsonify, request

from api.common import route_deps


rag_bp = Blueprint("rag_routes", __name__)


@rag_bp.post("/rag/ask")
def ask_rag():
    payload = request.get_json(silent=True) or {}

    try:
        result = route_deps().answer_rag_question(payload.get("question", ""))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    return jsonify(result)
