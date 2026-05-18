"""Routes for tournament archetype summaries and team classification."""

import json

from flask import Blueprint, jsonify, request

from api.common import route_deps


tournament_bp = Blueprint("tournament_routes", __name__)


def _json_error(message, status_code=400):
    return jsonify({"error": message}), status_code


def _group_clusters_by_parent(cluster_summary):
    grouped = {}
    for cluster in cluster_summary.get("clusters", []):
        parent = cluster.get("parentArchetype", "Unknown")
        grouped.setdefault(parent, []).append(cluster)
    return grouped


def _extract_cluster_quality_notes(cluster_summary):
    notes = []
    for cluster in cluster_summary.get("clusters", []):
        for note in cluster.get("commonDataQualityNotes", []):
            if note not in notes:
                notes.append(note)
    return notes


@tournament_bp.get("/tournament/archetypes/rules")
def get_tournament_archetype_rules():
    try:
        summary = route_deps().get_archetype_rule_summary()
    except FileNotFoundError:
        return _json_error("Tournament archetype rule summary has not been generated yet.", 404)
    except (OSError, json.JSONDecodeError):
        return _json_error("Tournament archetype rule summary could not be loaded.", 500)

    return jsonify({
        "totalTeamsAnalyzed": summary.get("totalTeamsAnalyzed", summary.get("teamCount", 0)),
        "archetypeSummaries": summary.get("archetypes", []),
        "dataQualityNotes": summary.get("missingDataWarnings", []),
    })


@tournament_bp.get("/tournament/archetypes/clusters")
def get_tournament_archetype_clusters():
    try:
        summary = route_deps().get_archetype_cluster_summary()
    except FileNotFoundError:
        return _json_error("Tournament archetype cluster summary has not been generated yet.", 404)
    except (OSError, json.JSONDecodeError):
        return _json_error("Tournament archetype cluster summary could not be loaded.", 500)

    return jsonify({
        "totalTeamsAnalyzed": summary.get("totalTeamsAnalyzed", 0),
        "totalClusters": summary.get("clusterCount", 0),
        "clustersByParentArchetype": _group_clusters_by_parent(summary),
        "dataQualityNotes": _extract_cluster_quality_notes(summary),
    })


@tournament_bp.post("/tournament/archetypes/classify")
def classify_tournament_archetype():
    payload = request.get_json(silent=True)
    if payload is None:
        return _json_error("A JSON request body is required.", 400)

    pokemon = payload.get("pokemon") or []
    if not isinstance(pokemon, list) or not pokemon:
        return _json_error("At least one Pokemon name is required.", 400)

    pokemon_lookup = route_deps().load_pokemon_stat_lookup()
    known_pokemon = [name for name in pokemon if route_deps().load_pokemon_stat_lookup().get(route_deps().normalize_name(name))]
    if not known_pokemon:
        return _json_error("None of the provided Pokemon names were recognized.", 400)

    classification = route_deps().classify_team_archetype(payload)

    try:
        matching_cluster = route_deps().find_matching_archetype_cluster(payload)
    except FileNotFoundError:
        matching_cluster = {
            "warning": "Archetype cluster summary has not been generated yet."
        }
    except (OSError, json.JSONDecodeError):
        matching_cluster = {
            "warning": "Archetype cluster summary could not be loaded."
        }

    try:
        rule_summary = route_deps().get_archetype_rule_summary()
        archetype_summaries = rule_summary.get("archetypes", [])
        archetype_summary = next(
            (
                entry
                for entry in archetype_summaries
                if entry.get("archetype") == classification.get("primaryArchetype")
            ),
            {},
        )
    except FileNotFoundError:
        archetype_summary = {}
    except (OSError, json.JSONDecodeError):
        archetype_summary = {}

    team_name_set = {route_deps().normalize_name(name) for name in pokemon if name}
    common_pokemon = archetype_summary.get("mostCommonPokemon", [])[:8]
    common_moves = archetype_summary.get("mostCommonMoves", [])[:8]
    common_items = archetype_summary.get("mostCommonItems", [])[:8]
    common_threats = [
        entry
        for entry in common_pokemon
        if route_deps().normalize_name(entry.get("name")) not in team_name_set
    ][:6]

    data_quality_notes = list(classification.get("dataQualityNotes", []))
    unknown_pokemon = [
        name
        for name in pokemon
        if route_deps().normalize_name(name) not in pokemon_lookup
    ]
    if unknown_pokemon:
        data_quality_notes.append(f"Unknown Pokemon names: {', '.join(unknown_pokemon)}")

    return jsonify({
        "primaryArchetype": classification.get("primaryArchetype"),
        "secondaryArchetype": classification.get("secondaryArchetype"),
        "displayLabel": classification.get("displayLabel"),
        "shortExplanation": classification.get("shortExplanation"),
        "detailedReasons": classification.get("detailedReasons"),
        "confidence": classification.get("confidence"),
        "confidenceLabel": classification.get("confidenceLabel"),
        "scores": classification.get("scores"),
        "reasons": classification.get("reasons"),
        "matchingCluster": matching_cluster,
        "commonPokemon": common_pokemon,
        "commonMoves": common_moves,
        "commonItems": common_items,
        "commonThreats": common_threats,
        "dataQualityNotes": data_quality_notes,
    })
