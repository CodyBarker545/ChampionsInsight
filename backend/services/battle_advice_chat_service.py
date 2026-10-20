"""Battle advice chat orchestration for tournament, mechanics, and matchup questions."""

from __future__ import annotations

import copy
import json
import os
import re
import urllib.error
import urllib.request
from functools import lru_cache
from typing import Any

from services.archetype_cluster_service import find_matching_archetype_cluster
from services.archetype_rule_service import classify_team_archetype
from services.battle_advice_cache_service import predict_team_sets_from_cache
from services.battle_question_router import route_battle_question
from services.meta_trend_service import get_pokemon_meta_intelligence
from services.pokedex_service import get_pokedex_entry_by_name
from services.pokedex_service import get_pokedex_grid_entries
from services.pokemon_stats_service import get_level_50_stats
from services.rag_service import answer_from_retrieved_context, answer_rag_question, retrieve_rag_context
from services.rag_service import mechanic_sections, normalize_battle_query
from services.set_probability_service import predict_team_sets


QUESTION_CATEGORIES = [
    "opponent_team_strategy",
    "likely_sets",
    "counterplay",
    "damage_calc",
    "speed_calc",
    "gameplay_mechanic",
    "team_recommendation",
    "build_recommendation",
    "meta_trend",
    "general_strategy",
]

DEFAULT_RESPONSE = {
    "answer": "",
    "keyPoints": [],
    "recommendedActions": [],
    "relevantPokemon": [],
    "likelySetsUsed": [],
    "mechanicsUsed": [],
    "tournamentFactsUsed": [],
    "damageOrSpeedNotes": [],
    "uncertaintyNotes": [],
}

TYPE_CHART = {
    "normal": {"rock": 0.5, "ghost": 0.0, "steel": 0.5},
    "fire": {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 2.0, "bug": 2.0, "rock": 0.5, "dragon": 0.5, "steel": 2.0},
    "water": {"fire": 2.0, "water": 0.5, "grass": 0.5, "ground": 2.0, "rock": 2.0, "dragon": 0.5},
    "electric": {"water": 2.0, "electric": 0.5, "grass": 0.5, "ground": 0.0, "flying": 2.0, "dragon": 0.5},
    "grass": {"fire": 0.5, "water": 2.0, "grass": 0.5, "poison": 0.5, "ground": 2.0, "flying": 0.5, "bug": 0.5, "rock": 2.0, "dragon": 0.5, "steel": 0.5},
    "ice": {"fire": 0.5, "water": 0.5, "grass": 2.0, "ground": 2.0, "flying": 2.0, "dragon": 2.0, "steel": 0.5, "ice": 0.5},
    "fighting": {"normal": 2.0, "ice": 2.0, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "rock": 2.0, "ghost": 0.0, "dark": 2.0, "steel": 2.0, "fairy": 0.5},
    "poison": {"grass": 2.0, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0.0, "fairy": 2.0},
    "ground": {"fire": 2.0, "electric": 2.0, "grass": 0.5, "poison": 2.0, "flying": 0.0, "bug": 0.5, "rock": 2.0, "steel": 2.0},
    "flying": {"electric": 0.5, "grass": 2.0, "fighting": 2.0, "bug": 2.0, "rock": 0.5, "steel": 0.5},
    "psychic": {"fighting": 2.0, "poison": 2.0, "psychic": 0.5, "dark": 0.0, "steel": 0.5},
    "bug": {"fire": 0.5, "grass": 2.0, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2.0, "ghost": 0.5, "dark": 2.0, "steel": 0.5, "fairy": 0.5},
    "rock": {"fire": 2.0, "ice": 2.0, "fighting": 0.5, "ground": 0.5, "flying": 2.0, "bug": 2.0, "steel": 0.5},
    "ghost": {"normal": 0.0, "psychic": 2.0, "ghost": 2.0, "dark": 0.5},
    "dragon": {"dragon": 2.0, "steel": 0.5, "fairy": 0.0},
    "dark": {"fighting": 0.5, "psychic": 2.0, "ghost": 2.0, "dark": 0.5, "fairy": 0.5},
    "steel": {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2.0, "rock": 2.0, "steel": 0.5, "fairy": 2.0},
    "fairy": {"fire": 0.5, "fighting": 2.0, "poison": 0.5, "dragon": 2.0, "dark": 2.0, "steel": 0.5},
}

MECHANICS_REFERENCE = {
    "trick room": {
        "answer": "Trick Room reverses normal turn order for five turns, so slower Pokemon move before faster ones unless priority changes that order.",
        "source": "Built-in Mechanics Reference",
    },
    "tailwind": {
        "answer": "Tailwind doubles the Speed of the affected side for four turns, which can flip otherwise losing speed matchups without changing move priority.",
        "source": "Built-in Mechanics Reference",
    },
    "clear amulet": {
        "answer": "Clear Amulet blocks stat drops caused by opposing moves and abilities, but it does not stop self-inflicted drops like Close Combat lowering your own defenses.",
        "source": "Built-in Mechanics Reference",
    },
    "psychic seed": {
        "answer": "Psychic Seed is consumed when Psychic Terrain is active and raises the holder's Special Defense by one stage.",
        "source": "Built-in Mechanics Reference",
    },
    "sleep": {
        "answer": "Sleep prevents a Pokemon from choosing most moves while it is asleep. In modern competitive play, sleep lasts a random number of turns, the sleep counter only advances while that Pokemon is active, and switching out preserves the remaining sleep turns.",
        "source": "Built-in Mechanics Reference",
    },
    "spore": {
        "answer": "Spore is a 100% accurate status move that puts the target to sleep, but Grass-type Pokemon, Overcoat users, Safety Goggles holders, and Pokemon protected by effects such as Misty Terrain are immune.",
        "source": "Built-in Mechanics Reference",
    },
}


MECHANIC_BATTLE_TIPS = {
    "paralysis": "Account for halved Speed and the chance of losing a turn when planning your next move.",
    "burn": "Account for reduced physical damage when choosing an attack.",
    "choice band": "Use it on a physical attacker and plan switches around the move lock; switching out resets the lock.",
    "choice specs": "Use it on a special attacker and plan switches around the move lock; switching out resets the lock.",
    "choice scarf": "Check whether the Speed boost wins the matchup before committing to a move lock.",
    "assault vest": "Choose attacking moves for the holder; it cannot select status moves while using Assault Vest.",
}


def _display_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return " ".join(part.capitalize() for part in text.replace("-", " ").split())


def _normalize_name(value: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", str(value or "").lower()))


def _empty_response(source: str = "rule_based") -> dict[str, Any]:
    return {
        **copy.deepcopy(DEFAULT_RESPONSE),
        "source": source,
    }


def _extract_team_names(team: list[Any] | None) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for member in team or []:
        if isinstance(member, str):
            name = member.strip()
        elif isinstance(member, dict):
            name = str(member.get("name") or member.get("pokemon") or member.get("pokemonName") or "").strip()
        else:
            name = str(member or "").strip()
        if not name:
            continue
        normalized = _normalize_name(name)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        names.append(name)
    return names


def _pokemon_snapshot(name: str) -> dict[str, Any]:
    pokedex_entry = get_pokedex_entry_by_name(name)
    stat_entry = get_level_50_stats(name)
    resolved_name = (
        (pokedex_entry or {}).get("name")
        or (pokedex_entry or {}).get("speciesName")
        or (stat_entry or {}).get("name")
        or name
    )
    types = (pokedex_entry or {}).get("types") or (stat_entry or {}).get("types") or []
    base_stats = (pokedex_entry or {}).get("baseStats") or (stat_entry or {}).get("baseStats") or {}
    final_stats = (stat_entry or {}).get("finalStats") or {}
    return {
        "name": resolved_name,
        "known": bool(pokedex_entry or stat_entry),
        "types": types,
        "baseStats": base_stats,
        "finalStats": final_stats,
        "speed": int(
            final_stats.get("speed")
            or final_stats.get("special_speed", 0)
            or base_stats.get("speed", 0)
            or 0
        ),
    }


def _best_stab_multiplier(attacking_types: list[str], defending_types: list[str]) -> float:
    if not attacking_types or not defending_types:
        return 1.0
    best = 1.0
    for attacking_type in attacking_types:
        attack_key = _normalize_name(attacking_type)
        multiplier = 1.0
        for defending_type in defending_types:
            defend_key = _normalize_name(defending_type)
            multiplier *= TYPE_CHART.get(attack_key, {}).get(defend_key, 1.0)
        best = max(best, multiplier)
    return best


def _build_counter_team_report(user_team: list[dict[str, Any]], opponent_team: list[dict[str, Any]]) -> dict[str, Any]:
    if not user_team or not opponent_team:
        return {
            "recommendedLeads": [],
            "favorableMatchups": [],
            "threats": [],
            "notes": ["Counter-team analysis needs both a user team and an opponent team."],
        }

    lead_scores: list[dict[str, Any]] = []
    favorable_matchups: list[dict[str, Any]] = []
    threat_scores: dict[str, float] = {}

    for user_member in user_team:
        favorable_count = 0
        reasons: list[str] = []
        for opponent_member in opponent_team:
            user_speed = int(user_member.get("speed") or 0)
            opponent_speed = int(opponent_member.get("speed") or 0)
            user_offense = _best_stab_multiplier(user_member.get("types", []), opponent_member.get("types", []))
            opponent_offense = _best_stab_multiplier(opponent_member.get("types", []), user_member.get("types", []))
            speed_edge = user_speed - opponent_speed
            matchup_score = 0.0
            if speed_edge > 0:
                matchup_score += 1.0
            if user_offense > 1.0:
                matchup_score += user_offense
            if opponent_offense > 1.0:
                matchup_score -= opponent_offense
                threat_scores[opponent_member["name"]] = threat_scores.get(opponent_member["name"], 0.0) + opponent_offense
            if matchup_score > 0.75:
                favorable_count += 1
                if len(favorable_matchups) < 8:
                    favorable_matchups.append(
                        {
                            "userPokemon": user_member["name"],
                            "opponentPokemon": opponent_member["name"],
                            "userSpeed": user_speed,
                            "opponentSpeed": opponent_speed,
                            "userBestStabMultiplier": round(user_offense, 2),
                            "opponentBestStabMultiplier": round(opponent_offense, 2),
                        }
                    )
            if speed_edge > 0:
                reasons.append(f"outspeeds {opponent_member['name']}")
            elif user_offense > 1.0:
                reasons.append(f"has super-effective STAB into {opponent_member['name']}")

        lead_scores.append(
            {
                "name": user_member["name"],
                "score": favorable_count,
                "reason": ", ".join(reasons[:3]) if reasons else "offers limited obvious edges from current base data",
            }
        )

    lead_scores.sort(key=lambda entry: (entry["score"], entry["name"]), reverse=True)
    threats = [
        {"name": name, "pressureScore": round(score, 2)}
        for name, score in sorted(threat_scores.items(), key=lambda item: item[1], reverse=True)[:4]
    ]

    return {
        "recommendedLeads": lead_scores[:3],
        "favorableMatchups": favorable_matchups[:8],
        "threats": threats,
        "notes": [
            "Lead suggestions use current speed baselines and STAB type pressure, not full move-by-move simulations."
        ],
    }


def _build_team_payload(team_names: list[str], set_predictions: dict[str, Any] | None = None) -> dict[str, Any]:
    predictions = (set_predictions or {}).get("predictions", {})
    pokemon = []
    moves = []
    items = []
    abilities = []
    for team_name in team_names:
        prediction = predictions.get(team_name) or predictions.get(_display_name(team_name)) or {}
        pokemon.append(team_name)
        predicted_set = prediction.get("predictedSet", {})
        moves.append(predicted_set.get("moves", []))
        items.append(predicted_set.get("item", ""))
        abilities.append(predicted_set.get("ability", ""))
    return {
        "pokemon": pokemon,
        "moves": moves,
        "items": items,
        "abilities": abilities,
    }


def _predict_likely_sets(opponent_names: list[str]) -> dict[str, Any]:
    if not opponent_names:
        return {}

    try:
        return predict_team_sets_from_cache(opponent_names)
    except Exception:
        return predict_team_sets(opponent_names)


@lru_cache(maxsize=128)
def _build_battle_context_cached(
    opponent_names_key: tuple[str, ...],
    user_names_key: tuple[str, ...],
) -> dict[str, Any]:
    opponent_names = list(opponent_names_key)
    user_names = list(user_names_key)
    data_quality_notes: list[str] = []

    opponent_snapshots = [_pokemon_snapshot(name) for name in opponent_names]
    user_snapshots = [_pokemon_snapshot(name) for name in user_names]

    unknown_opponent = [entry["name"] for entry in opponent_snapshots if not entry["known"]]
    unknown_user = [entry["name"] for entry in user_snapshots if not entry["known"]]
    if unknown_opponent:
        data_quality_notes.append(f"Unknown opponent Pokemon: {', '.join(unknown_opponent)}.")
    if unknown_user:
        data_quality_notes.append(f"Unknown user Pokemon: {', '.join(unknown_user)}.")

    likely_sets: dict[str, Any] = {}
    opponent_archetype: dict[str, Any] = {}
    matching_cluster: dict[str, Any] = {}
    try:
        likely_sets = _predict_likely_sets(opponent_names) if opponent_names else {}
        data_quality_notes.extend(likely_sets.get("dataQualityNotes", []))
    except Exception:
        data_quality_notes.append("Tournament set probability data was unavailable.")

    if opponent_names:
        try:
            archetype_payload = _build_team_payload(opponent_names, likely_sets)
            opponent_archetype = classify_team_archetype(archetype_payload)
        except Exception:
            data_quality_notes.append("Opponent archetype classification was unavailable.")

        try:
            matching_cluster = find_matching_archetype_cluster(_build_team_payload(opponent_names, likely_sets)) or {}
        except Exception:
            data_quality_notes.append("Matching archetype cluster data was unavailable.")

    meta_trends: dict[str, Any] = {}
    common_teammates: dict[str, Any] = {}
    common_cores: dict[str, Any] = {}
    for opponent_name in opponent_names:
        prediction = (likely_sets.get("predictions") or {}).get(opponent_name) or {}
        if prediction:
            common_teammates[opponent_name] = (
                prediction.get("baseProbabilities", {}).get("commonTeammates", [])[:5]
            )
        try:
            meta_result = get_pokemon_meta_intelligence(opponent_name)
            meta_trends[opponent_name] = {
                "usageTrend": meta_result.get("usageTrend", {}),
                "itemTrends": meta_result.get("itemTrends", [])[:3],
                "moveTrends": meta_result.get("moveTrends", [])[:3],
                "commonCores": meta_result.get("commonCores", [])[:3],
                "dataQualityNotes": meta_result.get("dataQualityNotes", []),
            }
            common_cores[opponent_name] = meta_result.get("commonCores", [])[:3]
            data_quality_notes.extend(meta_result.get("dataQualityNotes", []))
        except Exception:
            data_quality_notes.append(f"Meta trend data was unavailable for {opponent_name}.")

    counter_team_report = _build_counter_team_report(user_snapshots, opponent_snapshots)
    data_quality_notes.extend(counter_team_report.get("notes", []))

    deduped_notes: list[str] = []
    seen_notes: set[str] = set()
    for note in data_quality_notes:
        clean_note = str(note or "").strip()
        if not clean_note or clean_note in seen_notes:
            continue
        seen_notes.add(clean_note)
        deduped_notes.append(clean_note)

    return {
        "opponentTeam": opponent_snapshots,
        "userTeam": user_snapshots,
        "opponentArchetype": opponent_archetype,
        "matchingCluster": matching_cluster,
        "likelySets": likely_sets,
        "counterTeamReport": counter_team_report,
        "metaTrends": meta_trends,
        "commonTeammates": common_teammates,
        "commonCores": common_cores,
        "damageContext": {},
        "speedContext": {},
        "dataQualityNotes": deduped_notes,
    }


def build_battle_context(opponent_team=None, user_team=None):
    opponent_names = tuple(_extract_team_names(opponent_team))
    user_names = tuple(_extract_team_names(user_team))
    return copy.deepcopy(_build_battle_context_cached(opponent_names, user_names))


def retrieve_gameplay_mechanics_context(question):
    cleaned_question = str(question or "").strip()
    if not cleaned_question:
        return {
            "available": False,
            "question": "",
            "answer": "",
            "source": "",
            "notes": ["A gameplay mechanics question is required."],
        }

    lowered_question = normalize_battle_query(cleaned_question)
    for keyword, reference in MECHANICS_REFERENCE.items():
        if re.search(rf"\b{re.escape(keyword)}\b", lowered_question):
            return {
                "available": True,
                "question": cleaned_question,
                "answer": reference["answer"],
                "source": reference["source"],
                "notes": [],
            }

    try:
        rag_context = retrieve_rag_context(cleaned_question)
        answer_text = answer_from_retrieved_context(cleaned_question, rag_context).strip()
        lowered_answer = answer_text.lower()
        primary_terms = [
            term
            for term in [
                "trick room",
                "tailwind",
                "weather",
                "terrain",
                "priority",
                "status",
                "damage",
                "speed",
            ]
            if term in lowered_question
        ]
        suspicious_answer = (
            "magic ball" in lowered_answer
            or "insufficient information" in lowered_answer
            or len(answer_text) > 1200
        )
        missing_primary_term = bool(primary_terms) and not any(term in lowered_answer for term in primary_terms)
        # A named item lookup must actually mention that item. A generic berry
        # paragraph is not evidence about an unknown named berry.
        named_berries = re.findall(r"\b([a-z]+ berry)\b", lowered_question)
        missing_named_item = any(
            name.split()[0] not in {"a", "the", "this", "that", "held"}
            and name not in lowered_answer
            for name in named_berries
        )
        if suspicious_answer or missing_primary_term or missing_named_item:
            return {
                "available": False,
                "question": cleaned_question,
                "answer": "",
                "source": "",
                "notes": ["Gameplay mechanics retrieval was unavailable."],
            }
        return {
            "available": True,
            "question": rag_context.get("question", cleaned_question),
            "answer": answer_text,
            "source": rag_context.get("source", "Local RAG Knowledge Base"),
            "retrievalMode": rag_context.get("retrievalMode", "local_documents"),
            "notes": [],
        }
    except Exception:
        return {
            "available": False,
            "question": cleaned_question,
            "answer": "",
            "source": "",
            "notes": ["Gameplay mechanics retrieval was unavailable."],
        }


def classify_battle_question(question):
    text = normalize_battle_query(question).strip()
    if route_battle_question(text) == "battle_info":
        return ["gameplay_mechanic"]
    categories: list[str] = []

    def add(category: str):
        if category not in categories:
            categories.append(category)

    def contains(phrases):
        return any(re.search(rf"\b{re.escape(phrase)}\b", text) for phrase in phrases)

    if mechanic_sections(text) or contains(MECHANICS_REFERENCE) or contains(["what does", "how does", "trick room", "tailwind", "terrain", "weather", "priority", "status", "clear amulet", "seed", "mechanic"]):
        add("gameplay_mechanic")
    if contains(["likely item", "likely items", "likely move", "likely moves", "likely holding", "likely to run", "what item", "what move", "likely ability", "likely set", "likely sets", "moveset", "common moves", "common items", "common abilities", "common sets"]):
        add("likely_sets")
    if contains(["trying to do", "archetype", "mode", "team strategy", "what is this team"]):
        add("opponent_team_strategy")
    if contains(["beat", "counter", "handle", "watch out", "threat", "threats", "biggest threat"]):
        add("counterplay")
    if contains(["damage", "ko", "calc", "roll", "ohko", "2hko", "3hko"]):
        add("damage_calc")
    if contains(["faster", "outspeed", "speed", "tailwind", "trick room"]):
        add("speed_calc")
    if contains(["what should i lead", "what should i bring", "which pokemon on my team", "recommend", "lead into"]):
        add("team_recommendation")
    if re.search(r"\b(?:should (?:i|my pokemon) (?:use|run|hold)|which (?:item|move|berry|ability)|best item|(?:is|are) .+ (?:good|better|worth|useful|viable))\b", text):
        add("build_recommendation")
    if contains(["meta", "trend", "popular", "common in tournaments"]):
        add("meta_trend")

    if not categories:
        if re.match(r"^(what is|what are|explain|tell me about)\b", text):
            add("gameplay_mechanic")
        else:
            add("general_strategy")

    return categories


def build_battle_advice_prompt(question, battle_context, mechanics_context):
    compact_context = {
        "question": question,
        "opponentTeam": battle_context.get("opponentTeam", []),
        "userTeam": battle_context.get("userTeam", []),
        "opponentArchetype": {
            "primaryArchetype": battle_context.get("opponentArchetype", {}).get("primaryArchetype"),
            "displayLabel": battle_context.get("opponentArchetype", {}).get("displayLabel"),
            "shortExplanation": battle_context.get("opponentArchetype", {}).get("shortExplanation"),
            "confidenceLabel": battle_context.get("opponentArchetype", {}).get("confidenceLabel"),
        },
        "matchingCluster": {
            "clusterId": battle_context.get("matchingCluster", {}).get("clusterId"),
            "suggestedLabel": battle_context.get("matchingCluster", {}).get("suggestedLabel"),
            "similarityScore": battle_context.get("matchingCluster", {}).get("similarityScore"),
        },
        "likelySets": battle_context.get("likelySets", {}).get("predictions", {}),
        "counterTeamReport": battle_context.get("counterTeamReport", {}),
        "metaTrends": battle_context.get("metaTrends", {}),
        "commonTeammates": battle_context.get("commonTeammates", {}),
        "commonCores": battle_context.get("commonCores", {}),
        "damageContext": battle_context.get("damageContext", {}),
        "speedContext": battle_context.get("speedContext", {}),
        "dataQualityNotes": battle_context.get("dataQualityNotes", []),
        "mechanicsContext": mechanics_context,
    }

    return (
        "You are ChampionsInsight Battle Advice Chat.\n"
        "Use tournament data when available.\n"
        "Use gameplay mechanics from the retrieved mechanics context when available.\n"
        "Do not make unsupported claims or invent tournament facts.\n"
        "Mention low sample size, missing data, and uncertainty clearly.\n"
        "Explain reasoning in user-friendly language and give practical battle advice.\n"
        "Return JSON only with this exact shape:\n"
        "{"
        '"answer":"","keyPoints":[],"recommendedActions":[],"relevantPokemon":[],"likelySetsUsed":[],"mechanicsUsed":[],"tournamentFactsUsed":[],"damageOrSpeedNotes":[],"uncertaintyNotes":[],"source":"ai"'
        "}\n"
        f"Context JSON:\n{json.dumps(compact_context, ensure_ascii=False)}"
    )


def _llm_config() -> dict[str, str] | None:
    if os.getenv("OPENAI_API_KEY"):
        return {
            "provider": "openai",
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions"),
        }
    if os.getenv("GEMINI_API_KEY"):
        return {
            "provider": "gemini",
            "api_key": os.getenv("GEMINI_API_KEY", ""),
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            "base_url": os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/models"),
        }
    return None


def _call_llm(prompt: str) -> str:
    config = _llm_config()
    if config is None:
        raise RuntimeError("No LLM provider is configured.")

    if config["provider"] == "openai":
        payload = {
            "model": config["model"],
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
        }
        request = urllib.request.Request(
            config["base_url"],
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {config['api_key']}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=25) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return (
            payload.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2},
    }
    request = urllib.request.Request(
        f"{config['base_url']}/{config['model']}:generateContent?key={config['api_key']}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=25) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return (
        payload.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [{}])[0]
        .get("text", "")
    )


def _parse_llm_json(raw_text: str) -> dict[str, Any]:
    cleaned = str(raw_text or "").strip()
    if not cleaned:
        raise ValueError("LLM returned an empty response.")

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            raise ValueError("LLM returned invalid JSON.") from None
        return json.loads(match.group(0))


def _mentioned_pokemon(question: str, battle_context: dict[str, Any]) -> list[str]:
    lowered_question = str(question or "").lower()
    names = []
    for group_name in ("opponentTeam", "userTeam"):
        for member in battle_context.get(group_name, []):
            name = str(member.get("name") or "").strip()
            if name and name.lower() in lowered_question and name not in names:
                names.append(name)
    for name in _pokemon_names_in_question(question):
        if name not in names:
            names.append(name)
    return names


def _pokemon_names_in_question(question: str) -> list[str]:
    lowered_question = f" {str(question or '').lower()} "
    names: list[str] = []
    seen: set[str] = set()
    try:
        entries = get_pokedex_grid_entries()
    except Exception:
        entries = []

    for entry in entries:
        display_name = str(entry.get("speciesName") or entry.get("name") or "").strip()
        candidates = {
            display_name,
            str(entry.get("name") or "").strip(),
            str(entry.get("formApiName") or "").replace("-", " ").strip(),
        }
        for candidate in candidates:
            normalized = _normalize_name(candidate)
            if not normalized or normalized in seen:
                continue
            if re.search(rf"(?<![a-z0-9']){re.escape(normalized)}(?![a-z0-9'])", lowered_question):
                seen.add(normalized)
                names.append(display_name or candidate)
                break

    return names


def _target_set_prediction(question: str, battle_context: dict[str, Any]) -> tuple[str, dict[str, Any]] | tuple[str, None]:
    mentioned = _mentioned_pokemon(question, battle_context)
    predictions = battle_context.get("likelySets", {}).get("predictions", {})
    for name in mentioned:
        if name in predictions:
            return name, predictions[name]
    for name, prediction in predictions.items():
        return name, prediction
    return "", None


def _build_speed_note_from_context(speed_context: dict[str, Any]) -> str:
    if not speed_context:
        return ""
    result = speed_context.get("result")
    left = speed_context.get("yourSpeed") or speed_context.get("user", {}).get("modifiedSpeed")
    right = speed_context.get("opponentSpeed") or speed_context.get("opponent", {}).get("modifiedSpeed")
    if result and left is not None and right is not None:
        return f"{result} ({left} vs {right})."
    if result:
        return str(result)
    return ""


def _empty_battle_context(
    opponent_team=None,
    user_team=None,
    damage_context=None,
    speed_context=None,
) -> dict[str, Any]:
    return {
        "opponentTeam": [_pokemon_snapshot(name) for name in _extract_team_names(opponent_team)],
        "userTeam": [_pokemon_snapshot(name) for name in _extract_team_names(user_team)],
        "opponentArchetype": {},
        "matchingCluster": {},
        "likelySets": {},
        "counterTeamReport": {
            "recommendedLeads": [],
            "favorableMatchups": [],
            "threats": [],
            "notes": [],
        },
        "metaTrends": {},
        "commonTeammates": {},
        "commonCores": {},
        "damageContext": damage_context or {},
        "speedContext": speed_context or {},
        "dataQualityNotes": [],
    }


def _is_pure_mechanics_query(question_type: list[str], question: str) -> bool:
    return route_battle_question(question) == "battle_info"


def generate_rule_based_answer(question, battle_context, mechanics_context):
    categories = classify_battle_question(question)
    response = _empty_response("rule_based")
    response["uncertaintyNotes"] = (
        []
        if _is_pure_mechanics_query(categories, question)
        else list(battle_context.get("dataQualityNotes", []))
    )
    response["relevantPokemon"] = [member["name"] for member in battle_context.get("opponentTeam", [])[:6]]

    opponent_archetype = battle_context.get("opponentArchetype", {})
    matching_cluster = battle_context.get("matchingCluster", {})
    lead_options = battle_context.get("counterTeamReport", {}).get("recommendedLeads", [])
    threats = battle_context.get("counterTeamReport", {}).get("threats", [])
    speed_context = battle_context.get("speedContext", {})
    damage_context = battle_context.get("damageContext", {})

    if "gameplay_mechanic" in categories and mechanics_context.get("answer"):
        response["answer"] = mechanics_context["answer"]
        response["mechanicsUsed"] = [mechanics_context.get("source", "Local RAG Knowledge Base")]
        # The answer already contains the effects; avoid repeating it as a list.
        normalized_question = normalize_battle_query(question)
        for mechanic, tip in MECHANIC_BATTLE_TIPS.items():
            if re.search(rf"\b{re.escape(mechanic)}\b", normalized_question):
                response["recommendedActions"].append(tip)
        if "trick room" in str(question).lower():
            response["recommendedActions"].append("Deny setup turns when possible and protect key fast attackers from losing the speed race.")
        if "tailwind" in str(question).lower():
            response["recommendedActions"].append("Track the Tailwind turn count so you know when speed control expires.")
        if _is_pure_mechanics_query(categories, question):
            return response

    if _is_pure_mechanics_query(categories, question) and not mechanics_context.get("answer"):
        response["answer"] = "I could not find a reliable mechanics entry for that question. Try the full move, item, ability, or status name."
        return response

    if "build_recommendation" in categories:
        response["answer"] = mechanics_context.get("answer") or "I need more information to judge that build choice."
        response["recommendedActions"].append("Specify the Pokemon, its moves, and the item or move you are comparing so I can assess the tradeoff.")
        response["uncertaintyNotes"].append("An item or move effect alone does not establish the best build for this matchup.")
        return response

    if "likely_sets" in categories:
        target_name, prediction = _target_set_prediction(question, battle_context)
        if prediction:
            predicted_set = prediction.get("predictedSet", {})
            item = predicted_set.get("item") or "Unknown item"
            moves = predicted_set.get("moves", [])[:4]
            sample_note = f"{prediction.get('matchedSampleSize', 0)} matched samples from {prediction.get('baseSampleSize', 0)} total appearances"
            response["answer"] = f"{target_name} most likely leans toward {item} with common moves like {', '.join(moves) if moves else 'unknown coverage'}."
            response["likelySetsUsed"] = [f"{target_name}: item {item}; moves {', '.join(moves) if moves else 'unknown'}"]
            response["tournamentFactsUsed"] = [sample_note]
            response["keyPoints"] = [
                f"Predicted item: {item}.",
                f"Confidence: {prediction.get('confidenceLabel', 'Low')}.",
            ]
            response["recommendedActions"].append("Scout the first revealed move or item interaction before committing to a hard read.")
            response["uncertaintyNotes"].extend(prediction.get("dataQualityNotes", []))
            return response

    if "speed_calc" in categories:
        speed_note = _build_speed_note_from_context(speed_context)
        if speed_note:
            response["answer"] = speed_note
            response["damageOrSpeedNotes"] = [speed_note]
            response["keyPoints"] = [speed_note]
            return response

        favorable = battle_context.get("counterTeamReport", {}).get("favorableMatchups", [])
        mentioned = _mentioned_pokemon(question, battle_context)
        for matchup in favorable:
            if len(mentioned) >= 2 and matchup["userSpeed"] > matchup["opponentSpeed"] and all(name in {matchup["userPokemon"], matchup["opponentPokemon"]} for name in mentioned[:2]):
                response["answer"] = (
                    f"{matchup['userPokemon']} is currently projected faster than {matchup['opponentPokemon']} "
                    f"({matchup['userSpeed']} vs {matchup['opponentSpeed']}) from base level-50 stats."
                )
                response["damageOrSpeedNotes"] = [response["answer"]]
                return response

    if "damage_calc" in categories and damage_context:
        range_text = (
            damage_context.get("range")
            or damage_context.get("damageRange")
            or damage_context.get("percentRange")
            or damage_context.get("percentDamageRange")
        )
        ko_text = damage_context.get("koChance") or damage_context.get("chanceToKo") or ""
        if range_text:
            response["answer"] = f"Current calculator context shows {range_text}{f' with {ko_text}' if ko_text else ''}."
            response["damageOrSpeedNotes"] = [response["answer"]]
            return response

    if "opponent_team_strategy" in categories:
        archetype_label = (
            opponent_archetype.get("displayLabel")
            or opponent_archetype.get("primaryArchetype")
            or "an unclear archetype"
        )
        cluster_label = matching_cluster.get("suggestedLabel") or ""
        cluster_text = f" It looks closest to the {cluster_label} cluster." if cluster_label else ""
        response["answer"] = f"This opponent team most likely plays like {archetype_label}.{cluster_text}"
        response["keyPoints"] = [
            opponent_archetype.get("shortExplanation") or "The read is based on the visible team shell and tournament patterns.",
        ]
        response["tournamentFactsUsed"] = [
            f"Archetype confidence: {opponent_archetype.get('confidenceLabel', 'Low')}.",
        ]
        if threats:
            response["recommendedActions"].append(
                f"Respect pressure from {', '.join(entry['name'] for entry in threats[:2])} first."
            )
        return response

    if "team_recommendation" in categories or "counterplay" in categories or "general_strategy" in categories:
        if lead_options:
            best_lead = lead_options[0]
            response["answer"] = f"{best_lead['name']} looks like the safest early lead from the current visible data."
            response["recommendedActions"] = [
                f"Consider leading {entry['name']} because it {entry['reason']}."
                for entry in lead_options[:3]
            ]
        else:
            response["answer"] = "I do not have enough grounded matchup context to recommend a confident lead yet."

        if threats:
            response["keyPoints"].append(
                f"Biggest visible threats: {', '.join(entry['name'] for entry in threats[:3])}."
            )
        if opponent_archetype.get("displayLabel"):
            response["keyPoints"].append(
                f"Opponent archetype read: {opponent_archetype.get('displayLabel')}."
            )
        return response

    if "meta_trend" in categories:
        mentioned = _mentioned_pokemon(question, battle_context)
        for name in mentioned:
            meta_payload = battle_context.get("metaTrends", {}).get(name, {})
            if not meta_payload:
                try:
                    meta_payload = get_pokemon_meta_intelligence(name)
                except Exception:
                    meta_payload = {}
            trend = meta_payload.get("usageTrend", {})
            if trend:
                response["answer"] = (
                    f"{name} is currently {trend.get('trendDirection', 'stable')} in the tournament sample "
                    f"with {round(float(trend.get('currentUsageRate', 0.0)) * 100, 1)}% recent usage."
                )
                response["tournamentFactsUsed"] = [trend.get("explanation", "Tournament trend data used.")]
                response["uncertaintyNotes"].extend(meta_payload.get("dataQualityNotes", []))
                return response

    if not response["answer"]:
        response["answer"] = "I could not ground a stronger answer from the current tournament, mechanics, or calculator context, so I would treat this as a low-confidence read."

    return response


def _clean_llm_payload(payload: dict[str, Any], source: str) -> dict[str, Any]:
    cleaned = _empty_response(source)
    for key in DEFAULT_RESPONSE:
        value = payload.get(key, DEFAULT_RESPONSE[key])
        cleaned[key] = value if isinstance(DEFAULT_RESPONSE[key], type(value)) or key == "answer" else DEFAULT_RESPONSE[key]
    cleaned["answer"] = str(payload.get("answer") or "").strip()
    for key in DEFAULT_RESPONSE:
        if isinstance(DEFAULT_RESPONSE[key], list):
            cleaned[key] = [str(item).strip() for item in payload.get(key, []) if str(item).strip()]
    cleaned["source"] = source
    return cleaned


def answer_battle_question(question, opponent_team=None, user_team=None, damage_context=None, speed_context=None):
    cleaned_question = str(question or "").strip()
    if not cleaned_question:
        raise ValueError("Question is required.")

    intent = route_battle_question(cleaned_question)
    question_type = classify_battle_question(cleaned_question)
    pure_mechanics_query = intent == "battle_info"
    mechanics_context = (
        retrieve_gameplay_mechanics_context(cleaned_question)
        if "gameplay_mechanic" in question_type or "speed_calc" in question_type
        else {"available": False, "question": "", "answer": "", "source": "", "notes": []}
    )
    if pure_mechanics_query:
        # Factual lookups never load teams, calculators, tournament predictions,
        # or the strategy LLM, even when retrieval has no usable answer.
        result = generate_rule_based_answer(cleaned_question, _empty_battle_context(), mechanics_context)
        result["questionType"] = question_type
        result["intent"] = intent
        return result
    else:
        battle_context = build_battle_context(opponent_team=opponent_team, user_team=user_team)
        battle_context["damageContext"] = damage_context or {}
        battle_context["speedContext"] = speed_context or {}

    try:
        if _llm_config() is None:
            raise RuntimeError("No LLM provider is configured.")

        prompt = build_battle_advice_prompt(cleaned_question, battle_context, mechanics_context)
        llm_text = _call_llm(prompt)
        llm_payload = _parse_llm_json(llm_text)
        result = _clean_llm_payload(llm_payload, "ai")
    except (RuntimeError, ValueError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        result = generate_rule_based_answer(cleaned_question, battle_context, mechanics_context)

    result["questionType"] = question_type
    result["intent"] = intent
    result["source"] = result.get("source", "rule_based")
    include_battle_uncertainty = not pure_mechanics_query
    result["uncertaintyNotes"] = [
        *result.get("uncertaintyNotes", []),
        *(battle_context.get("dataQualityNotes", []) if include_battle_uncertainty else []),
        *mechanics_context.get("notes", []),
    ]

    deduped_notes: list[str] = []
    seen: set[str] = set()
    for note in result["uncertaintyNotes"]:
        clean_note = str(note or "").strip()
        if not clean_note or clean_note in seen:
            continue
        seen.add(clean_note)
        deduped_notes.append(clean_note)
    result["uncertaintyNotes"] = deduped_notes

    return result
