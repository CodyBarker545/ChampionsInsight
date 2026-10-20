"""Tests battle advice chat behavior for pure mechanics questions."""

from __future__ import annotations

from services import battle_advice_chat_service
from services.battle_question_router import route_battle_question
import pytest


@pytest.mark.parametrize("question, expected", [
    ("what does para do", ["Paralysis", "50%", "25%"]),
    ("choice band", ["Choice Band", "Attack by 50%", "Locks"]),
    ("what is choice scarf", ["Choice Scarf", "Speed by 50%"]),
    ("av", ["Assault Vest", "Special Defense", "status moves"]),
    ("what does brn do", ["Burn", "1/16"]),
    ("confusion", ["Confusion", "1–4 turns"]),
    ("STAB", ["Same-Type Attack Bonus", "1.5"]),
    ("sitrus berry", ["Sitrus Berry", "50%", "25%", "consumed"]),
])
def test_short_mechanics_questions_use_local_effects(monkeypatch, question, expected):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("Simple mechanics must not load team predictions or an LLM")

    monkeypatch.setattr(battle_advice_chat_service, "build_battle_context", unexpected)
    monkeypatch.setattr(battle_advice_chat_service, "_call_llm", unexpected)
    result = battle_advice_chat_service.answer_battle_question(question)
    assert result["questionType"] == ["gameplay_mechanic"]
    for phrase in expected:
        assert phrase in result["answer"]
    assert result["mechanicsUsed"]
    assert result["keyPoints"] == []


def test_response_lists_do_not_leak_across_questions():
    first = battle_advice_chat_service.answer_battle_question("Explain Trick Room")
    assert first["recommendedActions"]
    second = battle_advice_chat_service.answer_battle_question("choice band")
    assert not any("setup turns" in action for action in second["recommendedActions"])
    assert battle_advice_chat_service.DEFAULT_RESPONSE["recommendedActions"] == []


def test_choice_band_includes_battle_use_and_switching():
    result = battle_advice_chat_service.answer_battle_question("choice band")
    assert "physical attacker" in result["answer"]
    assert any("switching out resets" in action.lower() for action in result["recommendedActions"])


def test_unknown_mechanic_gives_specific_recovery_guidance(monkeypatch):
    monkeypatch.setattr(battle_advice_chat_service, "retrieve_gameplay_mechanics_context", lambda _q: {"answer": "", "notes": []})
    monkeypatch.setattr(battle_advice_chat_service, "_llm_config", lambda: None)
    result = battle_advice_chat_service.answer_battle_question("what is mystery orb")
    assert "full move, item, ability, or status name" in result["answer"]


def test_pokemon_names_do_not_trigger_substring_categories():
    assert battle_advice_chat_service.classify_battle_question("what does burn do to Kommo-o") == ["gameplay_mechanic"]
    assert "likely_sets" not in battle_advice_chat_service.classify_battle_question("what does psychic seed do")


@pytest.mark.parametrize("question, intent", [
    ("sitrus berry", "battle_info"),
    ("mystery berry", "battle_info"),
    ("what does burn do to my Garchomp", "battle_info"),
    ("Explain speed", "battle_info"),
    ("What is damage", "battle_info"),
    ("Explain Trick Room", "battle_info"),
    ("What should I lead?", "strategy"),
    ("How do I beat Trick Room?", "strategy"),
    ("What are the biggest threats?", "strategy"),
    ("What is this team trying to do?", "strategy"),
    ("What item is Sneasler likely holding?", "strategy"),
    ("Should I use Sitrus Berry on Pelipper?", "strategy"),
    ("What is Sitrus Berry and should I use it?", "strategy"),
    ("Can Garchomp outspeed Pelipper?", "strategy"),
    ("Is Sitrus Berry good on Pelipper?", "strategy"),
    ("Which berry should Pelipper hold?", "strategy"),
])
def test_intent_gate_separates_lookups_from_decisions(question, intent):
    assert route_battle_question(question) == intent


def test_sitrus_lookup_ignores_attached_teams_and_calculators(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("Lookup must not invoke strategy or load team snapshots")

    for name in ("build_battle_context", "_pokemon_snapshot", "_call_llm"):
        monkeypatch.setattr(battle_advice_chat_service, name, unexpected)
    result = battle_advice_chat_service.answer_battle_question(
        "sitrus berry", user_team=[{"name": "Pelipper"}],
        opponent_team=[{"name": "Tyranitar"}], damage_context={"range": "90-100%"},
    )
    assert result["intent"] == "battle_info"
    assert "25%" in result["answer"]
    assert result["relevantPokemon"] == []
    assert result["uncertaintyNotes"] == []
    assert "lead" not in str(result).lower()


def test_failed_lookup_never_falls_through_to_strategy(monkeypatch):
    def unexpected(*_args, **_kwargs):
        raise AssertionError("Failed lookups must remain lookups")

    monkeypatch.setattr(battle_advice_chat_service, "retrieve_gameplay_mechanics_context", lambda _q: {"answer": ""})
    monkeypatch.setattr(battle_advice_chat_service, "build_battle_context", unexpected)
    monkeypatch.setattr(battle_advice_chat_service, "_call_llm", unexpected)
    result = battle_advice_chat_service.answer_battle_question("mystery berry", user_team=["Pelipper"])
    assert result["intent"] == "battle_info"
    assert "could not find" in result["answer"]
    assert "lead" not in str(result).lower()


def test_generic_berry_excerpt_cannot_answer_an_unknown_named_berry(monkeypatch):
    monkeypatch.setattr(battle_advice_chat_service, "retrieve_rag_context", lambda _q: {
        "chunks": [(1.0, {"text": "Berries are held items that restore HP under certain conditions."})],
    })
    result = battle_advice_chat_service.answer_battle_question("mystery berry")
    assert result["intent"] == "battle_info"
    assert "could not find" in result["answer"]


def test_strategy_still_uses_team_context(monkeypatch):
    monkeypatch.setattr(battle_advice_chat_service, "_llm_config", lambda: None)
    monkeypatch.setattr(battle_advice_chat_service, "build_battle_context", lambda **kwargs: {
        "counterTeamReport": {"recommendedLeads": [{"name": "Pelipper", "reason": "offers useful type pressure"}]},
    })
    result = battle_advice_chat_service.answer_battle_question("What should I lead?", user_team=["Pelipper"])
    assert result["intent"] == "strategy"
    assert "Pelipper" in result["answer"]


def test_unknown_build_choice_does_not_turn_into_lead_advice(monkeypatch):
    monkeypatch.setattr(battle_advice_chat_service, "_llm_config", lambda: None)
    monkeypatch.setattr(battle_advice_chat_service, "build_battle_context", lambda **kwargs: {
        "counterTeamReport": {"recommendedLeads": [{"name": "Pelipper", "reason": "offers type pressure"}]},
    })
    result = battle_advice_chat_service.answer_battle_question("should I use mystery berry")
    assert result["intent"] == "strategy"
    assert "lead" not in result["answer"]
    assert "tradeoff" in result["recommendedActions"][0]


def test_generate_rule_based_answer_for_burn_avoids_tournament_uncertainty():
    battle_context = {
        "opponentTeam": [],
        "userTeam": [],
        "opponentArchetype": {},
        "matchingCluster": {},
        "likelySets": {},
        "counterTeamReport": {},
        "metaTrends": {},
        "commonTeammates": {},
        "commonCores": {},
        "damageContext": {},
        "speedContext": {},
        "dataQualityNotes": [
            "Low sample size. Treat these trend shifts as directional rather than definitive."
        ],
    }
    mechanics_context = {
        "available": True,
        "question": "what does the burn status do",
        "answer": "Burn deals damage each turn and reduces physical Attack by 50%.",
        "source": "status_conditions.txt",
        "notes": [],
    }

    result = battle_advice_chat_service.generate_rule_based_answer(
        "what does the burn status do",
        battle_context,
        mechanics_context,
    )

    assert result["answer"].startswith("Burn deals damage each turn")
    assert result["mechanicsUsed"] == ["status_conditions.txt"]
    assert result["uncertaintyNotes"] == []


def test_answer_battle_question_skips_heavy_battle_context_for_pure_mechanics(monkeypatch):
    def fail_if_called(**_kwargs):
        raise AssertionError("build_battle_context should not run for pure mechanics questions")

    monkeypatch.setattr(
        battle_advice_chat_service,
        "build_battle_context",
        fail_if_called,
    )
    monkeypatch.setattr(
        battle_advice_chat_service,
        "retrieve_gameplay_mechanics_context",
        lambda question: {
            "available": True,
            "question": question,
            "answer": "Burn deals damage each turn and reduces physical Attack by 50%.",
            "source": "status_conditions.txt",
            "notes": [],
        },
    )

    result = battle_advice_chat_service.answer_battle_question(
        "what does the burn status do"
    )

    assert result["answer"].startswith("Burn deals damage each turn")
    assert result["questionType"] == ["gameplay_mechanic"]


def test_answer_battle_question_answers_sleep_without_external_rag(monkeypatch):
    def fail_if_called(_question):
        raise AssertionError("external/vector RAG should not run for built-in sleep mechanics")

    monkeypatch.setattr(
        battle_advice_chat_service,
        "answer_rag_question",
        fail_if_called,
    )

    result = battle_advice_chat_service.answer_battle_question("what does sleep do")

    assert result["answer"].startswith("Sleep prevents")
    assert result["source"] == "rule_based"
    assert result["questionType"] == ["gameplay_mechanic"]


def test_retrieve_gameplay_mechanics_context_uses_shared_rag_context(monkeypatch):
    monkeypatch.setattr(
        battle_advice_chat_service,
        "retrieve_rag_context",
        lambda question: {
            "question": question,
            "chunks": [
                (
                    1.0,
                    {
                        "text": "Weather changes can boost attacks and activate abilities.",
                        "metadata": {"filename": "weather.txt", "chunk_index": 1},
                    },
                )
            ],
            "source": "weather.txt",
            "retrievalMode": "vector_store",
            "notes": [],
        },
    )

    result = battle_advice_chat_service.retrieve_gameplay_mechanics_context(
        "how does weather work"
    )

    assert result["available"] is True
    assert result["source"] == "weather.txt"
    assert result["retrievalMode"] == "vector_store"


def test_meta_question_can_use_pokemon_name_without_team_context(monkeypatch):
    monkeypatch.setattr(
        battle_advice_chat_service,
        "get_pokedex_grid_entries",
        lambda: [{"name": "Garchomp", "speciesName": "Garchomp", "formApiName": "garchomp"}],
    )
    monkeypatch.setattr(
        battle_advice_chat_service,
        "get_pokemon_meta_intelligence",
        lambda name: {
            "pokemon": name,
            "usageTrend": {
                "trendDirection": "rising",
                "currentUsageRate": 0.123,
                "explanation": f"{name} usage increased.",
            },
            "dataQualityNotes": ["Low sample size."],
        },
    )

    result = battle_advice_chat_service.answer_battle_question(
        "what is garchomp doing in the meta"
    )

    assert "Garchomp is currently rising" in result["answer"]
    assert "12.3% recent usage" in result["answer"]
    assert "Low sample size." in result["uncertaintyNotes"]
