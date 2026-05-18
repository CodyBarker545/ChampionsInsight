"""Explainable rule-based archetype scoring built on top of feature extraction."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from paths import ARCHETYPE_RULE_SUMMARY_PATH
from services.archetype_feature_service import (
    build_feature_records,
    build_feature_records as _build_feature_records,
    extract_team_features,
    load_tournament_teams,
    normalize_name,
)


ARCHETYPE_NAMES = [
    "Rain",
    "Sun",
    "Sand",
    "Snow",
    "Trick Room",
    "Tailwind Offense",
    "Hyper Offense",
    "Balance",
    "Stall or Control",
    "Bulky Setup",
    "Semi-Room",
    "Weather Offense",
]

RAIN_SETTERS = {"pelipper", "politoed"}
SUN_SETTERS = {"torkoal", "ninetales"}
SAND_SETTERS = {"tyranitar", "hippowdon"}
SNOW_SETTERS = {"abomasnow", "ninetales alola"}

RAIN_ABILITIES = {"drizzle", "swift swim"}
SUN_ABILITIES = {"drought", "chlorophyll", "protosynthesis"}
SAND_ABILITIES = {"sand stream", "sand rush"}
SNOW_ABILITIES = {"snow warning", "slush rush"}

TRICK_ROOM_SETTERS = {
    "cresselia",
    "porygon2",
    "hatterene",
    "indeedee",
    "indeedee f",
    "farigiraf",
    "dusclops",
    "bronzong",
    "oranguru",
    "mimikyu",
    "diancie",
}

TAILWIND_SETTERS = {
    "whimsicott",
    "tornadus",
    "talonflame",
    "murkrow",
    "aerodactyl",
    "pelipper",
    "zapdos",
    "crobat",
    "mandibuzz",
    "kilowattrel",
}

RAIN_ABUSER_NAMES = {"archaludon", "dragonite", "basculegion", "basculegion m", "basculegion f", "floatzel", "barraskewda", "ludicolo"}
SUN_ABUSER_NAMES = {"walking wake", "flutter mane", "great tusk", "brute bonnet", "roaring moon", "gouging fire", "lilligant", "lilligant hisui"}
SAND_ABUSER_NAMES = {"excadrill", "garchomp", "lycanroc", "tyranitar", "houndstone"}
SNOW_ABUSER_NAMES = {"ninetales alola", "baxcalibur", "cetitan", "froslass", "abomasnow"}



def _normalize_score(raw_score: float, max_score: float) -> float:
    if max_score <= 0:
        return 0.0
    return round(max(0.0, min(1.0, raw_score / max_score)), 3)


def _score_payload(raw_score: float, max_score: float, reasons: list[str]) -> dict[str, Any]:
    return {
        "rawScore": round(raw_score, 3),
        "score": _normalize_score(raw_score, max_score),
        "reasons": reasons,
    }


def _get_feature_value(features: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(features.get(key, default) or 0)
    except (TypeError, ValueError):
        return float(default)


def _get_name_set(features: dict[str, Any]) -> set[str]:
    return {normalize_name(name) for name in features.get("pokemon_names", []) if name}


def _get_type_distribution(features: dict[str, Any]) -> dict[str, int]:
    distribution = features.get("type_distribution", {}) or {}
    return {normalize_name(key): int(value or 0) for key, value in distribution.items()}


def _has_any(names: set[str], candidates: set[str]) -> bool:
    return bool(names & {normalize_name(candidate) for candidate in candidates})


def _count_matching_names(names: set[str], candidates: set[str]) -> int:
    return len(names & {normalize_name(candidate) for candidate in candidates})


def _missing_data_penalty(features: dict[str, Any]) -> float:
    notes = {normalize_name(note) for note in features.get("data_quality_notes", [])}
    penalty = 0.0
    if "missing pok mon stats" in notes or "missing pokemon stats" in notes:
        penalty += 0.2
    if "missing moves" in notes:
        penalty += 0.1
    if "missing items" in notes:
        penalty += 0.05
    if "missing abilities" in notes:
        penalty += 0.05
    if "missing win loss data" in notes:
        penalty += 0.05
    return min(0.35, penalty)


def _confidence_label(confidence: float) -> str:
    if confidence >= 0.7:
        return "High"
    if confidence >= 0.4:
        return "Medium"
    return "Low"


def _clean_note(note: str) -> str:
    raw = str(note or "").strip()
    if not raw:
        return ""
    return raw.replace("PokÃ©mon", "Pokémon")


def _join_reason_summary(reasons: list[str], limit: int = 2) -> str:
    selected = [str(reason).strip() for reason in reasons if str(reason).strip()][:limit]
    if not selected:
        return ""
    if len(selected) == 1:
        return selected[0]
    return f"{selected[0]} {selected[1]}"


def _derive_display_label(
    features: dict[str, Any],
    primary_name: str,
    secondary_name: str,
    scores: dict[str, dict[str, Any]],
) -> str:
    rain_score = float(scores.get("Rain", {}).get("score", 0))
    sun_score = float(scores.get("Sun", {}).get("score", 0))
    sand_score = float(scores.get("Sand", {}).get("score", 0))
    weather_offense_score = float(scores.get("Weather Offense", {}).get("score", 0))
    trick_room_score = float(scores.get("Trick Room", {}).get("score", 0))
    semi_room_score = float(scores.get("Semi-Room", {}).get("score", 0))
    tailwind_score = float(scores.get("Tailwind Offense", {}).get("score", 0))
    balance_score = float(scores.get("Balance", {}).get("score", 0))
    offense_score = max(
        float(scores.get("Hyper Offense", {}).get("score", 0)),
        float(scores.get("Tailwind Offense", {}).get("score", 0)),
        weather_offense_score,
    )
    bulky_setup_score = float(scores.get("Bulky Setup", {}).get("score", 0))
    control_score = float(scores.get("Stall or Control", {}).get("score", 0))
    fast_mode = _get_feature_value(features, "fast_pokemon_count") >= 2 or tailwind_score >= 0.5

    if weather_offense_score >= 0.7 and offense_score >= 0.7:
        if rain_score >= 0.7:
            return "Rain Offense"
        if sun_score >= 0.7 and trick_room_score >= 0.6:
            return "Sun Trick Room"
        if sand_score >= 0.7 and balance_score >= 0.6:
            return "Sand Balance"
        if sand_score >= 0.7:
            return "Sand Offense"
        return "Weather Offense"

    if sand_score >= 0.7 and balance_score >= 0.6:
        return "Sand Balance"
    if sand_score >= 0.7 and offense_score >= 0.6:
        return "Sand Offense"
    if rain_score >= 0.7 and offense_score >= 0.6:
        return "Rain Offense"
    if sun_score >= 0.7 and trick_room_score >= 0.6:
        return "Sun Trick Room"

    if trick_room_score >= 0.85 and not fast_mode:
        return "Hard Trick Room"
    if (
        ((semi_room_score >= 0.55 or trick_room_score >= 0.5) and fast_mode and trick_room_score >= 0.35)
        or (primary_name == "Semi-Room")
        or (primary_name == "Trick Room" and secondary_name == "Semi-Room")
    ):
        return "Semi-Room"

    if tailwind_score >= 0.65 and offense_score >= 0.6:
        return "Tailwind Offense"
    if bulky_setup_score >= 0.7:
        return "Bulky Setup"
    if control_score >= 0.6 and balance_score >= 0.55:
        return "Control Balance"
    if float(scores.get("Hyper Offense", {}).get("score", 0)) >= 0.7:
        return "Hyper Offense"

    return primary_name


def _build_explanations(
    features: dict[str, Any],
    primary_name: str,
    secondary_name: str,
    scores: dict[str, dict[str, Any]],
    confidence: float,
    data_quality_notes: list[str],
) -> dict[str, Any]:
    primary_reasons = list(scores.get(primary_name, {}).get("reasons", []))
    secondary_reasons = list(scores.get(secondary_name, {}).get("reasons", [])) if secondary_name else []
    confidence_label = _confidence_label(confidence)
    display_label = _derive_display_label(features, primary_name, secondary_name, scores)
    clean_notes = [_clean_note(note) for note in data_quality_notes if _clean_note(note)]

    primary_summary = _join_reason_summary(primary_reasons) or f"The team scored highest as {primary_name}."
    secondary_summary = (
        _join_reason_summary(secondary_reasons)
        if secondary_name and secondary_reasons
        else ""
    )
    missing_summary = (
        f"Missing data: {', '.join(clean_notes)}."
        if clean_notes
        else "The team had enough known data for a normal read."
    )
    confidence_summary = (
        f"{confidence_label} confidence based on score separation and available data."
    )

    short_explanation_parts = [f"{display_label} because {primary_summary.lower()}"]
    if secondary_name:
        if secondary_summary:
            short_explanation_parts.append(
                f"{secondary_name} was also considered because {secondary_summary.lower()}"
            )
        else:
            short_explanation_parts.append(f"{secondary_name} was the next closest mode")
    short_explanation_parts.append(confidence_summary.lower())
    if clean_notes:
        short_explanation_parts.append(missing_summary.lower())

    short_explanation = ". ".join(
        part.rstrip(".") for part in short_explanation_parts if part
    ) + "."

    return {
        "displayLabel": display_label,
        "shortExplanation": short_explanation,
        "detailedReasons": {
            "primary": {
                "archetype": primary_name,
                "summary": primary_summary,
                "reasons": primary_reasons,
            },
            "secondary": {
                "archetype": secondary_name,
                "summary": secondary_summary or (
                    f"{secondary_name} had meaningful overlap with the primary read."
                    if secondary_name
                    else ""
                ),
                "reasons": secondary_reasons,
            },
            "dataQuality": {
                "summary": missing_summary,
                "notes": clean_notes,
            },
            "confidence": {
                "label": confidence_label,
                "summary": confidence_summary,
                "value": confidence,
            },
        },
        "confidenceLabel": confidence_label,
    }


def score_rain(features: dict[str, Any]) -> dict[str, Any]:
    names = _get_name_set(features)
    types = _get_type_distribution(features)
    rain_payoff_moves = _get_feature_value(features, "rain_payoff_move_count")
    raw_score = 0.0
    reasons: list[str] = []

    has_named_setter = _has_any(names, RAIN_SETTERS)
    has_weather_setter = _get_feature_value(features, "weather_setter_count") >= 1
    rain_abuser_names = _count_matching_names(names, RAIN_ABUSER_NAMES)

    if has_named_setter:
        raw_score += 5
        reasons.append("Rain setter like Pelipper or Politoed is present.")
    if has_weather_setter:
        raw_score += 4
        reasons.append("A real weather-setting ability gives the team dependable rain turns.")
    if _get_feature_value(features, "weather_abuser_count") >= 1:
        raw_score += 4
        reasons.append("Rain-abusing ability support suggests strong rain payoff.")
    if rain_abuser_names >= 1:
        raw_score += 2
        reasons.append("Known rain attackers are present.")
    if rain_payoff_moves >= 1:
        raw_score += 2
        reasons.append("Rain payoff moves like Thunder, Hurricane, or Electro Shot are present.")
    if rain_payoff_moves >= 2:
        raw_score += 2
        reasons.append("Multiple rain payoff moves make the rain mode much clearer.")
    if _get_feature_value(features, "weather_move_count") >= 1:
        raw_score += 1
        reasons.append("Weather support moves are present.")
    if types.get("water", 0) >= 2:
        raw_score += 2
        reasons.append("Multiple Water types strengthen rain offense.")
    if (has_named_setter or has_weather_setter) and rain_payoff_moves >= 1:
        raw_score += 3
        reasons.append("The setter plus payoff package creates a clear rain identity.")

    return _score_payload(raw_score, 25, reasons)


def score_sun(features: dict[str, Any]) -> dict[str, Any]:
    names = _get_name_set(features)
    types = _get_type_distribution(features)
    sun_payoff_moves = _get_feature_value(features, "sun_payoff_move_count")
    raw_score = 0.0
    reasons: list[str] = []

    has_named_setter = _has_any(names, SUN_SETTERS)
    has_weather_setter = _get_feature_value(features, "weather_setter_count") >= 1

    if has_named_setter:
        raw_score += 5
        reasons.append("Sun setter like Torkoal or Ninetales is present.")
    if has_weather_setter:
        raw_score += 4
        reasons.append("A real weather-setting ability gives the team dependable sun turns.")
    if _get_feature_value(features, "weather_abuser_count") >= 1:
        raw_score += 4
        reasons.append("Sun-abusing abilities suggest a sun shell.")
    if _count_matching_names(names, SUN_ABUSER_NAMES) >= 1:
        raw_score += 2
        reasons.append("Known sun abusers are present.")
    if sun_payoff_moves >= 1:
        raw_score += 2
        reasons.append("Sun payoff moves like Solar Beam or Eruption are present.")
    if sun_payoff_moves >= 2:
        raw_score += 2
        reasons.append("Multiple sun payoff moves make the sun mode much clearer.")
    if _get_feature_value(features, "weather_move_count") >= 1:
        raw_score += 1
        reasons.append("Weather support moves are present.")
    if types.get("fire", 0) >= 1 and types.get("grass", 0) >= 1:
        raw_score += 2
        reasons.append("Fire/Grass offensive core fits sun.")
    if (has_named_setter or has_weather_setter) and (sun_payoff_moves >= 1 or _count_matching_names(names, SUN_ABUSER_NAMES) >= 1):
        raw_score += 3
        reasons.append("The setter plus payoff package creates a clear sun identity.")

    return _score_payload(raw_score, 25, reasons)


def score_sand(features: dict[str, Any]) -> dict[str, Any]:
    names = _get_name_set(features)
    types = _get_type_distribution(features)
    sand_payoff_moves = _get_feature_value(features, "sand_payoff_move_count")
    raw_score = 0.0
    reasons: list[str] = []

    has_named_setter = _has_any(names, SAND_SETTERS)
    has_weather_setter = _get_feature_value(features, "weather_setter_count") >= 1

    if has_named_setter:
        raw_score += 5
        reasons.append("Sand setter like Tyranitar or Hippowdon is present.")
    if has_weather_setter:
        raw_score += 4
        reasons.append("A real weather-setting ability gives the team dependable sand turns.")
    if _get_feature_value(features, "weather_abuser_count") >= 1:
        raw_score += 4
        reasons.append("Sand-abusing ability support is present.")
    if _count_matching_names(names, SAND_ABUSER_NAMES) >= 1:
        raw_score += 2
        reasons.append("Known sand abusers are present.")
    if sand_payoff_moves >= 1:
        raw_score += 1
        reasons.append("Sand-friendly moves are present.")
    if types.get("rock", 0) + types.get("ground", 0) + types.get("steel", 0) >= 3:
        raw_score += 3
        reasons.append("Rock/Ground/Steel density fits sand.")
    if (has_named_setter or has_weather_setter) and (_count_matching_names(names, SAND_ABUSER_NAMES) >= 1 or sand_payoff_moves >= 1):
        raw_score += 3
        reasons.append("The setter plus payoff package creates a clear sand identity.")

    return _score_payload(raw_score, 22, reasons)


def score_snow(features: dict[str, Any]) -> dict[str, Any]:
    names = _get_name_set(features)
    types = _get_type_distribution(features)
    snow_payoff_moves = _get_feature_value(features, "snow_payoff_move_count")
    raw_score = 0.0
    reasons: list[str] = []

    has_named_setter = _has_any(names, SNOW_SETTERS)
    has_weather_setter = _get_feature_value(features, "weather_setter_count") >= 1

    if has_named_setter:
        raw_score += 5
        reasons.append("Snow setter like Abomasnow or Alolan Ninetales is present.")
    if has_weather_setter:
        raw_score += 4
        reasons.append("A real weather-setting ability gives the team dependable snow turns.")
    if _get_feature_value(features, "weather_abuser_count") >= 1:
        raw_score += 4
        reasons.append("Snow-abusing ability support is present.")
    if _count_matching_names(names, SNOW_ABUSER_NAMES) >= 1:
        raw_score += 2
        reasons.append("Known snow abusers are present.")
    if snow_payoff_moves >= 1:
        raw_score += 2
        reasons.append("Snow payoff moves like Aurora Veil or Blizzard are present.")
    if types.get("ice", 0) >= 2:
        raw_score += 2
        reasons.append("Multiple Ice types fit snow offense.")
    if (has_named_setter or has_weather_setter) and (snow_payoff_moves >= 1 or _count_matching_names(names, SNOW_ABUSER_NAMES) >= 1):
        raw_score += 3
        reasons.append("The setter plus payoff package creates a clear snow identity.")

    return _score_payload(raw_score, 22, reasons)


def score_trick_room(features: dict[str, Any]) -> dict[str, Any]:
    names = _get_name_set(features)
    average_speed = _get_feature_value(features, "average_speed")
    raw_score = 0.0
    reasons: list[str] = []

    if _get_feature_value(features, "trick_room_move_count") >= 1:
        raw_score += 5
        reasons.append("Trick Room move support is present.")
    if _has_any(names, TRICK_ROOM_SETTERS) or _get_feature_value(features, "trick_room_setter_count") >= 1:
        raw_score += 3
        reasons.append("Known Trick Room setter support is present.")
    if average_speed and average_speed <= 75:
        raw_score += 2
        reasons.append("Average Speed is low enough for Trick Room.")
    if _get_feature_value(features, "slow_pokemon_count") >= 3 and _get_feature_value(features, "bulky_pokemon_count") >= 2:
        raw_score += 2
        reasons.append("The team has slow bulky attackers for Trick Room turns.")
    if average_speed >= 95:
        raw_score -= 2
        reasons.append("Very high average Speed works against full Trick Room.")

    return _score_payload(raw_score, 12, reasons)


def score_tailwind_offense(features: dict[str, Any]) -> dict[str, Any]:
    names = _get_name_set(features)
    raw_score = 0.0
    reasons: list[str] = []

    if _get_feature_value(features, "tailwind_move_count") >= 1:
        raw_score += 4
        reasons.append("Tailwind move support is present.")
    if _has_any(names, TAILWIND_SETTERS) or _get_feature_value(features, "tailwind_setter_count") >= 1:
        raw_score += 2
        reasons.append("Known Tailwind setter support is present.")
    if _get_feature_value(features, "fast_pokemon_count") >= 3:
        raw_score += 2
        reasons.append("Several naturally fast attackers fit Tailwind offense.")
    if max(_get_feature_value(features, "average_attack"), _get_feature_value(features, "average_sp_attack")) >= 100:
        raw_score += 2
        reasons.append("Offensive stats are high enough to capitalize on Tailwind.")
    if _get_feature_value(features, "speed_control_count") >= 2:
        raw_score += 1
        reasons.append("Additional speed control reinforces the Tailwind mode.")

    return _score_payload(raw_score, 11, reasons)


def score_hyper_offense(features: dict[str, Any]) -> dict[str, Any]:
    raw_score = 0.0
    reasons: list[str] = []
    average_attack = _get_feature_value(features, "average_attack")
    average_sp_attack = _get_feature_value(features, "average_sp_attack")

    if _get_feature_value(features, "average_speed") >= 95:
        raw_score += 2
        reasons.append("Average Speed is high.")
    if max(average_attack, average_sp_attack) >= 100:
        raw_score += 2
        reasons.append("Average offensive stats are high.")
    if _get_feature_value(features, "setup_move_count") >= 2:
        raw_score += 2
        reasons.append("Multiple setup moves fit hyper offense.")
    if _get_feature_value(features, "focus_sash_count") >= 1:
        raw_score += 1
        reasons.append("Focus Sash appears on the team.")
    if _get_feature_value(features, "choice_item_count") >= 1 or _get_feature_value(features, "offensive_item_count") >= 2:
        raw_score += 1
        reasons.append("Choice or offensive item pressure is present.")
    if _get_feature_value(features, "recovery_move_count") >= 2 or _get_feature_value(features, "defensive_item_count") >= 2:
        raw_score -= 2
        reasons.append("Recovery or defensive item density pulls the team away from pure hyper offense.")

    return _score_payload(raw_score, 8, reasons)


def score_balance(features: dict[str, Any]) -> dict[str, Any]:
    raw_score = 0.0
    reasons: list[str] = []
    offense_count = (
        _get_feature_value(features, "physical_attacker_count")
        + _get_feature_value(features, "special_attacker_count")
        + _get_feature_value(features, "mixed_attacker_count")
    )

    if 1 <= _get_feature_value(features, "bulky_pokemon_count") <= 3:
        raw_score += 2
        reasons.append("The team has a healthy amount of bulk.")
    if 2 <= offense_count <= 4:
        raw_score += 2
        reasons.append("The team has meaningful but not excessive offensive pressure.")
    if _get_feature_value(features, "status_move_count") >= 1 or _get_feature_value(features, "speed_control_count") >= 1:
        raw_score += 2
        reasons.append("Utility, status, or speed control support is present.")
    if _get_feature_value(features, "physical_attacker_count") >= 1 and _get_feature_value(features, "special_attacker_count") >= 1:
        raw_score += 2
        reasons.append("Both physical and special offense are represented.")
    if _get_feature_value(features, "recovery_move_count") >= 4:
        raw_score -= 2
        reasons.append("Too much passive recovery pushes the team away from balance.")
    if _get_feature_value(features, "setup_move_count") >= 3 and _get_feature_value(features, "fast_pokemon_count") >= 4:
        raw_score -= 1
        reasons.append("The team leans a bit too hard into all-in offense.")

    return _score_payload(raw_score, 8, reasons)


def score_stall_control(features: dict[str, Any]) -> dict[str, Any]:
    raw_score = 0.0
    reasons: list[str] = []
    offensive_peak = max(_get_feature_value(features, "average_attack"), _get_feature_value(features, "average_sp_attack"))

    if _get_feature_value(features, "average_bulk") >= 270:
        raw_score += 3
        reasons.append("Average bulk is high.")
    if _get_feature_value(features, "recovery_move_count") >= 2:
        raw_score += 2
        reasons.append("Recovery support is present.")
    if _get_feature_value(features, "status_move_count") >= 2 or _get_feature_value(features, "speed_control_count") >= 2:
        raw_score += 3
        reasons.append("Status or control move density is high.")
    if offensive_peak <= 95:
        raw_score += 2
        reasons.append("Offensive pressure is modest enough for control play.")
    if _get_feature_value(features, "fast_pokemon_count") >= 3:
        raw_score -= 2
        reasons.append("Several fast sweepers work against a stall/control reading.")

    return _score_payload(raw_score, 10, reasons)


def score_bulky_setup(features: dict[str, Any]) -> dict[str, Any]:
    raw_score = 0.0
    reasons: list[str] = []

    if _get_feature_value(features, "bulky_pokemon_count") >= 2:
        raw_score += 3
        reasons.append("The team has enough bulk to support setup turns.")
    if _get_feature_value(features, "setup_move_count") >= 1:
        raw_score += 3
        reasons.append("Setup move support is present.")
    if _get_feature_value(features, "protect_count") >= 2 or _get_feature_value(features, "recovery_move_count") >= 1:
        raw_score += 2
        reasons.append("Protect or recovery helps create setup windows.")
    if max(_get_feature_value(features, "average_attack"), _get_feature_value(features, "average_sp_attack")) >= 95:
        raw_score += 2
        reasons.append("Offensive stats are high enough to convert setup into pressure.")

    return _score_payload(raw_score, 10, reasons)


def score_semi_room(features: dict[str, Any]) -> dict[str, Any]:
    raw_score = 0.0
    reasons: list[str] = []
    average_speed = _get_feature_value(features, "average_speed")

    if _get_feature_value(features, "trick_room_move_count") >= 1:
        raw_score += 3
        reasons.append("The team can access Trick Room.")
    if _get_feature_value(features, "slow_pokemon_count") >= 2 and _get_feature_value(features, "fast_pokemon_count") >= 1:
        raw_score += 3
        reasons.append("The team mixes slow and fast modes.")
    if 70 <= average_speed <= 95:
        raw_score += 2
        reasons.append("Average Speed fits a mixed-speed semi-room shell.")
    if _get_feature_value(features, "speed_control_count") >= 2:
        raw_score += 1
        reasons.append("Multiple speed modes support semi-room flexibility.")

    return _score_payload(raw_score, 9, reasons)


def score_weather_offense(features: dict[str, Any]) -> dict[str, Any]:
    raw_score = 0.0
    reasons: list[str] = []
    total_payoff_moves = (
        _get_feature_value(features, "rain_payoff_move_count")
        + _get_feature_value(features, "sun_payoff_move_count")
        + _get_feature_value(features, "sand_payoff_move_count")
        + _get_feature_value(features, "snow_payoff_move_count")
    )

    if _get_feature_value(features, "weather_setter_count") >= 1:
        raw_score += 5
        reasons.append("A true weather setter is present.")
    if _get_feature_value(features, "weather_abuser_count") >= 1:
        raw_score += 4
        reasons.append("The team includes weather abusers.")
    if max(_get_feature_value(features, "average_attack"), _get_feature_value(features, "average_sp_attack")) >= 100:
        raw_score += 2
        reasons.append("Offensive stats are high enough to exploit weather turns.")
    if total_payoff_moves >= 1:
        raw_score += 2
        reasons.append("Weather payoff moves are present.")
    if total_payoff_moves >= 2:
        raw_score += 2
        reasons.append("Multiple weather payoff moves reinforce the offensive plan.")
    if _get_feature_value(features, "weather_move_count") >= 1 or _get_feature_value(features, "offensive_item_count") >= 2:
        raw_score += 1
        reasons.append("Weather-compatible support or aggressive item pressure is present.")

    return _score_payload(raw_score, 16, reasons)


def _coerce_features(team_or_features: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(team_or_features, dict):
        raise TypeError("Team input must be a dictionary.")

    if "average_speed" in team_or_features and "pokemon_names" in team_or_features:
        return team_or_features

    if "pokemon" in team_or_features or "members" in team_or_features:
        return extract_team_features(team_or_features)

    raise ValueError("Input must be either a raw team payload or an extracted feature record.")


def score_all_archetypes(features: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "Rain": score_rain(features),
        "Sun": score_sun(features),
        "Sand": score_sand(features),
        "Snow": score_snow(features),
        "Trick Room": score_trick_room(features),
        "Tailwind Offense": score_tailwind_offense(features),
        "Hyper Offense": score_hyper_offense(features),
        "Balance": score_balance(features),
        "Stall or Control": score_stall_control(features),
        "Bulky Setup": score_bulky_setup(features),
        "Semi-Room": score_semi_room(features),
        "Weather Offense": score_weather_offense(features),
    }


def classify_team_archetype(team_or_features: dict[str, Any]) -> dict[str, Any]:
    features = _coerce_features(team_or_features)
    scores = score_all_archetypes(features)
    ranked_scores = sorted(scores.items(), key=lambda item: item[1]["score"], reverse=True)

    primary_name, primary_data = ranked_scores[0]
    secondary_name, secondary_data = ranked_scores[1]

    top_score = float(primary_data["score"])
    second_score = float(secondary_data["score"])
    score_gap = max(0.0, top_score - second_score)
    confidence = top_score * 0.7 + score_gap * 0.4
    confidence -= _missing_data_penalty(features)
    confidence = round(max(0.0, min(1.0, confidence)), 3)

    overall_reasons = list(primary_data["reasons"])
    if secondary_data["score"] >= 0.35:
        overall_reasons.extend([f"Secondary signal: {reason}" for reason in secondary_data["reasons"][:2]])
    else:
        secondary_name = ""

    data_quality_notes = list(features.get("data_quality_notes", []))
    explanations = _build_explanations(
        features,
        primary_name,
        secondary_name,
        scores,
        confidence,
        data_quality_notes,
    )

    return {
        "primaryArchetype": primary_name,
        "secondaryArchetype": secondary_name,
        "displayLabel": explanations["displayLabel"],
        "shortExplanation": explanations["shortExplanation"],
        "detailedReasons": explanations["detailedReasons"],
        "confidenceLabel": explanations["confidenceLabel"],
        "scores": scores,
        "confidence": confidence,
        "reasons": overall_reasons,
        "dataQualityNotes": data_quality_notes,
    }


def build_archetype_rule_summary() -> dict[str, Any]:
    feature_records = _build_feature_records()
    summary = {
        "teamCount": len(feature_records),
        "archetypes": {name: {"teamCount": 0, "averageScore": 0.0} for name in ARCHETYPE_NAMES},
    }

    for features in feature_records:
        classification = classify_team_archetype(features)
        for archetype_name, archetype_score in classification["scores"].items():
            summary["archetypes"][archetype_name]["averageScore"] += archetype_score["score"]
        summary["archetypes"][classification["primaryArchetype"]]["teamCount"] += 1

    for archetype_name, archetype_summary in summary["archetypes"].items():
        if feature_records:
            archetype_summary["averageScore"] = round(
                archetype_summary["averageScore"] / len(feature_records),
                4,
            )

    ARCHETYPE_RULE_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ARCHETYPE_RULE_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


@lru_cache(maxsize=1)
def get_archetype_rule_summary(force_rebuild: bool = False) -> dict[str, Any]:
    if force_rebuild or not ARCHETYPE_RULE_SUMMARY_PATH.exists():
        return build_archetype_rule_summary()
    return json.loads(ARCHETYPE_RULE_SUMMARY_PATH.read_text(encoding="utf-8"))
