"""Python bridge to the bundled Smogon JavaScript damage calculator."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .backend_process import DAMAGE_CALC_URL, start_damage_calc_backend


def _validate_payload(payload: dict) -> None:
    if not payload.get("attacker"):
        raise ValueError("Attacker is required.")
    if not payload.get("defender"):
        raise ValueError("Defender is required.")
    if not payload.get("move"):
        raise ValueError("Move is required.")


def _compare_speed(payload: dict) -> dict:
    attacker_speed = int((payload.get("attacker", {}).get("stats") or {}).get("speed", 0) or 0)
    defender_speed = int((payload.get("defender", {}).get("stats") or {}).get("speed", 0) or 0)
    field = payload.get("field") or {}
    attacker_side = field.get("attackerSide", field.get("attacker_side", {})) or {}
    defender_side = field.get("defenderSide", field.get("defender_side", {})) or {}
    trick_room = bool(field.get("trickRoom", field.get("trick_room", False)))

    if attacker_side.get("tailwind"):
        attacker_speed *= 2
    if defender_side.get("tailwind"):
        defender_speed *= 2

    if attacker_speed == defender_speed:
        result = "Speed tie"
    elif (attacker_speed > defender_speed) != trick_room:
        result = "Attacker is faster"
    else:
        result = "Defender is faster"

    return {
        "result": result,
        "attackerSpeed": attacker_speed,
        "defenderSpeed": defender_speed,
    }


def _request_damage(payload: dict) -> dict:
    start_damage_calc_backend()
    request = Request(
        f"{DAMAGE_CALC_URL}/calculate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        body = error.read().decode("utf-8")
        try:
            message = json.loads(body).get("error", body)
        except ValueError:
            message = body
        raise ValueError(message) from error
    except URLError as error:
        raise RuntimeError("Could not reach the Smogon damage calculator backend.") from error


def analyze_battle(payload: dict) -> dict:
    """Calculates one move using Smogon's JavaScript engine."""
    _validate_payload(payload)
    damage = _request_damage(payload)
    damage["range"] = f"{damage['minDamage']} - {damage['maxDamage']}"

    return {
        "attacker": payload["attacker"]["name"],
        "defender": payload["defender"]["name"],
        "move": payload["move"]["name"] if isinstance(payload["move"], dict) else payload["move"],
        "speed": _compare_speed(payload),
        "damage": damage,
    }
