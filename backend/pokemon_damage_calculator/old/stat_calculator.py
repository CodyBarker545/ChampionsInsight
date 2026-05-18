from __future__ import annotations

from math import floor
from .utils import normalize_name

STAT_NAMES = ["hp", "attack", "defense", "special_attack", "special_defense", "speed"]

NATURES: dict[str, tuple[str, str]] = {
    "hardy": ("", ""), "docile": ("", ""), "bashful": ("", ""), "quirky": ("", ""), "serious": ("", ""),
    "lonely": ("attack", "defense"),
    "brave": ("attack", "speed"),
    "adamant": ("attack", "special_attack"),
    "naughty": ("attack", "special_defense"),
    "bold": ("defense", "attack"),
    "relaxed": ("defense", "speed"),
    "impish": ("defense", "special_attack"),
    "lax": ("defense", "special_defense"),
    "timid": ("speed", "attack"),
    "hasty": ("speed", "defense"),
    "jolly": ("speed", "special_attack"),
    "naive": ("speed", "special_defense"),
    "modest": ("special_attack", "attack"),
    "mild": ("special_attack", "defense"),
    "quiet": ("special_attack", "speed"),
    "rash": ("special_attack", "special_defense"),
    "calm": ("special_defense", "attack"),
    "gentle": ("special_defense", "defense"),
    "sassy": ("special_defense", "speed"),
    "careful": ("special_defense", "special_attack"),
}


def nature_multiplier(stat_name: str, nature: str) -> float:
    # Keep internal stat keys as snake_case: special_attack, special_defense.
    stat_name = str(stat_name or "").strip().lower().replace("-", "_")
    increased, decreased = NATURES.get(normalize_name(nature), ("", ""))
    if stat_name == increased:
        return 1.1
    if stat_name == decreased:
        return 0.9
    return 1.0


def calculate_stat(base: int, iv: int, ev: int, level: int, stat_name: str, nature: str) -> int:
    stat_name = str(stat_name or "").strip().lower().replace("-", "_")
    base = int(base)
    iv = int(iv)
    ev = int(ev)
    level = int(level)

    if stat_name == "hp":
        return floor(((2 * base + iv + floor(ev / 4)) * level) / 100) + level + 10

    value = floor(((2 * base + iv + floor(ev / 4)) * level) / 100) + 5
    return max(1, floor(value * nature_multiplier(stat_name, nature)))


def calculate_all_stats(base_stats: dict, ivs: dict | None, evs: dict | None, level: int, nature: str) -> dict[str, int]:
    ivs = ivs or {}
    evs = evs or {}
    stats: dict[str, int] = {}
    for stat_name in STAT_NAMES:
        stats[stat_name] = calculate_stat(
            base=int(base_stats.get(stat_name, 1)),
            iv=int(ivs.get(stat_name, 31)),
            ev=int(evs.get(stat_name, 0)),
            level=level,
            stat_name=stat_name,
            nature=nature,
        )
    return stats


def highest_non_hp_stat(stats: dict[str, int]) -> str:
    ordered = ["attack", "defense", "special_attack", "special_defense", "speed"]
    return max(ordered, key=lambda stat: (int(stats.get(stat, 0)), -ordered.index(stat)))
