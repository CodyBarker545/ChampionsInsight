from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BattlePokemon:
    name: str
    level: int = 50
    types: list[str] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    max_hp: int = 1
    current_hp: int | None = None
    ability: str = ""
    item: str = ""
    status: str = ""
    nature: str = ""
    base_stats: dict[str, int] = field(default_factory=dict)
    ivs: dict[str, int] = field(default_factory=dict)
    evs: dict[str, int] = field(default_factory=dict)
    boosts: dict[str, int] = field(default_factory=dict)
    gender: str = ""
    is_grounded: bool = True
    has_acted: bool = False

    # Gen 9 mechanics
    tera_type: str = ""
    is_terastallized: bool = False

    # Extra state used by competitive calculations.
    weight_kg: float | None = None
    happiness: int = 255
    flash_fire_active: bool = False
    ability_on: bool = False
    helping_hand_active: bool = False
    charge_active: bool = False
    used_item: bool = False
    ability_triggered: bool = False
    switched_in_this_turn: bool = False
    last_damage_taken: int = 0
    fainted_allies: int = 0  # Supreme Overlord, max 5.
    booster_energy_active: bool = False
    protosynthesis_stat: str = ""
    quark_drive_stat: str = ""
    ability_suppressed: bool = False
    item_suppressed: bool = False
    protected: bool = False
    substitute: bool = False
    metronome_turns: int = 0

    def hp(self) -> int:
        return self.current_hp if self.current_hp is not None else self.max_hp


@dataclass
class BattleMove:
    name: str
    type: str
    category: str
    power: int | None = None
    accuracy: int | None = None
    priority: int = 0
    hits: int | str = 1
    target: str = "normal"

    makes_contact: bool = False
    is_spread: bool = False
    is_sound: bool = False
    is_punch: bool = False
    is_bite: bool = False
    is_bullet: bool = False
    is_pulse: bool = False
    is_recoil: bool = False
    is_slicing: bool = False
    has_secondary_effect: bool = False
    ignores_protect: bool = False


@dataclass
class SideConditions:
    reflect: bool = False
    light_screen: bool = False
    aurora_veil: bool = False
    tailwind: bool = False
    friend_guard: bool = False
    helping_hand: bool = False
    battery: bool = False
    power_spot: bool = False
    flower_gift: bool = False
    steely_spirit: bool = False

    # Entry hazards on this side.
    stealth_rock: bool = False
    spikes_layers: int = 0
    toxic_spikes_layers: int = 0
    sticky_web: bool = False


@dataclass
class BattleField:
    weather: str = ""       # rain, sun, sand, snow, harsh-sun, heavy-rain
    terrain: str = ""       # electric, grassy, psychic, misty
    critical: bool = False
    critical_stage: int = 0
    is_doubles: bool = False
    gravity: bool = False
    trick_room: bool = False
    wonder_room: bool = False
    magic_room: bool = False
    charge: bool = False
    attacker_side: SideConditions = field(default_factory=SideConditions)
    defender_side: SideConditions = field(default_factory=SideConditions)


@dataclass
class DamageContext:
    move_type: str = ""
    original_move_type: str = ""
    category: str = ""
    base_power: int = 0
    attack_stat: int = 1
    defense_stat: int = 1
    type_multiplier: float = 1.0
    stab: float = 1.0
    is_critical: bool = False
    is_super_effective: bool = False
    is_not_very_effective: bool = False
    notes: list[str] = field(default_factory=list)
    flags: dict[str, Any] = field(default_factory=dict)
