from __future__ import annotations

from math import floor

from .models import BattleField, BattleMove, BattlePokemon, DamageContext
from .utils import normalize_name

TYPE_BOOST_ITEMS = {
    "silk-scarf": "normal", "charcoal": "fire", "mystic-water": "water", "miracle-seed": "grass",
    "magnet": "electric", "never-melt-ice": "ice", "black-belt": "fighting", "poison-barb": "poison",
    "soft-sand": "ground", "sharp-beak": "flying", "twisted-spoon": "psychic", "silver-powder": "bug",
    "hard-stone": "rock", "spell-tag": "ghost", "dragon-fang": "dragon", "black-glasses": "dark",
    "metal-coat": "steel", "pixie-plate": "fairy",
    "sea-incense": "water", "wave-incense": "water", "rose-incense": "grass", "odd-incense": "psychic",
    "rock-incense": "rock", "fairy-feather": "fairy",
    "flame-plate": "fire", "splash-plate": "water", "meadow-plate": "grass", "zap-plate": "electric",
    "icicle-plate": "ice", "fist-plate": "fighting", "toxic-plate": "poison", "earth-plate": "ground",
    "sky-plate": "flying", "mind-plate": "psychic", "insect-plate": "bug", "stone-plate": "rock",
    "spooky-plate": "ghost", "draco-plate": "dragon", "dread-plate": "dark", "iron-plate": "steel",
}
GEMS = {
    "normal-gem": "normal", "fire-gem": "fire", "water-gem": "water", "electric-gem": "electric",
    "grass-gem": "grass", "ice-gem": "ice", "fighting-gem": "fighting", "poison-gem": "poison",
    "ground-gem": "ground", "flying-gem": "flying", "psychic-gem": "psychic", "bug-gem": "bug",
    "rock-gem": "rock", "ghost-gem": "ghost", "dragon-gem": "dragon", "dark-gem": "dark",
    "steel-gem": "steel", "fairy-gem": "fairy",
}
SPECIES_TYPE_BOOST_ITEMS = {
    "adamant-orb": {"dialga": {"dragon", "steel"}},
    "adamant-crystal": {"dialga": {"dragon", "steel"}},
    "lustrous-orb": {"palkia": {"dragon", "water"}},
    "lustrous-globe": {"palkia": {"dragon", "water"}},
    "griseous-orb": {"giratina": {"dragon", "ghost"}},
    "griseous-core": {"giratina": {"dragon", "ghost"}},
    "soul-dew": {"latias": {"dragon", "psychic"}, "latios": {"dragon", "psychic"}},
}
RESIST_BERRIES = {
    "occa-berry": "fire", "passho-berry": "water", "wacan-berry": "electric", "rindo-berry": "grass",
    "yache-berry": "ice", "chople-berry": "fighting", "kebia-berry": "poison", "shuca-berry": "ground",
    "coba-berry": "flying", "payapa-berry": "psychic", "tanga-berry": "bug", "charti-berry": "rock",
    "kasib-berry": "ghost", "haban-berry": "dragon", "colbur-berry": "dark", "babiri-berry": "steel",
    "chilan-berry": "normal", "roseli-berry": "fairy",
}


def item_damage_modifier(attacker: BattlePokemon, defender: BattlePokemon, move: BattleMove, field: BattleField, ctx: DamageContext) -> float:
    attacker_ability = "" if attacker.ability_suppressed else normalize_name(attacker.ability)
    if attacker.item_suppressed or field.magic_room or attacker_ability == "klutz":
        return 1.0
    item = normalize_name(attacker.item)
    mod = 1.0

    if item == "life-orb":
        mod *= 5324 / 4096; ctx.notes.append("Life Orb")
    if item == "expert-belt" and ctx.type_multiplier > 1:
        mod *= 4915 / 4096; ctx.notes.append("Expert Belt")
    if item == "metronome" and attacker.metronome_turns > 0:
        turns = min(5, int(attacker.metronome_turns))
        mod *= 1 + 0.2 * turns; ctx.notes.append("Metronome item")

    return mod


def defender_item_modifier(attacker: BattlePokemon, defender: BattlePokemon, move: BattleMove, field: BattleField, ctx: DamageContext) -> float:
    defender_ability = "" if defender.ability_suppressed else normalize_name(defender.ability)
    if defender.item_suppressed or field.magic_room or defender_ability == "klutz":
        return 1.0
    item = normalize_name(defender.item)
    attacker_ability = "" if attacker.ability_suppressed else normalize_name(attacker.ability)
    mod = 1.0

    if RESIST_BERRIES.get(item) == ctx.move_type and ctx.type_multiplier > 1 and attacker_ability not in {"unnerve", "as-one-glastrier", "as-one-spectrier"}:
        berry_mod = 0.25 if defender_ability == "ripen" else 0.5
        mod *= berry_mod; ctx.notes.append(f"{defender.item} resist berry")
        if berry_mod == 0.25:
            ctx.notes.append("Ripen doubles berry effect")

    return mod


def apply_survival_items(defender: BattlePokemon, damage: int, field: BattleField | None = None) -> tuple[int, str]:
    item = normalize_name(defender.item)
    ability = "" if defender.ability_suppressed else normalize_name(defender.ability)
    if defender.item_suppressed or ability == "klutz" or (field is not None and field.magic_room):
        item = ""
    if damage < defender.hp():
        return damage, ""
    if defender.hp() == defender.max_hp:
        if ability == "sturdy":
            return max(0, defender.hp() - 1), "Sturdy leaves the defender at 1 HP."
        if item == "focus-sash":
            return max(0, defender.hp() - 1), "Focus Sash leaves the defender at 1 HP."
    if item == "focus-band" and defender.used_item:
        return max(0, defender.hp() - 1), "Focus Band activated and leaves the defender at 1 HP."
    return damage, ""


def contact_damage_to_attacker(attacker: BattlePokemon, defender: BattlePokemon, move: BattleMove) -> dict[str, int]:
    damage: dict[str, int] = {}
    if not move.makes_contact:
        return damage

    defender_ability = normalize_name(defender.ability)
    defender_item = normalize_name(defender.item)
    category = normalize_name(move.category)

    if defender_ability == "rough-skin":
        damage["rough_skin"] = max(1, floor(attacker.max_hp / 8))
    if defender_ability == "iron-barbs":
        damage["iron_barbs"] = max(1, floor(attacker.max_hp / 8))
    if defender_item == "rocky-helmet":
        damage["rocky_helmet"] = max(1, floor(attacker.max_hp / 6))
    if defender_item == "sticky-barb":
        damage["sticky_barb_contact"] = max(1, floor(attacker.max_hp / 8))
    if defender_item == "jaboca-berry" and category == "physical":
        damage["jaboca_berry"] = max(1, floor(attacker.max_hp / 8))
    if defender_item == "rowap-berry" and category == "special":
        damage["rowap_berry"] = max(1, floor(attacker.max_hp / 8))

    return damage
