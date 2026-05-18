from __future__ import annotations

from .utils import normalize_name

PUNCH_MOVES = {
    "bullet-punch", "comet-punch", "dizzy-punch", "drain-punch", "dynamic-punch",
    "fire-punch", "focus-punch", "hammer-arm", "ice-hammer", "ice-punch", "mach-punch",
    "mega-punch", "meteor-mash", "plasma-fists", "power-up-punch", "rage-fist", "shadow-punch",
    "sky-uppercut", "surging-strikes", "thunder-punch", "wicked-blow"
}
BITE_MOVES = {"bite", "crunch", "fire-fang", "fishious-rend", "hyper-fang", "ice-fang", "jaw-lock", "poison-fang", "psychic-fangs", "thunder-fang"}
PULSE_MOVES = {"aura-sphere", "dark-pulse", "dragon-pulse", "heal-pulse", "origin-pulse", "terrain-pulse", "water-pulse"}
SOUND_MOVES = {"boomburst", "bug-buzz", "clanging-scales", "disarming-voice", "echoed-voice", "hyper-voice", "overdrive", "round", "snarl", "sparkling-aria", "uproar"}
BULLET_MOVES = {"acupressure", "aura-sphere", "barrage", "bullet-seed", "egg-bomb", "electro-ball", "energy-ball", "focus-blast", "gyro-ball", "ice-ball", "magnet-bomb", "mist-ball", "mud-bomb", "octazooka", "pollen-puff", "pyro-ball", "rock-blast", "searing-shot", "seed-bomb", "shadow-ball", "sludge-bomb", "weather-ball", "zap-cannon"}
SLICING_MOVES = {"aerial-ace", "air-cutter", "aqua-cutter", "behemoth-blade", "bitter-blade", "ceaseless-edge", "cross-poison", "cut", "fury-cutter", "kowtow-cleave", "leaf-blade", "mighty-cleave", "night-slash", "population-bomb", "psyblade", "psycho-cut", "razor-leaf", "razor-shell", "sacred-sword", "secret-sword", "slash", "solar-blade", "stone-axe", "tachyon-cutter", "x-scissor"}
RECOIL_MOVES = {"brave-bird", "double-edge", "flare-blitz", "head-charge", "head-smash", "submission", "take-down", "volt-tackle", "wave-crash", "wild-charge", "wood-hammer"}
SPREAD_MOVES = {"air-cutter", "blizzard", "boomburst", "bulldoze", "dazzling-gleam", "discharge", "earthquake", "eruption", "explosion", "heat-wave", "hyper-voice", "icy-wind", "lava-plume", "magnitude", "muddy-water", "precipice-blades", "razor-leaf", "rock-slide", "self-destruct", "sludge-wave", "snarl", "surf", "swift", "water-spout"}
CONTACT_EXCEPTIONS = {"earthquake", "rock-slide", "surf", "thunderbolt", "flamethrower", "ice-beam", "psychic", "shadow-ball", "energy-ball"}


def enrich_flags(move) -> None:
    name = normalize_name(move.name)
    move.is_punch = bool(move.is_punch or name in PUNCH_MOVES)
    move.is_bite = bool(move.is_bite or name in BITE_MOVES)
    move.is_pulse = bool(move.is_pulse or name in PULSE_MOVES)
    move.is_sound = bool(move.is_sound or name in SOUND_MOVES)
    move.is_bullet = bool(move.is_bullet or name in BULLET_MOVES)
    move.is_slicing = bool(move.is_slicing or name in SLICING_MOVES)
    move.is_recoil = bool(move.is_recoil or name in RECOIL_MOVES)
    move.is_spread = bool(move.is_spread or name in SPREAD_MOVES)
    if name in CONTACT_EXCEPTIONS:
        move.makes_contact = False
