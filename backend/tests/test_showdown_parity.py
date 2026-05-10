import json
import subprocess
from pathlib import Path

import pytest

from pokemon_damage_calculator import analyze_battle


REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE = Path(__file__).with_name("showdown_calc_bridge.mjs")

BASE_STATS = {
    "Charizard": {"hp": 78, "attack": 84, "defense": 78, "special_attack": 109, "special_defense": 85, "speed": 100},
    "Venusaur": {"hp": 80, "attack": 82, "defense": 83, "special_attack": 100, "special_defense": 100, "speed": 80},
    "Garchomp": {"hp": 108, "attack": 130, "defense": 95, "special_attack": 80, "special_defense": 85, "speed": 102},
    "Heatran": {"hp": 91, "attack": 90, "defense": 106, "special_attack": 130, "special_defense": 106, "speed": 77},
    "Dragonite": {"hp": 91, "attack": 134, "defense": 95, "special_attack": 100, "special_defense": 100, "speed": 80},
    "Tyranitar": {"hp": 100, "attack": 134, "defense": 110, "special_attack": 95, "special_defense": 100, "speed": 61},
    "Greninja": {"hp": 72, "attack": 95, "defense": 67, "special_attack": 103, "special_defense": 71, "speed": 122},
    "Landorus-Therian": {"hp": 89, "attack": 145, "defense": 90, "special_attack": 105, "special_defense": 80, "speed": 91},
    "Blastoise": {"hp": 79, "attack": 83, "defense": 100, "special_attack": 85, "special_defense": 105, "speed": 78},
    "Chansey": {"hp": 250, "attack": 5, "defense": 5, "special_attack": 35, "special_defense": 105, "speed": 50},
    "Lucario": {"hp": 70, "attack": 110, "defense": 70, "special_attack": 115, "special_defense": 70, "speed": 90},
    "Porygon-Z": {"hp": 85, "attack": 80, "defense": 70, "special_attack": 135, "special_defense": 75, "speed": 90},
    "Crawdaunt": {"hp": 63, "attack": 120, "defense": 85, "special_attack": 90, "special_defense": 55, "speed": 55},
    "Bronzong": {"hp": 67, "attack": 89, "defense": 116, "special_attack": 79, "special_defense": 116, "speed": 33},
    "Araquanid": {"hp": 68, "attack": 70, "defense": 92, "special_attack": 50, "special_defense": 132, "speed": 42},
    "Garganacl": {"hp": 100, "attack": 100, "defense": 130, "special_attack": 45, "special_defense": 90, "speed": 35},
    "Gengar": {"hp": 60, "attack": 65, "defense": 60, "special_attack": 130, "special_defense": 75, "speed": 110},
    "Scizor": {"hp": 70, "attack": 130, "defense": 100, "special_attack": 55, "special_defense": 80, "speed": 65},
    "Gallade": {"hp": 68, "attack": 125, "defense": 65, "special_attack": 65, "special_defense": 115, "speed": 80},
    "Tyrantrum": {"hp": 82, "attack": 121, "defense": 119, "special_attack": 69, "special_defense": 59, "speed": 71},
    "Metagross": {"hp": 80, "attack": 135, "defense": 130, "special_attack": 95, "special_defense": 90, "speed": 70},
    "Toxtricity": {"hp": 75, "attack": 98, "defense": 70, "special_attack": 114, "special_defense": 70, "speed": 75},
    "Conkeldurr": {"hp": 105, "attack": 140, "defense": 95, "special_attack": 55, "special_defense": 65, "speed": 45},
    "Zangoose": {"hp": 73, "attack": 115, "defense": 60, "special_attack": 60, "special_defense": 60, "speed": 90},
    "Drifblim": {"hp": 150, "attack": 80, "defense": 44, "special_attack": 90, "special_defense": 54, "speed": 80},
    "Sylveon": {"hp": 95, "attack": 65, "defense": 65, "special_attack": 110, "special_defense": 130, "speed": 60},
    "Xerneas": {"hp": 126, "attack": 131, "defense": 95, "special_attack": 131, "special_defense": 98, "speed": 99},
    "Yveltal": {"hp": 126, "attack": 131, "defense": 95, "special_attack": 131, "special_defense": 98, "speed": 99},
    "Chien-Pao": {"hp": 80, "attack": 120, "defense": 80, "special_attack": 90, "special_defense": 65, "speed": 135},
    "Chi-Yu": {"hp": 55, "attack": 80, "defense": 80, "special_attack": 135, "special_defense": 120, "speed": 100},
    "Wo-Chien": {"hp": 85, "attack": 85, "defense": 100, "special_attack": 95, "special_defense": 135, "speed": 70},
    "Ting-Lu": {"hp": 155, "attack": 110, "defense": 125, "special_attack": 55, "special_defense": 80, "speed": 45},
    "Frosmoth": {"hp": 70, "attack": 65, "defense": 60, "special_attack": 125, "special_defense": 90, "speed": 65},
    "Persian-Alola": {"hp": 65, "attack": 60, "defense": 60, "special_attack": 75, "special_defense": 65, "speed": 115},
    "Milotic": {"hp": 95, "attack": 60, "defense": 79, "special_attack": 100, "special_defense": 125, "speed": 81},
    "Gogoat": {"hp": 123, "attack": 100, "defense": 62, "special_attack": 97, "special_defense": 81, "speed": 68},
    "Regieleki": {"hp": 80, "attack": 100, "defense": 50, "special_attack": 100, "special_defense": 50, "speed": 200},
    "Delcatty": {"hp": 70, "attack": 65, "defense": 65, "special_attack": 55, "special_defense": 55, "speed": 90},
    "Primarina": {"hp": 80, "attack": 74, "defense": 74, "special_attack": 126, "special_defense": 116, "speed": 60},
    "Bewear": {"hp": 120, "attack": 125, "defense": 80, "special_attack": 55, "special_defense": 60, "speed": 60},
    "Decidueye": {"hp": 78, "attack": 107, "defense": 75, "special_attack": 100, "special_defense": 100, "speed": 70},
}


def pokemon(name, **overrides):
    data = {
        "name": name,
        "level": 50,
        "baseStats": BASE_STATS[name],
        "ivs": {"hp": 31, "attack": 31, "defense": 31, "special_attack": 31, "special_defense": 31, "speed": 31},
        "evs": {},
        "nature": "Serious",
        "types": [],
    }
    data.update(overrides)
    return data


def showdown_damage(payload):
    result = subprocess.run(
        ["node", str(BRIDGE)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
        cwd=REPO_ROOT,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "attacker": pokemon(
                "Charizard",
                types=["fire", "flying"],
                nature="Modest",
                evs={"special_attack": 252},
            ),
            "defender": pokemon("Venusaur", types=["grass", "poison"], evs={"hp": 252}),
            "move": {"name": "Flamethrower"},
            "field": {},
        },
        {
            "attacker": pokemon(
                "Garchomp",
                types=["dragon", "ground"],
                nature="Adamant",
                evs={"attack": 252},
                item="Life Orb",
            ),
            "defender": pokemon("Heatran", types=["fire", "steel"], evs={"hp": 252}),
            "move": {"name": "Earthquake"},
            "field": {},
        },
        {
            "attacker": pokemon(
                "Charizard",
                types=["fire", "flying"],
                nature="Modest",
                evs={"special_attack": 252},
                ability="Solar Power",
                item="Life Orb",
            ),
            "defender": pokemon("Venusaur", types=["grass", "poison"], evs={"hp": 252}),
            "move": {"name": "Flamethrower"},
            "field": {"weather": "sun"},
        },
        {
            "attacker": pokemon(
                "Tyranitar",
                types=["rock", "dark"],
                nature="Adamant",
                evs={"attack": 252},
            ),
            "defender": pokemon("Dragonite", types=["dragon", "flying"], evs={"hp": 252}, ability="Multiscale"),
            "move": {"name": "Stone Edge"},
            "field": {},
        },
        {
            "attacker": pokemon(
                "Greninja",
                types=["water", "dark"],
                nature="Timid",
                evs={"special_attack": 252},
                ability="Protean",
            ),
            "defender": pokemon("Landorus-Therian", types=["ground", "flying"], evs={"hp": 252}),
            "move": {"name": "Ice Beam"},
            "field": {},
        },
        {
            "attacker": pokemon(
                "Landorus-Therian",
                types=["ground", "flying"],
                nature="Adamant",
                evs={"attack": 252},
            ),
            "defender": pokemon("Tyranitar", types=["rock", "dark"], evs={"hp": 252}, item="Shuca Berry"),
            "move": {"name": "Earthquake"},
            "field": {},
        },
        {
            "attacker": pokemon(
                "Garchomp",
                types=["dragon", "ground"],
                nature="Adamant",
                evs={"attack": 252},
                item="Choice Band",
                status="burn",
            ),
            "defender": pokemon("Blastoise", types=["water"], evs={"hp": 252}, item="Assault Vest"),
            "move": {"name": "Dragon Claw"},
            "field": {"defenderSide": {"reflect": True}},
        },
        {
            "attacker": pokemon(
                "Charizard",
                types=["fire", "flying"],
                nature="Modest",
                evs={"special_attack": 252},
                item="Choice Specs",
            ),
            "defender": pokemon("Venusaur", types=["grass", "poison"], evs={"hp": 252}, ability="Thick Fat"),
            "move": {"name": "Flamethrower"},
            "field": {"weather": "rain"},
        },
        {
            "attacker": pokemon(
                "Lucario",
                types=["fighting", "steel"],
                nature="Adamant",
                evs={"attack": 252},
                item="Expert Belt",
            ),
            "defender": pokemon("Tyranitar", types=["rock", "dark"], evs={"hp": 252}),
            "move": {"name": "Close Combat"},
            "field": {},
        },
        {
            "attacker": pokemon(
                "Charizard",
                types=["fire", "flying"],
                nature="Modest",
                evs={"special_attack": 252},
            ),
            "defender": pokemon("Chansey", types=["normal"], evs={"hp": 252}, item="Eviolite"),
            "move": {"name": "Flamethrower"},
            "field": {},
        },
        {
            "attacker": pokemon(
                "Crawdaunt",
                types=["water", "dark"],
                nature="Adamant",
                evs={"attack": 252},
                ability="Adaptability",
            ),
            "defender": pokemon("Heatran", types=["fire", "steel"], evs={"hp": 252}),
            "move": {"name": "Crabhammer"},
            "field": {},
        },
        {
            "attacker": pokemon(
                "Charizard",
                types=["fire", "flying"],
                nature="Modest",
                evs={"special_attack": 252},
                item="Choice Specs",
            ),
            "defender": pokemon("Bronzong", types=["steel", "psychic"], evs={"hp": 252}, ability="Heatproof"),
            "move": {"name": "Flamethrower"},
            "field": {},
        },
        {
            "attacker": pokemon(
                "Heatran",
                types=["fire", "steel"],
                nature="Modest",
                evs={"special_attack": 252},
                item="Life Orb",
            ),
            "defender": pokemon("Araquanid", types=["water", "bug"], evs={"hp": 252}, ability="Water Bubble"),
            "move": {"name": "Flamethrower"},
            "field": {"weather": "sun"},
        },
        {
            "attacker": pokemon(
                "Gengar",
                types=["ghost", "poison"],
                nature="Timid",
                evs={"special_attack": 252},
                item="Choice Specs",
            ),
            "defender": pokemon("Garganacl", types=["rock"], evs={"hp": 252}, ability="Purifying Salt"),
            "move": {"name": "Shadow Ball"},
            "field": {},
        },
        {
            "attacker": pokemon("Scizor", types=["bug", "steel"], nature="Adamant", evs={"attack": 252}, ability="Technician"),
            "defender": pokemon("Tyranitar", types=["rock", "dark"], evs={"hp": 252}),
            "move": {"name": "Bullet Punch"},
            "field": {},
        },
        {
            "attacker": pokemon("Gallade", types=["psychic", "fighting"], nature="Adamant", evs={"attack": 252}, ability="Sharpness"),
            "defender": pokemon("Venusaur", types=["grass", "poison"], evs={"hp": 252}),
            "move": {"name": "Psycho Cut"},
            "field": {},
        },
        {
            "attacker": pokemon("Tyrantrum", types=["rock", "dragon"], nature="Adamant", evs={"attack": 252}, ability="Strong Jaw"),
            "defender": pokemon("Gengar", types=["ghost", "poison"], evs={"hp": 252}),
            "move": {"name": "Crunch"},
            "field": {},
        },
        {
            "attacker": pokemon("Metagross", types=["steel", "psychic"], nature="Adamant", evs={"attack": 252}, ability="Tough Claws"),
            "defender": pokemon("Tyranitar", types=["rock", "dark"], evs={"hp": 252}),
            "move": {"name": "Meteor Mash", "makesContact": True},
            "field": {},
        },
        {
            "attacker": pokemon("Toxtricity", types=["electric", "poison"], nature="Modest", evs={"special_attack": 252}, ability="Punk Rock"),
            "defender": pokemon("Venusaur", types=["grass", "poison"], evs={"hp": 252}),
            "move": {"name": "Boomburst"},
            "field": {},
        },
        {
            "attacker": pokemon("Conkeldurr", types=["fighting"], nature="Adamant", evs={"attack": 252}, ability="Iron Fist"),
            "defender": pokemon("Tyranitar", types=["rock", "dark"], evs={"hp": 252}),
            "move": {"name": "Drain Punch"},
            "field": {},
        },
        {
            "attacker": pokemon("Zangoose", types=["normal"], nature="Adamant", evs={"attack": 252}, ability="Toxic Boost", status="poison"),
            "defender": pokemon("Blastoise", types=["water"], evs={"hp": 252}),
            "move": {"name": "Facade"},
            "field": {},
        },
        {
            "attacker": pokemon("Drifblim", types=["ghost", "flying"], nature="Modest", evs={"special_attack": 252}, ability="Flare Boost", status="burn"),
            "defender": pokemon("Bronzong", types=["steel", "psychic"], evs={"hp": 252}),
            "move": {"name": "Shadow Ball"},
            "field": {},
        },
        {
            "attacker": pokemon("Charizard", types=["fire", "flying"], nature="Modest", evs={"special_attack": 252}, item="Wise Glasses"),
            "defender": pokemon("Venusaur", types=["grass", "poison"], evs={"hp": 252}),
            "move": {"name": "Flamethrower"},
            "field": {},
        },
        {
            "attacker": pokemon("Garchomp", types=["dragon", "ground"], nature="Adamant", evs={"attack": 252}, item="Muscle Band"),
            "defender": pokemon("Heatran", types=["fire", "steel"], evs={"hp": 252}),
            "move": {"name": "Earthquake"},
            "field": {},
        },
        {
            "attacker": pokemon("Charizard", types=["fire", "flying"], nature="Modest", evs={"special_attack": 252}, item="Charcoal"),
            "defender": pokemon("Venusaur", types=["grass", "poison"], evs={"hp": 252}),
            "move": {"name": "Flamethrower"},
            "field": {},
        },
        {
            "attacker": pokemon("Sylveon", types=["fairy"], nature="Modest", evs={"special_attack": 252}, ability="Pixilate"),
            "defender": pokemon("Garchomp", types=["dragon", "ground"], evs={"hp": 252}),
            "move": {"name": "Hyper Voice"},
            "field": {},
        },
        {
            "attacker": pokemon("Xerneas", types=["fairy"], nature="Modest", evs={"special_attack": 252}, ability="Fairy Aura"),
            "defender": pokemon("Garchomp", types=["dragon", "ground"], evs={"hp": 252}),
            "move": {"name": "Moonblast"},
            "field": {},
        },
        {
            "attacker": pokemon("Xerneas", types=["fairy"], nature="Modest", evs={"special_attack": 252}, ability="Fairy Aura"),
            "defender": pokemon("Garchomp", types=["dragon", "ground"], evs={"hp": 252}, ability="Aura Break"),
            "move": {"name": "Moonblast"},
            "field": {},
        },
        {
            "attacker": pokemon("Yveltal", types=["dark", "flying"], nature="Modest", evs={"special_attack": 252}, ability="Dark Aura"),
            "defender": pokemon("Gengar", types=["ghost", "poison"], evs={"hp": 252}),
            "move": {"name": "Dark Pulse"},
            "field": {},
        },
        {
            "attacker": pokemon("Porygon-Z", types=["normal"], nature="Modest", evs={"special_attack": 252}),
            "defender": pokemon("Blastoise", types=["water"], evs={"hp": 252}),
            "move": {"name": "Thunderbolt"},
            "field": {"terrain": "electric"},
        },
        {
            "attacker": pokemon("Garchomp", types=["dragon", "ground"], nature="Adamant", evs={"attack": 252}),
            "defender": pokemon("Heatran", types=["fire", "steel"], evs={"hp": 252}),
            "move": {"name": "Earthquake"},
            "field": {"terrain": "grassy"},
        },
        {
            "attacker": pokemon("Charizard", types=["fire", "flying"], nature="Modest", evs={"special_attack": 252}),
            "defender": pokemon("Venusaur", types=["grass", "poison"], evs={"hp": 252}),
            "move": {"name": "Flamethrower"},
            "field": {"attackerSide": {"helpingHand": True}},
        },
        {
            "attacker": pokemon("Chien-Pao", types=["dark", "ice"], nature="Adamant", evs={"attack": 252}, ability="Sword of Ruin"),
            "defender": pokemon("Blastoise", types=["water"], evs={"hp": 252}),
            "move": {"name": "Ice Spinner"},
            "field": {},
        },
        {
            "attacker": pokemon("Chi-Yu", types=["dark", "fire"], nature="Modest", evs={"special_attack": 252}, ability="Beads of Ruin"),
            "defender": pokemon("Blastoise", types=["water"], evs={"hp": 252}),
            "move": {"name": "Flamethrower"},
            "field": {},
        },
        {
            "attacker": pokemon("Garchomp", types=["dragon", "ground"], nature="Adamant", evs={"attack": 252}),
            "defender": pokemon("Wo-Chien", types=["dark", "grass"], evs={"hp": 252}, ability="Tablets of Ruin"),
            "move": {"name": "Dragon Claw"},
            "field": {},
        },
        {
            "attacker": pokemon("Charizard", types=["fire", "flying"], nature="Modest", evs={"special_attack": 252}),
            "defender": pokemon("Ting-Lu", types=["dark", "ground"], evs={"hp": 252}, ability="Vessel of Ruin"),
            "move": {"name": "Flamethrower"},
            "field": {},
        },
        {
            "attacker": pokemon("Charizard", types=["fire", "flying"], nature="Modest", evs={"special_attack": 252}),
            "defender": pokemon("Frosmoth", types=["ice", "bug"], evs={"hp": 252}, ability="Ice Scales"),
            "move": {"name": "Flamethrower"},
            "field": {},
        },
        {
            "attacker": pokemon("Lucario", types=["fighting", "steel"], nature="Adamant", evs={"attack": 252}),
            "defender": pokemon("Persian-Alola", types=["dark"], evs={"hp": 252}, ability="Fur Coat"),
            "move": {"name": "Close Combat"},
            "field": {},
        },
        {
            "attacker": pokemon("Garchomp", types=["dragon", "ground"], nature="Adamant", evs={"attack": 252}),
            "defender": pokemon("Milotic", types=["water"], evs={"hp": 252}, ability="Marvel Scale", status="burn"),
            "move": {"name": "Earthquake"},
            "field": {},
        },
        {
            "attacker": pokemon("Garchomp", types=["dragon", "ground"], nature="Adamant", evs={"attack": 252}),
            "defender": pokemon("Gogoat", types=["grass"], evs={"hp": 252}, ability="Grass Pelt"),
            "move": {"name": "Earthquake"},
            "field": {"terrain": "grassy"},
        },
        {
            "attacker": pokemon(
                "Charizard",
                types=["fire", "flying"],
                nature="Modest",
                evs={"special_attack": 252},
                ability="Blaze",
                currentHp=51,
            ),
            "defender": pokemon("Venusaur", types=["grass", "poison"], evs={"hp": 252}),
            "move": {"name": "Flamethrower"},
            "field": {},
        },
        {
            "attacker": pokemon(
                "Heatran",
                types=["fire", "steel"],
                nature="Modest",
                evs={"special_attack": 252},
                ability="Flash Fire",
                flashFireActive=True,
            ),
            "defender": pokemon("Venusaur", types=["grass", "poison"], evs={"hp": 252}),
            "move": {"name": "Flamethrower"},
            "field": {},
        },
        {
            "attacker": pokemon("Araquanid", types=["water", "bug"], nature="Modest", evs={"special_attack": 252}, ability="Water Bubble"),
            "defender": pokemon("Heatran", types=["fire", "steel"], evs={"hp": 252}),
            "move": {"name": "Surf"},
            "field": {},
        },
        {
            "attacker": pokemon("Metagross", types=["steel", "psychic"], nature="Adamant", evs={"attack": 252}, ability="Steelworker"),
            "defender": pokemon("Tyranitar", types=["rock", "dark"], evs={"hp": 252}),
            "move": {"name": "Meteor Mash"},
            "field": {},
        },
        {
            "attacker": pokemon("Dragonite", types=["dragon", "flying"], nature="Adamant", evs={"attack": 252}, ability="Dragon's Maw"),
            "defender": pokemon("Blastoise", types=["water"], evs={"hp": 252}),
            "move": {"name": "Dragon Claw"},
            "field": {},
        },
        {
            "attacker": pokemon("Tyranitar", types=["rock", "dark"], nature="Adamant", evs={"attack": 252}, ability="Rocky Payload"),
            "defender": pokemon("Dragonite", types=["dragon", "flying"], evs={"hp": 252}),
            "move": {"name": "Stone Edge"},
            "field": {},
        },
        {
            "attacker": pokemon("Regieleki", types=["electric"], nature="Modest", evs={"special_attack": 252}, ability="Transistor"),
            "defender": pokemon("Blastoise", types=["water"], evs={"hp": 252}),
            "move": {"name": "Thunderbolt"},
            "field": {},
        },
        {
            "attacker": pokemon("Gengar", types=["ghost", "poison"], nature="Timid", evs={"special_attack": 252}, ability="Stakeout", abilityOn=True),
            "defender": pokemon("Blastoise", types=["water"], evs={"hp": 252}, switchedInThisTurn=True),
            "move": {"name": "Shadow Ball"},
            "field": {},
        },
        {
            "attacker": pokemon("Charizard", types=["fire", "flying"], nature="Modest", evs={"special_attack": 252}),
            "defender": pokemon("Venusaur", types=["grass", "poison"], evs={"hp": 252}),
            "move": {"name": "Flamethrower"},
            "field": {"attackerSide": {"battery": True}},
        },
        {
            "attacker": pokemon("Garchomp", types=["dragon", "ground"], nature="Adamant", evs={"attack": 252}),
            "defender": pokemon("Heatran", types=["fire", "steel"], evs={"hp": 252}),
            "move": {"name": "Earthquake"},
            "field": {"attackerSide": {"powerSpot": True}},
        },
        {
            "attacker": pokemon("Metagross", types=["steel", "psychic"], nature="Adamant", evs={"attack": 252}),
            "defender": pokemon("Tyranitar", types=["rock", "dark"], evs={"hp": 252}),
            "move": {"name": "Meteor Mash"},
            "field": {"attackerSide": {"steelySpirit": True}},
        },
        {
            "attacker": pokemon("Garchomp", types=["dragon", "ground"], nature="Adamant", evs={"attack": 252}),
            "defender": pokemon("Heatran", types=["fire", "steel"], evs={"hp": 252}),
            "move": {"name": "Earthquake"},
            "field": {"weather": "sun", "attackerSide": {"flowerGift": True}},
        },
        {
            "attacker": pokemon("Charizard", types=["fire", "flying"], nature="Modest", evs={"special_attack": 252}),
            "defender": pokemon("Blastoise", types=["water"], evs={"hp": 252}),
            "move": {"name": "Flamethrower"},
            "field": {"weather": "sun", "defenderSide": {"flowerGift": True}},
        },
        {
            "attacker": pokemon("Porygon-Z", types=["normal"], nature="Modest", evs={"special_attack": 252}, ability="Plus", abilityOn=True),
            "defender": pokemon("Blastoise", types=["water"], evs={"hp": 252}),
            "move": {"name": "Thunderbolt"},
            "field": {},
        },
        {
            "attacker": pokemon("Porygon-Z", types=["normal"], nature="Modest", evs={"special_attack": 252}, ability="Normalize"),
            "defender": pokemon("Blastoise", types=["water"], evs={"hp": 252}),
            "move": {"name": "Thunderbolt"},
            "field": {},
        },
        {
            "attacker": pokemon("Primarina", types=["water", "fairy"], nature="Modest", evs={"special_attack": 252}, ability="Liquid Voice"),
            "defender": pokemon("Heatran", types=["fire", "steel"], evs={"hp": 252}),
            "move": {"name": "Hyper Voice"},
            "field": {},
        },
        {
            "attacker": pokemon("Garchomp", types=["dragon", "ground"], nature="Adamant", evs={"attack": 252}),
            "defender": pokemon("Bewear", types=["normal", "fighting"], evs={"hp": 252}, ability="Fluffy"),
            "move": {"name": "Dragon Claw", "makesContact": True},
            "field": {},
        },
        {
            "attacker": pokemon("Decidueye", types=["grass", "ghost"], nature="Adamant", evs={"attack": 252}, ability="Long Reach"),
            "defender": pokemon("Bewear", types=["normal", "fighting"], evs={"hp": 252}, ability="Fluffy"),
            "move": {"name": "Leaf Blade"},
            "field": {},
        },
        {
            "attacker": pokemon("Garchomp", types=["dragon", "ground"], nature="Adamant", evs={"attack": 252}, ability="Unnerve"),
            "defender": pokemon("Tyranitar", types=["rock", "dark"], evs={"hp": 252}, item="Shuca Berry"),
            "move": {"name": "Earthquake"},
            "field": {},
        },
        {
            "attacker": pokemon("Charizard", types=["fire", "flying"], nature="Modest", evs={"special_attack": 252}),
            "defender": pokemon("Blastoise", types=["water"], evs={"hp": 252}, ability="Klutz", item="Assault Vest"),
            "move": {"name": "Flamethrower"},
            "field": {},
        },
        {
            "attacker": pokemon("Lucario", types=["fighting", "steel"], nature="Adamant", evs={"attack": 252}, ability="Mold Breaker"),
            "defender": pokemon("Persian-Alola", types=["dark"], evs={"hp": 252}, ability="Fur Coat"),
            "move": {"name": "Close Combat"},
            "field": {},
        },
        {
            "attacker": pokemon("Lucario", types=["fighting", "steel"], nature="Adamant", evs={"attack": 252}, ability="Mold Breaker"),
            "defender": pokemon("Persian-Alola", types=["dark"], evs={"hp": 252}, ability="Fur Coat", item="Ability Shield"),
            "move": {"name": "Close Combat"},
            "field": {},
        },
        {
            "attacker": pokemon("Gengar", types=["ghost", "poison"], nature="Timid", evs={"special_attack": 252}),
            "defender": pokemon("Dragonite", types=["dragon", "flying"], evs={"hp": 252}, ability="Multiscale"),
            "move": {"name": "Moongeist Beam"},
            "field": {},
        },
        {
            "attacker": pokemon("Gengar", types=["ghost", "poison"], nature="Timid", evs={"special_attack": 252}),
            "defender": pokemon("Dragonite", types=["dragon", "flying"], evs={"hp": 252}, ability="Multiscale", item="Ability Shield"),
            "move": {"name": "Moongeist Beam"},
            "field": {},
        },
        {
            "attacker": pokemon("Metagross", types=["steel", "psychic"], nature="Adamant", evs={"attack": 252}),
            "defender": pokemon("Persian-Alola", types=["dark"], evs={"hp": 252}, ability="Fur Coat"),
            "move": {"name": "Sunsteel Strike"},
            "field": {},
        },
        {
            "attacker": pokemon("Charizard", types=["fire", "flying"], nature="Modest", evs={"special_attack": 252}, ability="Neutralizing Gas"),
            "defender": pokemon("Venusaur", types=["grass", "poison"], evs={"hp": 252}, ability="Thick Fat"),
            "move": {"name": "Flamethrower"},
            "field": {},
        },
        {
            "attacker": pokemon("Regieleki", types=["electric"], nature="Modest", evs={"special_attack": 252}, ability="Transistor"),
            "defender": pokemon("Blastoise", types=["water"], evs={"hp": 252}, ability="Neutralizing Gas"),
            "move": {"name": "Thunderbolt"},
            "field": {},
        },
    ],
)
def test_damage_rolls_match_pokemon_showdown(payload):
    ours = analyze_battle(payload)["damage"]
    showdown = showdown_damage(payload)

    assert ours["damageValues"] == showdown["damageValues"], showdown["description"]
    assert ours["minDamage"] == showdown["minDamage"]
    assert ours["maxDamage"] == showdown["maxDamage"]
