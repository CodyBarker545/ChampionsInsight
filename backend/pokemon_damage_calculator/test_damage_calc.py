from pokemon_damage_calculator import analyze_battle


def base_defender(**extra):
    data = {
        "name": "Blastoise",
        "level": 50,
        "types": ["water"],
        "stats": {"hp": 186, "attack": 92, "defense": 120, "special_attack": 105, "special_defense": 125, "speed": 98},
        "maxHp": 186,
        "currentHp": 186,
    }
    data.update(extra)
    return data


def test_weather_ball_type_boost_uses_final_type():
    payload = {
        "attacker": {
            "name": "Castform", "level": 50, "types": ["normal"],
            "stats": {"hp": 150, "attack": 80, "defense": 80, "special_attack": 100, "special_defense": 80, "speed": 80},
            "maxHp": 150, "currentHp": 150, "item": "Mystic Water"
        },
        "defender": base_defender(types=["fire", "flying"], maxHp=153, currentHp=153),
        "move": {"name": "Weather Ball", "type": "normal", "category": "special", "power": 50},
        "field": {"weather": "rain"}
    }
    result = analyze_battle(payload)
    assert result["damage"]["moveType"] == "water"
    assert result["damage"]["powerUsed"] == 120
    assert "Mystic Water type boost" in result["damage"]["notes"]


def test_nature_calculates_final_stats_and_hp():
    payload = {
        "attacker": {
            "name": "Charizard",
            "level": 50,
            "types": ["fire", "flying"],
            "nature": "Modest",
            "baseStats": {"hp": 78, "attack": 84, "defense": 78, "special_attack": 109, "special_defense": 85, "speed": 100},
            "ivs": {"hp": 31, "attack": 31, "defense": 31, "special_attack": 31, "special_defense": 31, "speed": 31},
            "evs": {"hp": 0, "attack": 0, "defense": 0, "special_attack": 252, "special_defense": 4, "speed": 252},
        },
        "defender": base_defender(),
        "move": {"name": "Flamethrower"},
        "field": {}
    }
    result = analyze_battle(payload)
    assert result["damage"]["attackStatUsed"] == 177  # Modest Charizard level 50, 252 SpA EVs, 31 IVs
    assert result["damage"]["minDamage"] > 0


def test_special_moves_use_special_defense_from_payload():
    payload = {
        "attacker": {
            "name": "Dragonite Mega",
            "level": 50,
            "types": ["dragon", "flying"],
            "stats": {
                "hp": 198,
                "attack": 138,
                "defense": 119,
                "special_attack": 158,
                "special_defense": 121,
                "speed": 105,
            },
            "maxHp": 198,
        },
        "defender": {
            "name": "Toxapex",
            "level": 50,
            "types": ["poison", "water"],
            "stats": {
                "hp": 125,
                "attack": 83,
                "defense": 172,
                "special_attack": 73,
                "special_defense": 162,
                "speed": 55,
            },
            "maxHp": 125,
        },
        "move": {"name": "Hurricane"},
        "field": {"isDoubles": True},
    }

    result = analyze_battle(payload)
    damage = result["damage"]

    assert damage["category"] == "special"
    assert damage["defenseStatName"] == "special_defense"
    assert damage["defenseStatUsed"] == 162
    assert damage["damageRange"] == "61 - 73"
    assert damage["percentRange"] == "48.8% - 58.4%"
    assert damage["koChance"] == "Possible 2HKO"


def test_physical_moves_use_attack_and_defense_from_payload():
    payload = {
        "attacker": {
            "name": "Scizor",
            "level": 50,
            "types": ["bug", "steel"],
            "stats": {
                "hp": 145,
                "attack": 180,
                "defense": 120,
                "special_attack": 75,
                "special_defense": 100,
                "speed": 85,
            },
            "maxHp": 145,
        },
        "defender": {
            "name": "Aerodactyl",
            "level": 50,
            "types": ["rock", "flying"],
            "stats": {
                "hp": 155,
                "attack": 125,
                "defense": 85,
                "special_attack": 80,
                "special_defense": 95,
                "speed": 150,
            },
            "maxHp": 155,
        },
        "move": {"name": "Bullet Punch"},
        "field": {"isDoubles": True},
    }

    result = analyze_battle(payload)
    damage = result["damage"]

    assert damage["category"] == "physical"
    assert damage["attackStatName"] == "attack"
    assert damage["attackStatUsed"] == 180
    assert damage["defenseStatName"] == "defense"
    assert damage["defenseStatUsed"] == 85
    assert damage["minPercent"] == round(damage["minDamage"] / 155 * 100, 1)
    assert result["speed"]["user"]["baseSpeed"] == 85
    assert result["speed"]["opponent"]["baseSpeed"] == 150


def test_damage_payload_accepts_frontend_camel_case_stats():
    payload = {
        "attacker": {
            "name": "Dragonite Mega",
            "level": 50,
            "types": ["Dragon", "Flying"],
            "stats": {
                "hp": 198,
                "attack": 138,
                "defense": 119,
                "specialAttack": 158,
                "specialDefense": 121,
                "speed": 105,
            },
            "maxHp": 198,
        },
        "defender": {
            "name": "Toxapex",
            "level": 50,
            "types": ["Poison", "Water"],
            "stats": {
                "hp": 125,
                "attack": 83,
                "defense": 172,
                "specialAttack": 73,
                "specialDefense": 162,
                "speed": 55,
            },
            "maxHp": 125,
        },
        "move": {"name": "Hurricane"},
        "field": {"isDoubles": True},
    }

    result = analyze_battle(payload)
    damage = result["damage"]

    assert damage["attackStatUsed"] == 158
    assert damage["defenseStatUsed"] == 162
    assert damage["percentRange"] == "48.8% - 58.4%"


def test_protosynthesis_auto_detects_highest_stat_after_nature():
    payload = {
        "attacker": {
            "name": "Flutter Mane",
            "level": 50,
            "types": ["ghost", "fairy"],
            "ability": "Protosynthesis",
            "nature": "Modest",
            "baseStats": {"hp": 55, "attack": 55, "defense": 55, "special_attack": 135, "special_defense": 135, "speed": 135},
            "ivs": {"hp": 31, "attack": 31, "defense": 31, "special_attack": 31, "special_defense": 31, "speed": 31},
            "evs": {"special_attack": 252, "speed": 252, "special_defense": 4},
        },
        "defender": base_defender(types=["dragon"], maxHp=180, currentHp=180),
        "move": {"name": "Moonblast", "type": "fairy", "category": "special", "power": 95, "accuracy": 100},
        "field": {"weather": "sun"}
    }
    result = analyze_battle(payload)
    assert result["damage"]["attackStatUsed"] > 205
    assert any("Protosynthesis boosts special_attack" in note for note in result["damage"]["notes"])


def test_protect_blocks_damage():
    payload = {
        "attacker": {
            "name": "Garchomp", "level": 50, "types": ["dragon", "ground"],
            "stats": {"hp": 183, "attack": 200, "defense": 115, "special_attack": 90, "special_defense": 105, "speed": 169},
            "maxHp": 183, "currentHp": 183
        },
        "defender": base_defender(protected=True),
        "move": {"name": "Earthquake", "type": "ground", "category": "physical", "power": 100, "accuracy": 100},
        "field": {}
    }
    result = analyze_battle(payload)
    assert result["damage"]["maxDamage"] == 0
    assert "Target is protected." in result["damage"]["notes"]


def test_accuracy_reports_hustle_penalty():
    payload = {
        "attacker": {
            "name": "Durant", "level": 50, "types": ["bug", "steel"], "ability": "Hustle",
            "stats": {"hp": 140, "attack": 160, "defense": 130, "special_attack": 60, "special_defense": 80, "speed": 177},
            "maxHp": 140, "currentHp": 140
        },
        "defender": base_defender(),
        "move": {"name": "Iron Head", "type": "steel", "category": "physical", "power": 80, "accuracy": 100},
        "field": {}
    }
    result = analyze_battle(payload)
    assert result["damage"]["accuracy"]["accuracy"] == 80.0
    assert "Hustle accuracy penalty" in result["damage"]["accuracy"]["notes"]


def test_shell_armor_prevents_all_critical_effects():
    payload = {
        "attacker": {
            "name": "Garchomp", "level": 50, "types": ["dragon", "ground"],
            "stats": {"hp": 183, "attack": 200, "defense": 115, "special_attack": 90, "special_defense": 105, "speed": 169},
            "maxHp": 183, "currentHp": 183,
            "boosts": {"attack": -2},
        },
        "defender": base_defender(ability="Shell Armor", boosts={"defense": 2}),
        "move": {"name": "Earthquake", "type": "ground", "category": "physical", "power": 100, "accuracy": 100},
        "field": {"critical": True, "defenderSide": {"reflect": True}},
    }
    result = analyze_battle(payload)
    damage = result["damage"]

    assert damage["modifiers"]["critical"] == 1.0
    assert damage["modifiers"]["screen"] == 0.5
    assert "Critical hit ignores negative offensive stages." not in damage["notes"]
    assert "Critical hit ignores positive defensive stages." not in damage["notes"]


def test_tailwind_applies_to_each_side_independently():
    payload = {
        "attacker": {
            "name": "Slow Lead", "level": 50, "types": ["normal"],
            "stats": {"hp": 150, "attack": 100, "defense": 100, "special_attack": 100, "special_defense": 100, "speed": 60},
            "maxHp": 150,
        },
        "defender": base_defender(
            stats={"hp": 186, "attack": 92, "defense": 120, "special_attack": 105, "special_defense": 125, "speed": 100}
        ),
        "move": {"name": "Tackle", "type": "normal", "category": "physical", "power": 40, "accuracy": 100},
        "field": {"attackerSide": {"tailwind": True}, "defenderSide": {}},
    }
    result = analyze_battle(payload)

    assert result["speed"]["user"]["modifiedSpeed"] == 120
    assert result["speed"]["opponent"]["modifiedSpeed"] == 100
    assert result["speed"]["result"] == "Slow Lead moves first."


def test_low_hp_type_ability_and_gem_stack():
    payload = {
        "attacker": {
            "name": "Charizard", "level": 50, "types": ["fire", "flying"], "ability": "Blaze", "item": "Fire Gem",
            "stats": {"hp": 150, "attack": 90, "defense": 90, "special_attack": 120, "special_defense": 90, "speed": 100},
            "maxHp": 150, "currentHp": 50,
        },
        "defender": base_defender(types=["grass"], maxHp=180, currentHp=180),
        "move": {"name": "Flamethrower", "type": "fire", "category": "special", "power": 90, "accuracy": 100},
        "field": {},
    }
    result = analyze_battle(payload)
    damage = result["damage"]

    assert damage["attackStatUsed"] == 180
    assert damage["powerUsed"] == 117
    assert "Blaze" in damage["notes"]
    assert "Fire Gem gem boost" in damage["notes"]


def test_species_stat_items_apply_to_damage_stats():
    pikachu_payload = {
        "attacker": {
            "name": "Pikachu", "level": 50, "types": ["electric"], "item": "Light Ball",
            "stats": {"hp": 110, "attack": 80, "defense": 50, "special_attack": 70, "special_defense": 60, "speed": 140},
            "maxHp": 110,
        },
        "defender": base_defender(),
        "move": {"name": "Thunderbolt", "type": "electric", "category": "special", "power": 90, "accuracy": 100},
        "field": {},
    }
    clamperl_payload = {
        "attacker": {
            "name": "Clamperl", "level": 50, "types": ["water"], "item": "Deep Sea Tooth",
            "stats": {"hp": 110, "attack": 60, "defense": 80, "special_attack": 74, "special_defense": 60, "speed": 40},
            "maxHp": 110,
        },
        "defender": base_defender(),
        "move": {"name": "Surf", "type": "water", "category": "special", "power": 90, "accuracy": 100},
        "field": {},
    }

    pikachu_result = analyze_battle(pikachu_payload)
    clamperl_result = analyze_battle(clamperl_payload)

    assert pikachu_result["damage"]["attackStatUsed"] == 140
    assert "Light Ball" in pikachu_result["damage"]["notes"]
    assert clamperl_result["damage"]["attackStatUsed"] == 148
    assert "Deep Sea Tooth" in clamperl_result["damage"]["notes"]


def test_status_and_gender_abilities_affect_damage():
    payload = {
        "attacker": {
            "name": "Zangoose", "level": 50, "types": ["normal"], "ability": "Toxic Boost", "status": "poison", "gender": "male",
            "stats": {"hp": 150, "attack": 120, "defense": 80, "special_attack": 60, "special_defense": 80, "speed": 100},
            "maxHp": 150,
        },
        "defender": base_defender(gender="female"),
        "move": {"name": "Facade", "type": "normal", "category": "physical", "power": 70, "accuracy": 100},
        "field": {},
    }
    result = analyze_battle(payload)

    assert result["damage"]["attackStatUsed"] == 120
    assert result["damage"]["powerUsed"] == 210
    assert "Toxic Boost" in result["damage"]["notes"]

    payload["attacker"]["ability"] = "Rivalry"
    payload["attacker"]["status"] = ""
    result = analyze_battle(payload)

    assert result["damage"]["powerUsed"] == 52
    assert "Rivalry" in result["damage"]["notes"]


def test_species_type_boost_and_defensive_ability_additions():
    payload = {
        "attacker": {
            "name": "Dialga", "level": 50, "types": ["steel", "dragon"], "item": "Adamant Orb",
            "stats": {"hp": 180, "attack": 120, "defense": 120, "special_attack": 150, "special_defense": 100, "speed": 90},
            "maxHp": 180,
        },
        "defender": base_defender(types=["fairy"], ability="Dry Skin", maxHp=180, currentHp=180),
        "move": {"name": "Flash Cannon", "type": "steel", "category": "special", "power": 80, "accuracy": 100},
        "field": {},
    }
    result = analyze_battle(payload)

    assert result["damage"]["powerUsed"] == 96
    assert "Adamant Orb species boost" in result["damage"]["notes"]

    payload["move"] = {"name": "Flamethrower", "type": "fire", "category": "special", "power": 90, "accuracy": 100}
    result = analyze_battle(payload)

    assert result["damage"]["powerUsed"] == 112
    assert "Dry Skin Fire weakness" in result["damage"]["notes"]


def test_ripen_strengthens_resist_berry():
    payload = {
        "attacker": {
            "name": "Arcanine", "level": 50, "types": ["fire"],
            "stats": {"hp": 170, "attack": 120, "defense": 90, "special_attack": 120, "special_defense": 90, "speed": 100},
            "maxHp": 170,
        },
        "defender": base_defender(types=["grass"], ability="Ripen", item="Occa Berry", maxHp=180, currentHp=180),
        "move": {"name": "Flamethrower", "type": "fire", "category": "special", "power": 90, "accuracy": 100},
        "field": {},
    }
    result = analyze_battle(payload)

    assert result["damage"]["modifiers"]["defenderItem"] == 0.25
    assert "Ripen doubles berry effect" in result["damage"]["notes"]


def test_analytic_stakeout_and_parental_bond():
    payload = {
        "attacker": {
            "name": "Kangaskhan", "level": 50, "types": ["normal"], "ability": "Parental Bond",
            "stats": {"hp": 180, "attack": 130, "defense": 100, "special_attack": 60, "special_defense": 100, "speed": 90},
            "maxHp": 180,
        },
        "defender": base_defender(),
        "move": {"name": "Return", "type": "normal", "category": "physical", "power": 102, "accuracy": 100},
        "field": {},
    }
    parental = analyze_battle(payload)

    payload["attacker"]["ability"] = "Analytic"
    payload["attacker"]["hasActed"] = True
    analytic = analyze_battle(payload)

    payload["attacker"]["ability"] = "Stakeout"
    payload["attacker"]["hasActed"] = False
    payload["defender"]["switchedInThisTurn"] = True
    stakeout = analyze_battle(payload)

    assert "Parental Bond second hit approximated at 25%." in parental["damage"]["notes"]
    assert analytic["damage"]["powerUsed"] == 133
    assert stakeout["damage"]["attackStatUsed"] == 260


def test_skill_link_slow_start_and_terrain_seed():
    payload = {
        "attacker": {
            "name": "Cloyster", "level": 50, "types": ["water", "ice"], "ability": "Skill Link",
            "stats": {"hp": 140, "attack": 150, "defense": 180, "special_attack": 90, "special_defense": 70, "speed": 90},
            "maxHp": 140,
        },
        "defender": base_defender(item="Grassy Seed"),
        "move": {"name": "Icicle Spear", "type": "ice", "category": "physical", "power": 25, "accuracy": 100},
        "field": {"terrain": "grassy"},
    }
    result = analyze_battle(payload)

    assert result["damage"]["hitCountsConsidered"] == [5]
    assert result["damage"]["defenseStatUsed"] == 180
    assert "Grassy Seed" in result["damage"]["notes"]

    payload["attacker"]["name"] = "Regigigas"
    payload["attacker"]["ability"] = "Slow Start"
    payload["attacker"]["stats"]["attack"] = 200
    payload["attacker"]["stats"]["speed"] = 100
    payload["move"] = {"name": "Tackle", "type": "normal", "category": "physical", "power": 40, "accuracy": 100}
    result = analyze_battle(payload)

    assert result["damage"]["attackStatUsed"] == 100
    assert result["speed"]["user"]["modifiedSpeed"] == 50


def test_entry_field_abilities_accuracy_and_residuals():
    payload = {
        "attacker": {
            "name": "Pelipper", "level": 50, "types": ["water", "flying"], "ability": "Drizzle",
            "stats": {"hp": 160, "attack": 70, "defense": 100, "special_attack": 120, "special_defense": 90, "speed": 85},
            "maxHp": 160,
        },
        "defender": base_defender(status="poison"),
        "move": {"name": "Hydro Pump", "type": "water", "category": "special", "power": 110, "accuracy": 80},
        "field": {},
    }
    result = analyze_battle(payload)

    assert result["damage"]["modifiers"]["weather"] == 1.5

    payload["defender"]["ability"] = "Dry Skin"
    result = analyze_battle(payload)
    assert result["damage"]["maxDamage"] == 0
    assert result["damage"]["residualOnDefenderEndTurn"]["dry_skin_rain"] > 0

    payload["attacker"]["ability"] = "Victory Star"
    payload["defender"]["ability"] = ""
    payload["defender"]["item"] = "Lax Incense"
    result = analyze_battle(payload)

    assert "Victory Star" in result["damage"]["accuracy"]["notes"]
    assert "Lax Incense" in result["damage"]["accuracy"]["notes"]


def test_disguise_ice_face_and_delta_stream():
    payload = {
        "attacker": {
            "name": "Tyranitar", "level": 50, "types": ["rock", "dark"],
            "stats": {"hp": 180, "attack": 160, "defense": 120, "special_attack": 95, "special_defense": 120, "speed": 80},
            "maxHp": 180,
        },
        "defender": base_defender(types=["ghost", "fairy"], ability="Disguise"),
        "move": {"name": "Stone Edge", "type": "rock", "category": "physical", "power": 100, "accuracy": 80},
        "field": {},
    }
    result = analyze_battle(payload)

    assert result["damage"]["maxDamage"] == 0
    assert "Disguise blocks this hit." in result["damage"]["notes"]

    payload["defender"] = base_defender(types=["flying"], ability="Delta Stream")
    payload["field"] = {}
    result = analyze_battle(payload)

    assert result["damage"]["modifiers"]["type"] == 1.0
    assert "Delta Stream removes Flying-type super-effective weakness." in result["damage"]["notes"]


def test_busted_mimikyu_does_not_block_damage_with_disguise():
    payload = {
        "attacker": {
            "name": "Dragonite Mega",
            "level": 50,
            "types": ["dragon", "flying"],
            "stats": {
                "hp": 198,
                "attack": 139,
                "defense": 139,
                "special_attack": 207,
                "special_defense": 146,
                "speed": 125,
            },
            "maxHp": 198,
        },
        "defender": {
            "name": "Mimikyu Busted",
            "level": 50,
            "types": ["ghost", "fairy"],
            "ability": "Disguise",
            "stats": {
                "hp": 130,
                "attack": 110,
                "defense": 100,
                "special_attack": 70,
                "special_defense": 125,
                "speed": 116,
            },
            "maxHp": 130,
        },
        "move": {"name": "Hurricane"},
        "field": {"isDoubles": True},
    }

    result = analyze_battle(payload)
    damage = result["damage"]

    assert damage["moveType"] == "flying"
    assert damage["modifiers"]["type"] == 1.0
    assert damage["minDamage"] > 0
    assert "Disguise blocks this hit." not in damage["notes"]


def test_triggered_items_and_focus_band():
    payload = {
        "attacker": {
            "name": "Lucario", "level": 50, "types": ["fighting", "steel"], "item": "Weakness Policy", "usedItem": True,
            "stats": {"hp": 150, "attack": 120, "defense": 90, "special_attack": 110, "special_defense": 90, "speed": 100},
            "maxHp": 150,
        },
        "defender": base_defender(item="Focus Band", usedItem=True, currentHp=50, maxHp=186),
        "move": {"name": "Close Combat", "type": "fighting", "category": "physical", "power": 120, "accuracy": 100},
        "field": {},
    }
    result = analyze_battle(payload)

    assert result["damage"]["attackStatUsed"] == 240
    assert result["damage"]["maxDamage"] == 49
    assert "Weakness Policy activated" in result["damage"]["notes"]
    assert "Focus Band activated and leaves the defender at 1 HP." in result["damage"]["notes"]


def test_protean_stab_and_item_move_types():
    payload = {
        "attacker": {
            "name": "Greninja", "level": 50, "types": ["water", "dark"], "ability": "Protean",
            "stats": {"hp": 150, "attack": 100, "defense": 80, "special_attack": 120, "special_defense": 80, "speed": 180},
            "maxHp": 150,
        },
        "defender": base_defender(types=["normal"]),
        "move": {"name": "Ice Beam", "type": "ice", "category": "special", "power": 90, "accuracy": 100},
        "field": {},
    }
    result = analyze_battle(payload)
    assert result["damage"]["modifiers"]["stab"] == 1.5

    payload["attacker"]["name"] = "Silvally"
    payload["attacker"]["ability"] = "RKS System"
    payload["attacker"]["item"] = "Fire Memory"
    payload["move"] = {"name": "Multi-Attack", "type": "normal", "category": "physical", "power": 120, "accuracy": 100}
    result = analyze_battle(payload)
    assert result["damage"]["moveType"] == "fire"
    assert "Multi-Attack type follows held Memory." in result["damage"]["notes"]


def test_entry_intimidate_guard_dog_and_contact_chip():
    payload = {
        "attacker": {
            "name": "Tauros", "level": 50, "types": ["normal"], "ability": "Intimidate",
            "stats": {"hp": 150, "attack": 120, "defense": 90, "special_attack": 60, "special_defense": 80, "speed": 110},
            "maxHp": 150,
        },
        "defender": base_defender(ability="Rough Skin", item="Rocky Helmet"),
        "move": {"name": "Take Down", "type": "normal", "category": "physical", "power": 90, "accuracy": 85, "makesContact": True},
        "field": {},
    }
    result = analyze_battle(payload)

    assert result["damage"]["contactDamageToAttacker"]["rough_skin"] == 18
    assert result["damage"]["contactDamageToAttacker"]["rocky_helmet"] == 25

    payload["defender"]["ability"] = "Guard Dog"
    payload["defender"]["item"] = ""
    result = analyze_battle(payload)

    assert result["damage"]["defenseStatUsed"] == 120
    assert result["speed"]["opponent"]["boostStage"] == 0


def test_download_simple_metal_powder_and_speed_items():
    payload = {
        "attacker": {
            "name": "Porygon-Z", "level": 50, "types": ["normal"], "ability": "Download",
            "stats": {"hp": 150, "attack": 80, "defense": 80, "special_attack": 150, "special_defense": 80, "speed": 100},
            "maxHp": 150,
        },
        "defender": base_defender(stats={"hp": 186, "attack": 92, "defense": 120, "special_attack": 105, "special_defense": 80, "speed": 98}),
        "move": {"name": "Tri Attack", "type": "normal", "category": "special", "power": 80, "accuracy": 100},
        "field": {},
    }
    result = analyze_battle(payload)
    assert result["damage"]["attackStatUsed"] == 225
    assert "Download" in result["damage"]["notes"]

    payload["attacker"]["ability"] = "Simple"
    payload["attacker"]["boosts"] = {"special_attack": 1}
    result = analyze_battle(payload)
    assert result["damage"]["attackStatUsed"] == 300

    payload["attacker"]["ability"] = ""
    payload["attacker"]["item"] = "Salac Berry"
    payload["attacker"]["usedItem"] = True
    result = analyze_battle(payload)
    assert result["speed"]["user"]["modifiedSpeed"] == 150


if __name__ == "__main__":
    import json
    sample = {
        "attacker": {
            "name": "Charizard", "level": 50, "types": ["fire", "flying"], "ability": "Solar Power", "item": "Life Orb",
            "nature": "Modest",
            "baseStats": {"hp": 78, "attack": 84, "defense": 78, "special_attack": 109, "special_defense": 85, "speed": 100},
            "evs": {"special_attack": 252, "speed": 252},
            "teraType": "fire", "isTerastallized": True
        },
        "defender": {"name": "Venusaur", "level": 50, "types": ["grass", "poison"], "ability": "Thick Fat", "item": "Assault Vest", "stats": {"hp": 187, "attack": 100, "defense": 103, "special_attack": 122, "special_defense": 120, "speed": 100}, "maxHp": 187, "currentHp": 187},
        "move": {"name": "Flamethrower"},
        "field": {"weather": "sun"}
    }
    print(json.dumps(analyze_battle(sample), indent=2))
