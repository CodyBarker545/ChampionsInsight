from pokemon_damage_calculator import analyze_battle
import json


def run_basic_test():
    payload = {
        "attacker": {
        "name": "Charizard",
        "level": 50,
        "types": ["fire", "flying"],
        "nature": "Timid",
        "baseStats": {
            "hp": 78,
            "attack": 84,
            "defense": 78,
            "special_attack": 109,
            "special_defense": 85,
            "speed": 100
        },
        "ivs": {"special_attack": 31},
        "evs": {"special_attack": 252}
    },  
        "defender": {
            "name": "Venusaur",
            "level": 50,
            "types": ["grass", "poison"],
            "stats": {
                "hp": 187,
                "attack": 100,
                "defense": 103,
                "special_attack": 122,
                "special_defense": 120,
                "speed": 100
            },
            "maxHp": 187,
            "currentHp": 187
        },
        "move": {
            "name": "Flamethrower",
            "type": "fire",
            "category": "special",
            "power": 90,
            "accuracy": 100
        },
        "field": {}
    }

    result = analyze_battle(payload)

    print("\n=== BASIC DAMAGE TEST ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run_basic_test()