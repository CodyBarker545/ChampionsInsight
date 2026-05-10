from services.pokemon_stats_service import get_level_50_stats


def build_opponent_team(detected_pokemon, default_nature="hardy"):
    opponent_team = []

    for pokemon in detected_pokemon:
        name = pokemon.get("name")

        if not name:
            continue

        stats_data = get_level_50_stats(name, default_nature)

        opponent_team.append({
            "slot": pokemon.get("slot"),
            "name": name,
            "confidence": pokemon.get("confidence"),
            "image": pokemon.get("image"),
            "stats": stats_data
        })

    return opponent_team