"""Current Champions held items, including initial inventory and Mega Stones."""
import json
from functools import lru_cache

from paths import POKEMON_DATA_DIR


@lru_cache(maxsize=1)
def get_champions_held_items():
    path = POKEMON_DATA_DIR / "champions_held_items.json"
    if not path.exists():
        return []
    items = json.loads(path.read_text(encoding="utf-8"))
    return sorted(({
        "name": item["name"],
        "displayName": item["display_name"],
        "shortEffect": item.get("short_effect", ""),
        "isMegaStone": item.get("is_mega_stone", False),
    } for item in items.values()), key=lambda item: item["displayName"])
