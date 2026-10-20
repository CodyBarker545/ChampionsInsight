"""Refresh Champions sprites and held items from Bulbagarden; safe to rerun.

Uses MediaWiki continuation (all category pages), preserves legacy Menu_CP file
names, and only downloads missing assets. Run with --apply to write changes.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import sys
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BACKEND = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BACKEND))
from paths import POKEMON_DATA_DIR, SPRITE_ROOT, SPRITE_METADATA_PATH

API = "https://archives.bulbagarden.net/w/api.php"
SHOP = "https://bulbapedia.bulbagarden.net/wiki/Shop_(Champions)"
ITEM_LIST = "https://www.serebii.net/pokemonchampions/items.shtml"
REPORT = POKEMON_DATA_DIR / "champions_catalog_update.json"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def category_members(session, category):
    params = dict(action="query", list="categorymembers", cmtitle="Category:" + category,
                  cmlimit=500, format="json")
    result = []
    while True:
        response = session.get(API, params=params, timeout=60)
        response.raise_for_status()
        data = response.json()
        result.extend(x["title"] for x in data["query"]["categorymembers"] if x["ns"] == 6)
        if "continue" not in data:
            return result
        params.update(data["continue"])


def local_filename(title):
    # Retain URL-escaped legacy cosmetic names as well as the original CP prefix.
    name = unquote(title.removeprefix("File:")).replace("Menu Champs ", "Menu CP ")
    name = name.replace("Menu_CP_", "Menu CP ")
    name = name.replace("Poké", "Pok%C3%A9")
    return name.replace(" ", "_")


def sprite_identity(filename):
    name = unquote(filename).lower().replace("menu_champs_", "menu_cp_")
    return name


def find_form(lookup, filename):
    match = re.fullmatch(r"Menu_CP_(\d+)(?:-(.*?))?(?:_shiny)?\.png", unquote(filename))
    if not match:
        raise ValueError(f"Unrecognized sprite: {filename}")
    dex, suffix = match.groups()
    suffix = (suffix or "").removesuffix("_shiny").replace("_", " ").lower()
    forms = lookup[dex]
    if not suffix:
        return dex, suffix, next((x for x in forms if x.get("is_default")), forms[0])
    aliases = {"alola": "alola", "galar": "galar", "hisui": "hisui",
               "paldea combat": "paldea combat breed", "paldea blaze": "paldea blaze breed",
               "paldea aqua": "paldea aqua breed", "blue": "blue plumage",
               "white": "white plumage", "yellow": "yellow plumage"}
    suffix_key = aliases.get(suffix, suffix)
    candidates = []
    for form in forms:
        api = form["form_api_name"].replace("-", " ")
        display = form["form_display_name"].lower().replace("-", " ")
        if api.endswith(" " + suffix_key) or display.endswith(" " + suffix_key):
            candidates.append(form)
    if len(candidates) == 1:
        return dex, suffix, candidates[0]
    # Pyroar's sex changes appearance only; PokeAPI stores just the male form.
    if dex == "0668" and suffix == "female":
        form = copy.deepcopy(next(x for x in forms if x["form_api_name"] == "pyroar-male"))
        form.update(form_api_name="pyroar-female", form_display_name="Pyroar Female", is_default=False)
        return dex, suffix, form
    raise ValueError(f"No unambiguous form mapping: {filename}: {[x['form_api_name'] for x in candidates]}")


def image_urls(session, titles):
    result = {}
    for start in range(0, len(titles), 50):
        response = session.get(API, params=dict(action="query", titles="|".join(titles[start:start + 50]),
                                               prop="imageinfo", iiprop="url", format="json"), timeout=60)
        response.raise_for_status()
        for page in response.json()["query"]["pages"].values():
            if "imageinfo" in page:
                result[page["title"]] = page["imageinfo"][0]["url"]
    return result


def parse_shop(html):
    soup = BeautifulSoup(html, "html.parser")
    items = {}
    for section in ("Held_Items", "Mega_Stones"):
        heading = soup.find(id=section)
        if heading is None:
            raise ValueError(f"Shop section missing: {section}")
        for element in heading.parent.next_siblings:
            if getattr(element, "name", None) in {"h2", "h3"}:
                break
            if not hasattr(element, "find_all"):
                continue
            for image in element.find_all("img"):
                name = image.get("alt", "")
                if not name or "Sprite" in name:
                    continue
                key = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
                # Sprite cells carry the real item name in alt text.
                parent = image.find_parent("table")
                price = re.search(r"([\d,]+)\s*VP", parent.get_text(" ", strip=True))
                if not price:
                    continue
                src = image.get("src", "")
                if "/thumb/" in src:
                    src = src.replace("/thumb/", "/").rsplit("/", 1)[0]
                items[key] = dict(name=key, display_name=name, vp_cost=int(price[1].replace(",", "")),
                                  is_mega_stone=section == "Mega_Stones", sprite=src, source=SHOP)
    if len(items) < 80:
        raise ValueError(f"Shop parse incomplete: only {len(items)} items")
    return items


def add_starting_items(items, html, dictionary):
    """The shop excludes initial inventory and Mega Evolution tutorial rewards."""
    soup = BeautifulSoup(html, "html.parser")
    for link in soup.select('a[href*="/itemdex/"]'):
        row = link.find_parent("tr")
        if row is None:
            continue
        cells = row.find_all("td", recursive=False)
        if len(cells) != 4:
            continue
        location = cells[-1].get_text(" ", strip=True)
        if "Beginning" not in location and "Mega Evolution Tutorial" not in location:
            continue
        name = link.get_text(" ", strip=True)
        if not name:
            continue
        key = re.sub(r"[^a-z0-9]+", "-", name.lower().replace("'", "")).strip("-")
        if key in items:
            continue
        if key not in dictionary:
            raise ValueError(f"Unknown starting item: {name}")
        data = dictionary[key]
        items[key] = dict(name=key, display_name=name, vp_cost=None,
                         is_mega_stone=data.get("category") == "mega-stones",
                         sprite=data.get("sprite", ""), source=ITEM_LIST)


def ensure_battle_forms(records, lookup):
    """Add cosmetic sex variants with identical mechanics to all runtime lookups."""
    battle_path = POKEMON_DATA_DIR / "pokemon_battle_data.json"
    battle = read_json(battle_path)
    by_api = {x["form_api_name"]: x for x in battle}
    added = []
    for sprite in records:
        api = sprite["form_api_name"]
        if api in by_api:
            continue
        if api != "pyroar-female":
            raise ValueError(f"Missing battle data for {api}")
        form = copy.deepcopy(by_api["pyroar-male"])
        form.update(next(x for x in lookup["0668"] if x["form_api_name"] == api))
        battle.append(form)
        by_api[api] = form
        added.append(form)
    if added:
        write_json(battle_path, battle)
        path = POKEMON_DATA_DIR / "pokemon_lookup_by_form_name.json"
        form_lookup = read_json(path)
        form_lookup.update({x["form_api_name"]: x for x in added})
        write_json(path, form_lookup)


def sync(apply=False):
    session = requests.Session()
    session.headers["User-Agent"] = "ChampionsInsight/1.0 (local catalog refresh)"
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1,
                  status_forcelist=[429, 500, 502, 503, 504])))
    lookup = read_json(POKEMON_DATA_DIR / "pokemon_lookup_by_dex.json")
    records = read_json(SPRITE_METADATA_PATH)
    existing = {sprite_identity(x["local_filename"]): x for x in records}
    additions, unresolved, category_counts = [], [], {}
    for category, variant in (("Champions menu sprites", "normal"), ("Champions Shiny menu sprites", "shiny")):
        titles = category_members(session, category)
        category_counts[variant] = len(titles)
        for title in titles:
            filename = local_filename(title)
            if sprite_identity(filename) in existing:
                continue
            try:
                dex, suffix, form = find_form(lookup, filename)
            except (ValueError, KeyError) as exc:
                unresolved.append(str(exc))
                continue
            additions.append(dict(display_name=form["form_display_name"],
                species_display_name=form["species_display_name"], form_display_name=form["form_display_name"],
                dex_number=dex, sprite_form=suffix.title(), form_api_name=form["form_api_name"],
                is_shiny=variant == "shiny", sprite_type=variant, types=form["types"], base_stats=form["base_stats"],
                original_filename=title.removeprefix("File:"), local_filename=filename,
                local_path=f"{variant}/{filename}", source_page="https://archives.bulbagarden.net/wiki/" + title.replace(" ", "_")))
    response = session.get(SHOP, timeout=60)
    response.raise_for_status()
    items = parse_shop(response.text)
    dictionary = read_json(POKEMON_DATA_DIR / "item_dictionary.json")
    response = session.get(ITEM_LIST, timeout=60)
    response.raise_for_status()
    add_starting_items(items, response.text, dictionary)
    missing_items = sorted(set(items) - set(dictionary))
    report = dict(category_counts=category_counts, added_sprites=additions, unresolved=unresolved,
                  shop_items=len(items), missing_dictionary_items=missing_items)
    print(json.dumps({**report, "added_sprites": [x["local_filename"] for x in additions]}, indent=2), flush=True)
    if unresolved:
        raise ValueError("Resolve form mappings before applying this refresh")
    if not apply:
        return report
    titles = ["File:" + x["original_filename"] for x in additions]
    urls = image_urls(session, titles)
    for index, record in enumerate(additions):
        title = "File:" + record["original_filename"]
        record["image_url"] = urls[title]
        destination = SPRITE_ROOT / record["sprite_type"] / record["local_filename"]
        if not destination.exists():
            response = session.get(record["image_url"], timeout=60)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            image.verify()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(response.content)
        print(f"[{index + 1}/{len(additions)}] {record['local_filename']}", flush=True)
        dex = record["dex_number"]
        if not any(x["form_api_name"] == record["form_api_name"] for x in lookup[dex]):
            lookup[dex].append(find_form(lookup, record["local_filename"])[2])
    records.extend(additions)
    write_json(SPRITE_METADATA_PATH, records)
    write_json(POKEMON_DATA_DIR / "pokemon_lookup_by_dex.json", lookup)
    ensure_battle_forms(records, lookup)
    with SPRITE_METADATA_PATH.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({**record, "types": "|".join(record["types"]), "base_stats": json.dumps(record["base_stats"])})
    for key, item in items.items():
        if key == "leek":
            dictionary[key] = {**dictionary.get(key, dictionary["stick"]), "name": key,
                "display_name": item["display_name"], "sprite": item["sprite"],
                "short_effect": "Raises the critical-hit ratio of Farfetch'd or Sirfetch'd by two stages.",
                "effect": "When held by Farfetch'd or Sirfetch'd, adds two stages to the critical-hit ratio."}
        if key not in dictionary:
            if not item["is_mega_stone"]:
                raise ValueError(f"Missing item mechanics: {key}")
            effect = "Allows the corresponding Pokemon to Mega Evolve when held in battle."
            dictionary[key] = dict(id=None, name=key, display_name=item["display_name"], category="mega-stones",
                                   cost=0, fling_power=None, fling_effect="", short_effect=effect,
                                   effect=effect, sprite=item["sprite"])
        item["short_effect"] = dictionary[key].get("short_effect", "")
    write_json(POKEMON_DATA_DIR / "item_dictionary.json", dictionary)
    write_json(POKEMON_DATA_DIR / "champions_held_items.json", items)
    # Preserve the last nonempty incremental manifest for resumable generation.
    if additions or not REPORT.exists():
        write_json(REPORT, report)
    rag_text = ("Champions available held items and Mega Stones\nSources: " + SHOP + "; " + ITEM_LIST +
        "\nThis catalog describes game availability, not ranked regulation legality.\n\n" +
        "\n".join(f"{x['display_name']}: {x['short_effect']}" for x in items.values()) + "\n")
    for folder in (BACKEND / "data/rag/docs", BACKEND / "RAGDocs"):
        (folder / "champions_available_items.txt").write_text(rag_text, encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Download new sprites and update the local catalogs")
    sync(parser.parse_args().apply)
