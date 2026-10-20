import json
from pathlib import Path

from flask import Flask
from PIL import Image

from scripts.data_build.sync_champions_catalog import category_members, local_filename
from scripts.synthetic_data import generate_synthetic_pokemon_cards as generator
from services.pokedex_service import get_all_pokedex_entries
from paths import POKEMON_DATA_DIR, SPRITE_ROOT


def test_category_follows_continuation():
    class Session:
        def __init__(self):
            self.params = []

        def get(self, url, params, timeout):
            self.params.append(dict(params))
            data = {"query": {"categorymembers": [{"ns": 6, "title": f"File:{len(self.params)}.png"}]}}
            if len(self.params) == 1:
                data["continue"] = {"cmcontinue": "second-page", "continue": "-||"}

            class Response:
                def raise_for_status(self):
                    pass

                def json(self):
                    return data

            return Response()

    session = Session()
    assert category_members(session, "Champions menu sprites") == ["File:1.png", "File:2.png"]
    assert session.params[1]["cmcontinue"] == "second-page"


def test_archive_rename_preserves_legacy_filename():
    assert local_filename("File:Menu Champs 0931-Blue.png") == "Menu_CP_0931-Blue.png"
    assert local_filename("File:Menu Champs 0666-Poké Ball.png") == "Menu_CP_0666-Pok%C3%A9_Ball.png"


def test_new_sprites_resolve_to_exact_pokedex_and_training_forms():
    report = json.loads((POKEMON_DATA_DIR / "champions_catalog_update.json").read_text(encoding="utf-8"))
    entries = {entry["spriteFilename"]: entry for entry in get_all_pokedex_entries()}
    for record in report["added_sprites"]:
        filename = record["local_filename"]
        with Image.open(SPRITE_ROOT / record["sprite_type"] / filename) as image:
            image.verify()
        if not record["is_shiny"]:
            assert entries[filename]["formApiName"] == record["form_api_name"]
        expected = None if "mega" in filename.lower() else generator.normalize_label(record["form_api_name"])
        assert generator.parse_label(Path(filename)) == expected


def test_resuming_a_pruned_dataset_preserves_existing_images(tmp_path, monkeypatch):
    source = tmp_path / "sprites"
    source.mkdir()
    sprite = source / "Menu_CP_0979.png"
    Image.new("RGBA", (32, 32)).save(sprite)
    output = tmp_path / "output"
    folder = output / "annihilape"
    folder.mkdir(parents=True)
    for index in (0, 2):
        Image.new("RGB", (32, 32), (index, 0, 0)).save(folder / f"Menu_CP_0979_synthetic_{index:04d}.jpg")
    existing = (folder / "Menu_CP_0979_synthetic_0002.jpg").read_bytes()
    monkeypatch.setattr(generator, "paste_sprite_on_card", lambda path: Image.new("RGB", (32, 32), "red"))
    generator.generate_dataset(source, output, None, 3, 10, 0, None)
    assert len(list(folder.glob("*.jpg"))) == 3
    assert (folder / "Menu_CP_0979_synthetic_0002.jpg").read_bytes() == existing
    assert (folder / "Menu_CP_0979_synthetic_0003.jpg").exists()


def test_held_item_api_contains_new_and_initial_items():
    from api.pokemon_routes import pokemon_bp
    app = Flask(__name__)
    app.register_blueprint(pokemon_bp, url_prefix="/api")
    response = app.test_client().get("/api/items")
    assert response.status_code == 200
    items = {item["name"]: item for item in response.json["items"]}
    assert {"grassy-seed", "rocky-helmet", "leek", "absolite-z", "focus-sash", "sitrus-berry"} <= items.keys()
    assert items["absolite-z"]["isMegaStone"] is True
