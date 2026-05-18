"""
Create an HTML preview page showing the first non-synthetic image from each image folder.

Run from backend:

python scripts/make_image_folder_preview.py

Output:
image_folder_preview.html

Open that file in your browser.
"""

from pathlib import Path
from collections import defaultdict
import html


BACKEND_DIR = Path(__file__).resolve().parents[2]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

SCAN_ROOTS = [
    BACKEND_DIR / "data",
    BACKEND_DIR / "uploads",
    BACKEND_DIR / "cv",
]

IGNORE_PARTS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
}

SYNTHETIC_NAME_MARKERS = {
    "_synthetic_",
    "synthetic",
}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def should_ignore_path(path: Path) -> bool:
    return any(part in IGNORE_PARTS for part in path.parts)


def is_synthetic_image(path: Path) -> bool:
    name = path.name.lower()
    return any(marker in name for marker in SYNTHETIC_NAME_MARKERS)


def guess_purpose(folder: Path) -> str:
    text = str(folder).lower()

    if "slot_pokemon_real" in text:
        return "Real sorted PokÃ©mon classifier data"
    if "reald" in text or "real" in text:
        return "Real sorted PokÃ©mon classifier data"
    if "unsorted_pokemon" in text:
        return "Unsorted PokÃ©mon crops; needs labeling"
    if "unsorted_slots" in text:
        return "Unsorted slot/card crops; useful after sorting"
    if "unsorted_type" in text:
        return "Unsorted type icon/type combo crops"
    if "champions_sprites" in text:
        return "Clean Champions source sprites"
    if "debug" in text or "crop" in text:
        return "Debug crops; inspect before using"
    if "upload" in text:
        return "Raw uploaded/full camera images"
    if "type" in text:
        return "Type icon/type combo data"
    if "sprite_detector" in text or "yolo" in text:
        return "Object detection dataset"

    return "Unknown / inspect manually"


def scan_non_synthetic_images():
    folder_images = defaultdict(list)
    synthetic_ignored = 0

    for root in SCAN_ROOTS:
        if not root.exists():
            continue

        for path in root.rglob("*"):
            if should_ignore_path(path):
                continue

            if not path.is_file() or not is_image(path):
                continue

            if is_synthetic_image(path):
                synthetic_ignored += 1
                continue

            folder_images[path.parent].append(path)

    return folder_images, synthetic_ignored


def make_windows_uri(path: Path) -> str:
    """
    Creates a browser-friendly file URI.
    """
    return path.resolve().as_uri()


def main():
    folder_images, synthetic_ignored = scan_non_synthetic_images()

    rows = []

    for folder, images in folder_images.items():
        images = sorted(images)

        if not images:
            continue

        try:
            relative_folder = folder.relative_to(BACKEND_DIR)
        except ValueError:
            relative_folder = folder

        first_image = images[0]

        rows.append({
            "folder": str(relative_folder),
            "count": len(images),
            "first_image_name": first_image.name,
            "first_image_uri": make_windows_uri(first_image),
            "purpose": guess_purpose(folder),
        })

    rows.sort(key=lambda row: row["count"], reverse=True)

    output_path = BACKEND_DIR / "image_folder_preview.html"

    cards_html = []

    for row in rows:
        folder = html.escape(row["folder"])
        first_image_name = html.escape(row["first_image_name"])
        purpose = html.escape(row["purpose"])
        image_uri = row["first_image_uri"]
        count = row["count"]

        cards_html.append(f"""
        <div class="card">
            <a href="{image_uri}" target="_blank">
                <img src="{image_uri}" alt="{first_image_name}">
            </a>
            <div class="info">
                <h2>{folder}</h2>
                <p><strong>Images:</strong> {count}</p>
                <p><strong>First image:</strong> {first_image_name}</p>
                <p><strong>Guess:</strong> {purpose}</p>
            </div>
        </div>
        """)

    page = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>ChampionsInsight Image Folder Preview</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: #151522;
            color: #f3f3f6;
            margin: 0;
            padding: 24px;
        }}

        h1 {{
            margin-bottom: 4px;
        }}

        .summary {{
            color: #c8c8d0;
            margin-bottom: 24px;
        }}

        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
            gap: 18px;
        }}

        .card {{
            background: #232336;
            border: 1px solid #34344d;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 20px rgba(0,0,0,0.25);
        }}

        .card img {{
            width: 100%;
            height: 210px;
            object-fit: contain;
            background: #11111a;
            display: block;
        }}

        .info {{
            padding: 14px;
        }}

        .info h2 {{
            font-size: 15px;
            line-height: 1.35;
            margin: 0 0 10px;
            word-break: break-all;
            color: #ffffff;
        }}

        .info p {{
            margin: 6px 0;
            color: #d6d6df;
            font-size: 14px;
        }}

        strong {{
            color: #ffffff;
        }}
    </style>
</head>
<body>
    <h1>ChampionsInsight Image Folder Preview</h1>
    <div class="summary">
        <p>Backend: {html.escape(str(BACKEND_DIR))}</p>
        <p>Folders shown: {len(rows)}</p>
        <p>Synthetic images ignored: {synthetic_ignored}</p>
    </div>

    <div class="grid">
        {''.join(cards_html)}
    </div>
</body>
</html>
"""

    output_path.write_text(page, encoding="utf-8")

    print("Preview created.")
    print(f"Open this file:")
    print(output_path)


if __name__ == "__main__":
    main()
