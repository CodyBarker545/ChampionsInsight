"""Run opponent screenshots through DINOv2/FAISS indexes and write a PDF report."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from paths import CV_INDEX_DIR, UPLOAD_DIR
from services import cv_service, slot_object_detection_service
from services.dinov2_faiss_service import Dinov2FaissIndex


DEFAULT_INDEX_ROOT = CV_INDEX_DIR / "dinov2_faiss"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "data" / "cv" / "reports"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def cv_to_pil(image) -> Image.Image:
    cv2, _np = cv_service.load_cv_dependencies()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb).convert("RGB")


def thumbnail(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    copy = image.copy()
    copy.thumbnail(size, Image.Resampling.LANCZOS)
    return copy


def draw_wrapped(draw: ImageDraw.ImageDraw, xy, text: str, font, fill=(20, 20, 20), width=72, line_gap=4):
    x, y = xy
    lines = []
    for raw_line in str(text).splitlines():
        lines.extend(textwrap.wrap(raw_line, width=width) or [""])
    for line in lines:
        draw.text((x, y), line, font=font, fill=fill)
        bbox = draw.textbbox((x, y), line or " ", font=font)
        y += (bbox[3] - bbox[1]) + line_gap
    return y


def format_hits(hits: list[dict], title: str) -> str:
    lines = [title]
    for index, hit in enumerate(hits, start=1):
        lines.append(
            f"{index}. {hit.get('label', '')}  sim={float(hit.get('similarity', 0)):.3f}"
        )
    return "\n".join(lines)


def classify_slot(slot_image, pokemon_index: Dinov2FaissIndex, type_index: Dinov2FaissIndex) -> dict:
    object_layer = slot_object_detection_service.detect_slot_objects(slot_image)
    pokemon_object = object_layer.get("pokemon_sprite") or cv_service.detect_pokemon_sprite_object(slot_image)
    pokemon_crop = None

    if pokemon_object and not cv_service.is_empty_image(pokemon_object.get("image")):
        pokemon_crop = pokemon_object["image"]
    else:
        fallback_crop = cv_service.extract_opponent_pokemon_region(slot_image)
        if not cv_service.is_empty_image(fallback_crop):
            pokemon_crop = fallback_crop

    type_icon_crops = slot_object_detection_service.get_type_icon_crops(object_layer)
    type_crop = cv_service.build_type_combo_candidate_crop(slot_image, type_icon_crops=type_icon_crops)
    if cv_service.is_empty_image(type_crop):
        type_crop = cv_service.extract_type_icon_region(slot_image)

    pokemon_hits = []
    type_hits = []
    if pokemon_crop is not None and not cv_service.is_empty_image(pokemon_crop):
        pokemon_hits = pokemon_index.search(cv_to_pil(pokemon_crop), top_k=5)
    if type_crop is not None and not cv_service.is_empty_image(type_crop):
        type_hits = type_index.search(cv_to_pil(type_crop), top_k=5)

    return {
        "pokemonCrop": pokemon_crop,
        "typeCrop": type_crop,
        "pokemonHits": pokemon_hits,
        "typeHits": type_hits,
        "pokemonPrediction": pokemon_hits[0]["label"] if pokemon_hits else "unknown",
        "typePrediction": type_hits[0]["label"] if type_hits else "unknown",
    }


def collect_input_images(input_dirs: list[Path], limit: int | None, recursive: bool = True) -> list[Path]:
    images = []
    for input_dir in input_dirs:
        input_dir = Path(input_dir)
        iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
        images.extend(
            path
            for path in sorted(iterator)
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
    images = sorted(dict.fromkeys(images))
    return images[:limit] if limit else images


def save_pdf_chunks(pages: list[Image.Image], output_pdf: Path, pages_per_pdf: int) -> list[Path]:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    if not pages:
        return []

    if len(pages) <= pages_per_pdf:
        pages[0].save(output_pdf, save_all=True, append_images=pages[1:], resolution=120.0)
        return [output_pdf]

    output_paths = []
    for start in range(0, len(pages), pages_per_pdf):
        chunk = pages[start:start + pages_per_pdf]
        part_number = (start // pages_per_pdf) + 1
        chunk_path = output_pdf.with_name(f"{output_pdf.stem}_part{part_number:03d}{output_pdf.suffix}")
        chunk[0].save(chunk_path, save_all=True, append_images=chunk[1:], resolution=120.0)
        output_paths.append(chunk_path)
    return output_paths


def top_hit_similarity(hits: list[dict]) -> float:
    if not hits:
        return 0.0
    return round(float(hits[0].get("similarity", 0) or 0), 4)


def top_hit_labels(hits: list[dict]) -> str:
    return " | ".join(
        f"{hit.get('label', '')}:{float(hit.get('similarity', 0) or 0):.3f}"
        for hit in hits[:5]
    )


def write_review_csv(json_results: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "slot",
        "pokemon_pred",
        "pokemon_similarity",
        "pokemon_ok",
        "pokemon_crop_ok",
        "pokemon_top5",
        "type_pred",
        "type_similarity",
        "type_ok",
        "type_crop_ok",
        "type_top5",
        "failure_cause",
        "notes",
    ]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for image_record in json_results:
            image_name = Path(image_record["image"]).name
            for slot in image_record["slots"]:
                writer.writerow({
                    "image": image_name,
                    "slot": slot["position"],
                    "pokemon_pred": slot["pokemonPrediction"],
                    "pokemon_similarity": top_hit_similarity(slot["pokemonHits"]),
                    "pokemon_ok": "",
                    "pokemon_crop_ok": "",
                    "pokemon_top5": top_hit_labels(slot["pokemonHits"]),
                    "type_pred": slot["typePrediction"],
                    "type_similarity": top_hit_similarity(slot["typeHits"]),
                    "type_ok": "",
                    "type_crop_ok": "",
                    "type_top5": top_hit_labels(slot["typeHits"]),
                    "failure_cause": "",
                    "notes": "",
                })


def build_report_page(image_path: Path, slots: list[dict]) -> Image.Image:
    page = Image.new("RGB", (1650, 2200), "white")
    draw = ImageDraw.Draw(page)
    title_font = ImageFont.truetype("arial.ttf", 34)
    header_font = ImageFont.truetype("arial.ttf", 24)
    body_font = ImageFont.truetype("arial.ttf", 18)
    small_font = ImageFont.truetype("arial.ttf", 15)

    draw.text((48, 34), f"DINOv2 + FAISS opponent test", font=title_font, fill=(10, 10, 10))
    draw.text((48, 78), image_path.name, font=body_font, fill=(65, 65, 65))

    original = thumbnail(Image.open(image_path).convert("RGB"), (520, 280))
    page.paste(original, (1080, 34))

    y = 340
    row_height = 285
    for slot in slots:
        draw.rounded_rectangle(
            (36, y - 16, 1614, y + row_height - 18),
            radius=8,
            outline=(205, 205, 205),
            width=2,
            fill=(250, 250, 250),
        )

        draw.text((58, y), f"Slot {slot['position']}", font=header_font, fill=(20, 20, 20))

        slot_thumb = thumbnail(slot["slotImage"], (210, 210))
        page.paste(slot_thumb, (58, y + 42))
        draw.text((58, y + 258), "red slot", font=small_font, fill=(70, 70, 70))

        if slot["pokemonCrop"] is not None:
            pokemon_thumb = thumbnail(cv_to_pil(slot["pokemonCrop"]), (190, 190))
            page.paste(pokemon_thumb, (300, y + 52))
        draw.text((300, y + 258), "pokemon crop", font=small_font, fill=(70, 70, 70))

        if slot["typeCrop"] is not None and not cv_service.is_empty_image(slot["typeCrop"]):
            type_thumb = thumbnail(cv_to_pil(slot["typeCrop"]), (220, 120))
            page.paste(type_thumb, (535, y + 88))
        draw.text((535, y + 258), "type crop", font=small_font, fill=(70, 70, 70))

        prediction_text = (
            f"Pokemon: {slot['pokemonPrediction']}\n"
            f"Type: {slot['typePrediction']}\n\n"
            f"{format_hits(slot['pokemonHits'][:3], 'Pokemon top 3')}\n\n"
            f"{format_hits(slot['typeHits'][:3], 'Type top 3')}"
        )
        draw_wrapped(draw, (805, y + 4), prediction_text, body_font, width=58)
        y += row_height

    return page


def run_report(
    input_dirs: list[Path],
    output_pdf: Path,
    output_json: Path,
    output_csv: Path,
    limit: int | None,
    pages_per_pdf: int,
) -> dict:
    pokemon_index = Dinov2FaissIndex(
        DEFAULT_INDEX_ROOT / "pokemon" / "index.faiss",
        DEFAULT_INDEX_ROOT / "pokemon" / "metadata.json",
    )
    type_index = Dinov2FaissIndex(
        DEFAULT_INDEX_ROOT / "types" / "index.faiss",
        DEFAULT_INDEX_ROOT / "types" / "metadata.json",
    )

    image_paths = collect_input_images(input_dirs, limit)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dirs}")

    pages = []
    json_results = []

    for image_number, image_path in enumerate(image_paths, start=1):
        print(f"[{image_number}/{len(image_paths)}] {image_path.name}")
        crops = cv_service.crop_opponent_team_slots(image_path, save_debug=True)

        slots = []
        for crop in crops:
            result = classify_slot(crop["image"], pokemon_index, type_index)
            slot_record = {
                "position": crop["position"],
                "slotImage": cv_to_pil(crop["image"]),
                **result,
            }
            slots.append(slot_record)

        pages.append(build_report_page(image_path, slots))
        json_results.append({
            "image": str(image_path),
            "slots": [
                {
                    "position": slot["position"],
                    "pokemonPrediction": slot["pokemonPrediction"],
                    "typePrediction": slot["typePrediction"],
                    "pokemonHits": [
                        {
                            "label": hit.get("label"),
                            "similarity": round(float(hit.get("similarity", 0)), 4),
                            "path": hit.get("path"),
                        }
                        for hit in slot["pokemonHits"][:5]
                    ],
                    "typeHits": [
                        {
                            "label": hit.get("label"),
                            "similarity": round(float(hit.get("similarity", 0)), 4),
                            "path": hit.get("path"),
                        }
                        for hit in slot["typeHits"][:5]
                    ],
                }
                for slot in slots
            ],
        })

    pdf_paths = save_pdf_chunks(pages, output_pdf, pages_per_pdf=pages_per_pdf)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(json_results, indent=2), encoding="utf-8")
    write_review_csv(json_results, output_csv)

    return {
        "imageCount": len(image_paths),
        "pdfs": [str(path) for path in pdf_paths],
        "json": str(output_json),
        "csv": str(output_csv),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", action="append", type=Path, default=[])
    parser.add_argument("--output-pdf", type=Path, default=DEFAULT_OUTPUT_DIR / "dinov2_faiss_opponent_report.pdf")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_DIR / "dinov2_faiss_opponent_report.json")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_DIR / "dinov2_faiss_opponent_review.csv")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--pages-per-pdf", type=int, default=40)
    args = parser.parse_args()

    result = run_report(
        input_dirs=args.input_dir or [UPLOAD_DIR],
        output_pdf=args.output_pdf,
        output_json=args.output_json,
        output_csv=args.output_csv,
        limit=args.limit,
        pages_per_pdf=args.pages_per_pdf,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

