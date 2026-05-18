"""Generate real-like synthetic type/type-combo crops from sorted references.

The input is expected to be a class-folder dataset such as:

    data/training_dataset/type_combos/grass_fighting/*.jpg

For each class, the script augments real sorted crops when available and can
fall back to canonical generated references from cv/references/types/type_combo_icons.
The output keeps the same class-folder layout for classifier training.
"""

from __future__ import annotations

import argparse
import random
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = BACKEND_DIR / "data" / "training_dataset" / "type_combos"
DEFAULT_REFERENCE_DIR = BACKEND_DIR / "data" / "cv" / "references" / "types" / "type_combo_icons"
DEFAULT_OUTPUT_DIR = BACKEND_DIR / "data" / "training_dataset" / "type_combos_synthetic_visible"

CARD_BACKGROUNDS = [
    (122, 0, 50),
    (145, 0, 65),
    (158, 5, 75),
    (92, 92, 92),
    (54, 54, 54),
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def image_paths(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def make_background(width: int, height: int) -> Image.Image:
    base = np.zeros((height, width, 3), dtype=np.int16)
    base[:, :] = random.choice(CARD_BACKGROUNDS)

    x_gradient = np.linspace(-10, 10, width).astype(np.int16)
    y_gradient = np.linspace(-8, 8, height).astype(np.int16)
    base[:, :, 0] += x_gradient[None, :]
    base[:, :, 1] += y_gradient[:, None]
    base[:, :, 2] += x_gradient[None, :] // 2
    base += np.random.normal(0, random.uniform(1.5, 8.0), base.shape).astype(np.int16)

    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")


def trim_plain_border(image: Image.Image) -> Image.Image:
    if image.width < 6 or image.height < 6:
        return image

    rgb = image.convert("RGB")
    arr = np.array(rgb).astype(np.int16)
    corner_pixels = np.array(
        [
            arr[0, 0],
            arr[0, -1],
            arr[-1, 0],
            arr[-1, -1],
        ],
        dtype=np.int16,
    )
    background = np.median(corner_pixels, axis=0)
    diff = np.abs(arr - background[None, None, :]).sum(axis=2)
    mask = diff > 26

    ys, xs = np.where(mask)
    if not len(xs) or not len(ys):
        return image

    margin = random.randint(0, 3)
    left = max(0, int(xs.min()) - margin)
    top = max(0, int(ys.min()) - margin)
    right = min(image.width, int(xs.max()) + 1 + margin)
    bottom = min(image.height, int(ys.max()) + 1 + margin)

    if right - left < 12 or bottom - top < 12:
        return image

    return image.crop((left, top, right, bottom))


def paste_on_context(image: Image.Image) -> Image.Image:
    if random.random() < 0.35:
        image = trim_plain_border(image)

    scale = random.uniform(0.72, 1.34)
    resized = image.resize(
        (
            max(12, int(round(image.width * scale))),
            max(12, int(round(image.height * scale))),
        ),
        Image.Resampling.BICUBIC,
    )

    pad_x = random.randint(0, max(8, int(resized.width * 0.28)))
    pad_y = random.randint(0, max(6, int(resized.height * 0.22)))
    canvas_width = max(16, resized.width + pad_x * 2)
    canvas_height = max(16, resized.height + pad_y * 2)
    canvas = make_background(canvas_width, canvas_height)

    max_shift_x = max(1, int(resized.width * 0.09))
    max_shift_y = max(1, int(resized.height * 0.09))
    x = pad_x + random.randint(-max_shift_x, max_shift_x)
    y = pad_y + random.randint(-max_shift_y, max_shift_y)
    canvas.paste(resized, (x, y))

    return canvas


def crop_with_visible_center(image: Image.Image) -> Image.Image:
    width, height = image.size

    # Most crops are tight detector-like boxes. A few retain extra context.
    crop_width = int(round(width * random.uniform(0.78, 1.0)))
    crop_height = int(round(height * random.uniform(0.78, 1.0)))
    crop_width = max(16, min(width, crop_width))
    crop_height = max(16, min(height, crop_height))

    center_x = width // 2 + random.randint(
        -max(1, int(width * 0.08)),
        max(1, int(width * 0.08)),
    )
    center_y = height // 2 + random.randint(
        -max(1, int(height * 0.08)),
        max(1, int(height * 0.08)),
    )
    left = max(0, min(width - crop_width, center_x - crop_width // 2))
    top = max(0, min(height - crop_height, center_y - crop_height // 2))

    return image.crop((left, top, left + crop_width, top + crop_height))


def add_camera_effects(image: Image.Image) -> Image.Image:
    image = ImageEnhance.Brightness(image).enhance(random.uniform(0.62, 1.38))
    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.72, 1.42))
    image = ImageEnhance.Color(image).enhance(random.uniform(0.76, 1.30))

    if random.random() < 0.50:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.15, 1.15)))
    if random.random() < 0.16:
        image = image.filter(ImageFilter.SHARPEN)

    arr = np.array(image).astype(np.int16)
    if random.random() < 0.92:
        arr += np.random.normal(0, random.uniform(1.5, 10.0), arr.shape).astype(np.int16)

    if random.random() < 0.18:
        image_width, image_height = image.size
        glare_x = random.randint(0, image_width - 1)
        glare_y = random.randint(0, image_height - 1)
        radius = random.randint(
            max(8, int(min(image_width, image_height) * 0.22)),
            max(12, int(max(image_width, image_height) * 0.52)),
        )
        yy, xx = np.ogrid[:image_height, :image_width]
        dist = np.sqrt((xx - glare_x) ** 2 + (yy - glare_y) ** 2)
        mask = np.clip(1 - dist / radius, 0, 1)
        strength = random.uniform(12, 42)
        arr[:, :, 0] += (mask * strength).astype(np.int16)
        arr[:, :, 1] += (mask * strength).astype(np.int16)
        arr[:, :, 2] += (mask * strength).astype(np.int16)

    if random.random() < 0.22:
        every = random.randint(2, 5)
        arr[::every, :, :] -= random.randint(3, 10)

    image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    if random.random() < 0.65:
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=random.randint(46, 92))
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")

    if random.random() < 0.12:
        image = ImageOps.autocontrast(image, cutoff=random.uniform(0, 1.6))

    return image


def augment_type_crop(source_path: Path) -> Image.Image:
    image = Image.open(source_path).convert("RGB")
    image = paste_on_context(image)
    image = crop_with_visible_center(image)
    image = add_camera_effects(image)
    return image


def collect_class_sources(input_dir: Path, reference_dir: Path) -> dict[str, list[Path]]:
    class_names = {
        path.name
        for path in input_dir.iterdir()
        if path.is_dir()
    } if input_dir.exists() else set()

    if reference_dir.exists():
        class_names.update(path.stem for path in reference_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)

    sources_by_class: dict[str, list[Path]] = {}
    for class_name in sorted(class_names):
        sources = []
        sources.extend(image_paths(input_dir / class_name))

        reference_path = reference_dir / f"{class_name}.png"
        if reference_path.exists():
            sources.append(reference_path)

        if sources:
            sources_by_class[class_name] = sources

    return sources_by_class


def generate_dataset(
    input_dir: Path,
    reference_dir: Path,
    output_dir: Path,
    target_per_class: int,
    singles_only: bool,
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    sources_by_class = collect_class_sources(input_dir, reference_dir)
    if singles_only:
        sources_by_class = {
            class_name: sources
            for class_name, sources in sources_by_class.items()
            if "_" not in class_name
        }
    if not sources_by_class:
        raise FileNotFoundError(f"No type crop sources found in {input_dir} or {reference_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    print(f"Input folder: {input_dir}")
    print(f"Reference fallback folder: {reference_dir}")
    print(f"Output folder: {output_dir}")
    print(f"Classes with sources: {len(sources_by_class)}")
    print(f"Target per class: {target_per_class}")
    print()

    for index, (class_name, sources) in enumerate(sources_by_class.items(), start=1):
        class_dir = output_dir / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{index}/{len(sources_by_class)}] {class_name}: {len(sources)} source images")

        for image_index in range(target_per_class):
            source_path = random.choice(sources)
            try:
                image = augment_type_crop(source_path)
                output_path = class_dir / f"{class_name}_synthetic_{image_index:04d}.jpg"
                image.save(output_path, format="JPEG", quality=90)
                total_written += 1
            except Exception as exc:
                print(f"  ERROR {source_path}: {exc}")

    print()
    print("Done.")
    print(f"Classes written: {len(sources_by_class)}")
    print(f"Synthetic images written: {total_written}")
    print(f"Output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--references", default=str(DEFAULT_REFERENCE_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--target-per-class", type=int, default=120)
    parser.add_argument("--singles-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(
        input_dir=Path(args.input),
        reference_dir=Path(args.references),
        output_dir=Path(args.output),
        target_per_class=args.target_per_class,
        singles_only=args.singles_only,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

