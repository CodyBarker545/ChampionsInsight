"""
Generate synthetic Pokémon slot-card training images from clean Champions menu sprites.

This version:
- Reads both normal and shiny folders.
- Merges shiny with normal as the same class.
- Ignores Mega forms.
- Creates red/pink slot-card style images.
- Uses no meaningful tilt.
- Focuses on scale, shifting, partial cutoffs, blur, noise, glare, and JPEG artifacts.

Example:
python scripts/generate_synthetic_pokemon_cards.py ^
  --input "data/pokemon/champions_sprites" ^
  --output "data/training_dataset/slot_pokemon_synthetic" ^
  --images-per-sprite 150
"""

import argparse
import random
import re
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


CARD_WIDTH = 128
CARD_HEIGHT = 96

RED_BACKGROUNDS = [
    (135, 0, 55),
    (155, 0, 70),
    (120, 0, 50),
    (170, 5, 75),
    (145, 10, 65),
]


def parse_label(path: Path) -> str | None:
    """
    Returns the output class folder label.

    Rules:
    - Normal and shiny are merged into the same class.
    - Mega forms are skipped.
    """

    stem = path.stem

    if "mega" in stem.lower():
        return None

    stem = stem.replace("_shiny", "")
    stem = stem.replace("-shiny", "")
    stem = stem.replace(" shiny", "")

    stem = re.sub(r"[^A-Za-z0-9_\-]+", "_", stem)

    return stem


def make_red_background(width: int, height: int) -> Image.Image:
    """Create a red/pink slot-card background with mild texture."""

    base_color = random.choice(RED_BACKGROUNDS)

    bg = Image.new("RGB", (width, height), base_color)
    arr = np.array(bg).astype(np.int16)

    x_gradient = np.linspace(-12, 12, width).astype(np.int16)
    y_gradient = np.linspace(-8, 8, height).astype(np.int16)

    arr[:, :, 0] += x_gradient[None, :]
    arr[:, :, 1] += y_gradient[:, None]
    arr[:, :, 2] += x_gradient[None, :] // 2

    noise = np.random.normal(0, random.uniform(2.0, 8.0), arr.shape).astype(np.int16)
    arr += noise

    arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, "RGB")


def add_camera_effects(img: Image.Image) -> Image.Image:
    """Apply camera-like effects to make synthetic images closer to phone/screenshot crops."""

    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.75, 1.25))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.80, 1.35))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.85, 1.25))

    if random.random() < 0.45:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.20, 1.10)))

    if random.random() < 0.25:
        img = img.filter(ImageFilter.SHARPEN)

    arr = np.array(img).astype(np.int16)

    if random.random() < 0.90:
        noise = np.random.normal(0, random.uniform(2.0, 12.0), arr.shape).astype(np.int16)
        arr += noise

    if random.random() < 0.25:
        glare_x = random.randint(0, CARD_WIDTH - 1)
        glare_y = random.randint(0, CARD_HEIGHT - 1)
        radius = random.randint(18, 55)

        yy, xx = np.ogrid[:CARD_HEIGHT, :CARD_WIDTH]
        dist = np.sqrt((xx - glare_x) ** 2 + (yy - glare_y) ** 2)
        mask = np.clip(1 - dist / radius, 0, 1)

        strength = random.uniform(20, 65)

        arr[:, :, 0] += (mask * strength).astype(np.int16)
        arr[:, :, 1] += (mask * strength).astype(np.int16)
        arr[:, :, 2] += (mask * strength).astype(np.int16)

    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")

    if random.random() < 0.65:
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=random.randint(45, 92))
        buffer.seek(0)
        img = Image.open(buffer).convert("RGB")

    return img


def paste_sprite_on_card(sprite_path: Path) -> Image.Image:
    """
    Place one clean sprite onto a synthetic red card.

    Main augmentation strategy:
    - Different scales
    - Different positions
    - Random partial cutoff from all directions
    - No meaningful tilt
    """

    sprite = Image.open(sprite_path).convert("RGBA")

    bbox = sprite.getbbox()
    if bbox:
        sprite = sprite.crop(bbox)

    bg = make_red_background(CARD_WIDTH, CARD_HEIGHT).convert("RGBA")

    # Wide scale range.
    # This creates small, medium, large, and partially oversized examples.
    max_sprite_w = int(CARD_WIDTH * random.uniform(0.50, 1.40))
    max_sprite_h = int(CARD_HEIGHT * random.uniform(0.50, 1.40))

    scale = min(max_sprite_w / sprite.width, max_sprite_h / sprite.height)
    scale *= random.uniform(0.80, 1.35)

    new_w = max(8, int(sprite.width * scale))
    new_h = max(8, int(sprite.height * scale))

    sprite = sprite.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # No tilt. The real team slots should not be meaningfully rotated.
    # Keeping this at 0 avoids teaching the model unrealistic rotation.
    angle = 0
    if angle != 0:
        sprite = sprite.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)

    # Place sprite in a way that allows natural cutoff from any side.
    # The sprite can be partly outside the card boundaries.
    min_x = -int(sprite.width * 0.45)
    max_x = CARD_WIDTH - int(sprite.width * 0.55)

    min_y = -int(sprite.height * 0.45)
    max_y = CARD_HEIGHT - int(sprite.height * 0.55)

    # Guard against invalid ranges for very large sprites.
    if min_x > max_x:
        min_x, max_x = max_x, min_x

    if min_y > max_y:
        min_y, max_y = max_y, min_y

    x = random.randint(min_x, max_x)
    y = random.randint(min_y, max_y)

    bg.alpha_composite(sprite, (x, y))

    img = bg.convert("RGB")

    # Simulate imperfect crop boundaries from screenshots/phone captures.
    if random.random() < 0.45:
        left = random.randint(0, 10)
        top = random.randint(0, 8)
        right = CARD_WIDTH - random.randint(0, 10)
        bottom = CARD_HEIGHT - random.randint(0, 8)

        if right > left + 20 and bottom > top + 20:
            img = img.crop((left, top, right, bottom))
            img = img.resize((CARD_WIDTH, CARD_HEIGHT), Image.Resampling.BICUBIC)

    img = add_camera_effects(img)

    return img


def collect_sprite_paths(input_dir: Path) -> list[Path]:
    """
    Collect sprites from:
    - input_dir/normal
    - input_dir/shiny

    If those folders do not exist, fallback to recursive PNG search under input_dir.
    """

    normal_dir = input_dir / "normal"
    shiny_dir = input_dir / "shiny"

    sprite_paths: list[Path] = []

    if normal_dir.exists():
        sprite_paths.extend(sorted(normal_dir.rglob("*.png")))

    if shiny_dir.exists():
        sprite_paths.extend(sorted(shiny_dir.rglob("*.png")))

    if not sprite_paths:
        sprite_paths = sorted(input_dir.rglob("*.png"))

    return sorted(sprite_paths)


def generate_dataset(
    input_dir: Path,
    output_dir: Path,
    images_per_sprite: int,
    seed: int,
) -> None:
    """Generate the full synthetic dataset."""

    random.seed(seed)
    np.random.seed(seed)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    sprite_paths = collect_sprite_paths(input_dir)

    if not sprite_paths:
        raise FileNotFoundError(f"No PNG sprites found under: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    total_skipped_mega = 0
    total_used_sprites = 0

    print(f"Input folder: {input_dir}")
    print(f"Output folder: {output_dir}")
    print(f"Found {len(sprite_paths)} PNG sprite files.")
    print(f"Images per sprite: {images_per_sprite}")
    print()

    for index, sprite_path in enumerate(sprite_paths, start=1):
        label = parse_label(sprite_path)

        if label is None:
            total_skipped_mega += 1
            print(f"[{index}/{len(sprite_paths)}] SKIP mega: {sprite_path.name}")
            continue

        total_used_sprites += 1

        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)

        print(f"[{index}/{len(sprite_paths)}] {sprite_path.name} -> {label}")

        for i in range(images_per_sprite):
            try:
                img = paste_sprite_on_card(sprite_path)

                out_name = f"{sprite_path.stem}_synthetic_{i:04d}.jpg"
                out_path = label_dir / out_name

                img.save(out_path, format="JPEG", quality=90)
                total_written += 1

            except Exception as exc:
                print(f"  ERROR on {sprite_path}: {exc}")

    print()
    print("Done.")
    print(f"Sprites found: {len(sprite_paths)}")
    print(f"Sprites used: {total_used_sprites}")
    print(f"Mega sprites skipped: {total_skipped_mega}")
    print(f"Synthetic images written: {total_written}")
    print(f"Output: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Folder containing champions_sprites with normal/ and shiny/ folders.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output training dataset folder.",
    )

    parser.add_argument(
        "--images-per-sprite",
        type=int,
        default=150,
        help="Number of synthetic images to create per source sprite.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Random seed for repeatable generation.",
    )

    args = parser.parse_args()

    generate_dataset(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        images_per_sprite=args.images_per_sprite,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()