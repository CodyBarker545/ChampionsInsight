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
WORK_CARD_WIDTH = 176
WORK_CARD_HEIGHT = 144
MIN_VISIBLE_SPRITE_PIXELS = 260
MIN_VISIBLE_SPRITE_RATIO = 0.045
MIN_VISIBLE_SPRITE_BBOX_SIZE = 14
MAX_CROP_ATTEMPTS = 24

RED_BACKGROUNDS = [
    (135, 0, 55),
    (155, 0, 70),
    (120, 0, 50),
    (170, 5, 75),
    (145, 10, 65),
]

REAL_LIKE_CROP_PROFILES = [
    # Common tight portrait crops from real detector output.
    {"width": (44, 72), "height": (72, 104), "weight": 0.28},
    {"width": (60, 96), "height": (76, 112), "weight": 0.28},
    {"width": (82, 132), "height": (82, 124), "weight": 0.24},
    # Wider/looser crops, still plausible after red-card object detection.
    {"width": (112, 172), "height": (82, 150), "weight": 0.14},
    # Occasional messy crop that includes too much card/context.
    {"width": (180, 320), "height": (120, 210), "weight": 0.06},
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


def choose_real_like_crop_size() -> tuple[int, int]:
    profile = random.choices(
        REAL_LIKE_CROP_PROFILES,
        weights=[item["weight"] for item in REAL_LIKE_CROP_PROFILES],
        k=1,
    )[0]
    return (
        random.randint(*profile["width"]),
        random.randint(*profile["height"]),
    )


def crop_region_with_padding(
    img: Image.Image,
    mask: Image.Image,
    box: tuple[int, int, int, int],
) -> tuple[Image.Image, Image.Image]:
    left, top, right, bottom = box
    target_width = right - left
    target_height = bottom - top
    card_width, card_height = img.size

    padded = make_red_background(
        card_width + target_width * 2,
        card_height + target_height * 2,
    )
    padded_mask = Image.new("L", padded.size, 0)

    padded.paste(img, (target_width, target_height))
    padded_mask.paste(mask, (target_width, target_height))

    shifted_box = (
        left + target_width,
        top + target_height,
        right + target_width,
        bottom + target_height,
    )

    return padded.crop(shifted_box), padded_mask.crop(shifted_box)


def has_enough_visible_sprite(mask: Image.Image) -> bool:
    arr = np.array(mask)
    visible_pixels = int(np.count_nonzero(arr > 24))
    if visible_pixels < MIN_VISIBLE_SPRITE_PIXELS:
        return False

    visible_ratio = visible_pixels / max(1, arr.shape[0] * arr.shape[1])
    if visible_ratio < MIN_VISIBLE_SPRITE_RATIO:
        return False

    bbox = mask.getbbox()
    if not bbox:
        return False

    left, top, right, bottom = bbox
    return (
        right - left >= MIN_VISIBLE_SPRITE_BBOX_SIZE
        and bottom - top >= MIN_VISIBLE_SPRITE_BBOX_SIZE
    )


def mask_center_crop_box(
    mask: Image.Image,
    target_width: int,
    target_height: int,
) -> tuple[int, int, int, int]:
    bbox = mask.getbbox()
    if not bbox:
        card_width, card_height = mask.size
        center_x = card_width // 2
        center_y = card_height // 2
    else:
        left, top, right, bottom = bbox
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2

    left = center_x - target_width // 2 + random.randint(
        -max(2, target_width // 8),
        max(2, target_width // 8),
    )
    top = center_y - target_height // 2 + random.randint(
        -max(2, target_height // 8),
        max(2, target_height // 8),
    )

    return (left, top, left + target_width, top + target_height)


def crop_real_like_from_card(
    img: Image.Image,
    mask: Image.Image,
) -> tuple[Image.Image, Image.Image]:
    """
    Make synthetic crops resemble detector output from real screenshots.

    Real crops are often tight, portrait-ish, partially clipped, and not all
    resized to one canonical dimension. This intentionally keeps variable
    output sizes so the classifier sees the same rough distribution.
    """

    card_width, card_height = img.size
    target_width, target_height = choose_real_like_crop_size()

    for _ in range(MAX_CROP_ATTEMPTS):
        # Anchor mostly on the actual sprite, with controlled drift to create
        # cutoffs without producing blank red crops.
        if random.random() < 0.82:
            left, top, right, bottom = mask_center_crop_box(
                mask,
                target_width,
                target_height,
            )
        else:
            center_x = int(card_width * random.uniform(0.30, 0.52))
            center_y = int(card_height * random.uniform(0.42, 0.58))
            center_x += random.randint(
                -int(target_width * 0.28),
                int(target_width * 0.24),
            )
            center_y += random.randint(
                -int(target_height * 0.22),
                int(target_height * 0.22),
            )
            left = center_x - target_width // 2
            top = center_y - target_height // 2
            right = left + target_width
            bottom = top + target_height

        crop, crop_mask = crop_region_with_padding(img, mask, (left, top, right, bottom))
        if has_enough_visible_sprite(crop_mask):
            break
    else:
        box = mask_center_crop_box(mask, target_width, target_height)
        crop, crop_mask = crop_region_with_padding(img, mask, box)

    # Some real crops are small detector boxes that were saved directly; others
    # are slightly scaled by processing. Keep both.
    if random.random() < 0.28:
        scale = random.uniform(0.75, 1.35)
        size = (
            max(24, int(round(crop.width * scale))),
            max(32, int(round(crop.height * scale))),
        )
        crop = crop.resize(size, Image.Resampling.BICUBIC)
        crop_mask = crop_mask.resize(
            size,
            Image.Resampling.BICUBIC,
        )

    return crop, crop_mask


def crop_fixed_legacy_from_card(
    img: Image.Image,
    mask: Image.Image,
) -> tuple[Image.Image, Image.Image]:
    """Keep a slice of the old fixed-size behavior for backwards coverage."""

    card_width, card_height = img.size

    # Simulate imperfect crop boundaries from screenshots/phone captures.
    if random.random() < 0.55:
        left = random.randint(0, int(card_width * 0.12))
        top = random.randint(0, int(card_height * 0.10))
        right = card_width - random.randint(0, int(card_width * 0.12))
        bottom = card_height - random.randint(0, int(card_height * 0.10))

        if right > left + 20 and bottom > top + 20:
            img = img.crop((left, top, right, bottom))
            mask = mask.crop((left, top, right, bottom))

    size = (CARD_WIDTH, CARD_HEIGHT)
    return (
        img.resize(size, Image.Resampling.BICUBIC),
        mask.resize(size, Image.Resampling.BICUBIC),
    )


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
        image_width, image_height = img.size
        glare_x = random.randint(0, image_width - 1)
        glare_y = random.randint(0, image_height - 1)
        radius = random.randint(
            max(8, int(min(image_width, image_height) * 0.18)),
            max(16, int(max(image_width, image_height) * 0.48)),
        )

        yy, xx = np.ogrid[:image_height, :image_width]
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

    bg = make_red_background(WORK_CARD_WIDTH, WORK_CARD_HEIGHT).convert("RGBA")

    # Wide scale range.
    # This creates small, medium, large, and partially oversized examples.
    max_sprite_w = int(WORK_CARD_WIDTH * random.uniform(0.40, 1.65))
    max_sprite_h = int(WORK_CARD_HEIGHT * random.uniform(0.42, 1.65))

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
    min_x = -int(sprite.width * 0.58)
    max_x = WORK_CARD_WIDTH - int(sprite.width * 0.48)

    min_y = -int(sprite.height * 0.55)
    max_y = WORK_CARD_HEIGHT - int(sprite.height * 0.45)

    # Guard against invalid ranges for very large sprites.
    if min_x > max_x:
        min_x, max_x = max_x, min_x

    if min_y > max_y:
        min_y, max_y = max_y, min_y

    x = random.randint(min_x, max_x)
    y = random.randint(min_y, max_y)

    bg.alpha_composite(sprite, (x, y))
    sprite_mask = Image.new("L", bg.size, 0)
    sprite_mask.paste(sprite.getchannel("A"), (x, y), sprite.getchannel("A"))

    img = bg.convert("RGB")

    if random.random() < 0.72:
        img, _ = crop_real_like_from_card(img, sprite_mask)
    else:
        img, _ = crop_fixed_legacy_from_card(img, sprite_mask)

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
