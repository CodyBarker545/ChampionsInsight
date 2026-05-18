"""Build a YOLO dataset for opponent red slot detection.

This Phase 2 script uses the existing debug-crop workflow:
upload through the frontend or test script, collect debug crops, cluster/review
unknown crops, then leave full red slot crops in review/unsorted_slots.

It creates synthetic full team-column images from those slot crops, places them
onto varied backgrounds, and writes YOLO labels for one class: opponent_slot.

Run from backend:
    python scripts/build_yolo_slot_dataset.py

Example:
    python scripts/build_yolo_slot_dataset.py --count 1200 --val-ratio 0.15
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


BACKEND_DIR = Path(__file__).resolve().parents[2]
TRAINING_ROOT = BACKEND_DIR / "data" / "training_dataset"
REVIEW_DIR = TRAINING_ROOT / "review"

DEFAULT_INPUT_DIRS = [
    REVIEW_DIR / "unsorted_slots",
    TRAINING_ROOT / "yolo_slot_detector_cv_labels" / "crops" / "train",
    TRAINING_ROOT / "yolo_slot_detector_cv_labels" / "crops" / "val",
]
DEFAULT_OUTPUT_DIR = TRAINING_ROOT / "yolo_slot_detector"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_ID = 0
CLASS_NAME = "opponent_slot"


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    width: int
    height: int


def list_images(input_dirs: list[Path]) -> list[Path]:
    image_paths: list[Path] = []

    for input_dir in input_dirs:
        if not input_dir.exists():
            continue

        image_paths.extend(
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    return sorted(set(image_paths))


def read_valid_slot(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if image is None:
        print(f"[WARN] Skipping unreadable image: {path}")
        return None

    height, width = image.shape[:2]
    if width < 48 or height < 24:
        print(f"[WARN] Skipping tiny slot crop: {path} ({width}x{height})")
        return None

    return image


def load_slot_images(input_dirs: list[Path]) -> list[tuple[Path, np.ndarray]]:
    slots: list[tuple[Path, np.ndarray]] = []

    for path in list_images(input_dirs):
        image = read_valid_slot(path)
        if image is not None:
            slots.append((path, image))

    if not slots:
        searched = "\n".join(f"  {path}" for path in input_dirs)
        raise FileNotFoundError(f"No valid slot images found in:\n{searched}")

    return slots


def reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    for split in ["train", "val"]:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "overlays" / split).mkdir(parents=True, exist_ok=True)


def make_background(width: int, height: int, rng: random.Random) -> np.ndarray:
    base_color = np.array(
        [
            rng.randint(18, 70),
            rng.randint(18, 75),
            rng.randint(18, 80),
        ],
        dtype=np.float32,
    )
    y_gradient = np.linspace(0, rng.randint(12, 42), height, dtype=np.float32)[:, None]
    x_gradient = np.linspace(0, rng.randint(-16, 16), width, dtype=np.float32)[None, :]

    background = np.zeros((height, width, 3), dtype=np.float32)
    for channel in range(3):
        background[:, :, channel] = base_color[channel] + y_gradient + x_gradient

    noise = rng.normalvariate(0, 1)
    noise_scale = 5 + abs(noise) * 8
    background += np.random.default_rng(rng.randrange(1_000_000_000)).normal(
        0,
        noise_scale,
        background.shape,
    )

    return np.clip(background, 0, 255).astype(np.uint8)


def add_camera_artifacts(image: np.ndarray, rng: random.Random) -> np.ndarray:
    result = image.copy()

    # Simulate a broad family of phone captures rather than only clean screenshots.
    if rng.random() < 0.58:
        alpha = rng.uniform(0.68, 1.28)
        beta = rng.uniform(-34, 30)
        result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)

    if rng.random() < 0.32:
        tint = np.zeros_like(result, dtype=np.int16)
        tint[:, :, 0] += rng.randint(-18, 18)
        tint[:, :, 1] += rng.randint(-14, 14)
        tint[:, :, 2] += rng.randint(-18, 18)
        result = np.clip(result.astype(np.int16) + tint, 0, 255).astype(np.uint8)

    if rng.random() < 0.42:
        kernel = rng.choice([3, 5, 7])
        result = cv2.GaussianBlur(result, (kernel, kernel), 0)

    if rng.random() < 0.18:
        kernel_size = rng.choice([3, 5, 7])
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        if rng.random() < 0.5:
            kernel[kernel_size // 2, :] = 1.0 / kernel_size
        else:
            kernel[:, kernel_size // 2] = 1.0 / kernel_size
        result = cv2.filter2D(result, -1, kernel)

    if rng.random() < 0.30:
        height, width = result.shape[:2]
        overlay = result.copy()
        center = (rng.randint(0, width - 1), rng.randint(0, height - 1))
        radius = rng.randint(max(40, width // 10), max(60, width // 3))
        cv2.circle(overlay, center, radius, (255, 255, 255), -1)
        result = cv2.addWeighted(overlay, rng.uniform(0.04, 0.20), result, 1.0, 0)

    if rng.random() < 0.28:
        height, width = result.shape[:2]
        scanline_gap = rng.choice([3, 4, 5, 6])
        scanline_alpha = rng.uniform(0.05, 0.15)
        overlay = result.copy()
        overlay[::scanline_gap, :, :] = (overlay[::scanline_gap, :, :] * (1 - scanline_alpha)).astype(np.uint8)
        result = overlay

    if rng.random() < 0.36:
        noise_sigma = rng.uniform(4.0, 16.0)
        noise = np.random.default_rng(rng.randrange(1_000_000_000)).normal(
            0,
            noise_sigma,
            result.shape,
        )
        result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.26:
        height, width = result.shape[:2]
        scale = rng.uniform(0.45, 0.90)
        down = cv2.resize(
            result,
            (max(8, int(width * scale)), max(8, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        result = cv2.resize(down, (width, height), interpolation=cv2.INTER_LINEAR)

    if rng.random() < 0.40:
        quality = rng.randint(35, 92)
        ok, encoded = cv2.imencode(".jpg", result, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    return result


def rotate_slot(slot: np.ndarray, degrees: float) -> np.ndarray:
    if abs(degrees) < 0.1:
        return slot

    height, width = slot.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), degrees, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))
    matrix[0, 2] += (new_width / 2) - (width / 2)
    matrix[1, 2] += (new_height / 2) - (height / 2)

    return cv2.warpAffine(
        slot,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def augment_slot(slot: np.ndarray, target_width: int, rng: random.Random) -> np.ndarray:
    height, width = slot.shape[:2]
    scale = target_width / max(1, width)
    target_height = max(1, int(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(slot, (target_width, target_height), interpolation=interpolation)

    if rng.random() < 0.55:
        resized = rotate_slot(resized, rng.uniform(-3.5, 3.5))

    if rng.random() < 0.50:
        resized = cv2.convertScaleAbs(
            resized,
            alpha=rng.uniform(0.86, 1.14),
            beta=rng.uniform(-12, 14),
        )

    if rng.random() < 0.20:
        resized = cv2.GaussianBlur(resized, (3, 3), 0)

    return resized


def paste_image(canvas: np.ndarray, patch: np.ndarray, x: int, y: int) -> Box:
    canvas_height, canvas_width = canvas.shape[:2]
    patch_height, patch_width = patch.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(canvas_width, x + patch_width)
    y2 = min(canvas_height, y + patch_height)

    if x2 <= x1 or y2 <= y1:
        return Box(0, 0, 0, 0)

    patch_x1 = x1 - x
    patch_y1 = y1 - y
    patch_x2 = patch_x1 + (x2 - x1)
    patch_y2 = patch_y1 + (y2 - y1)

    canvas[y1:y2, x1:x2] = patch[patch_y1:patch_y2, patch_x1:patch_x2]

    return Box(x1, y1, x2 - x1, y2 - y1)


def paste_shadow(canvas: np.ndarray, patch: np.ndarray, x: int, y: int, alpha: float = 0.18) -> None:
    canvas_height, canvas_width = canvas.shape[:2]
    patch_height, patch_width = patch.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(canvas_width, x + patch_width)
    y2 = min(canvas_height, y + patch_height)

    if x2 <= x1 or y2 <= y1:
        return

    region = canvas[y1:y2, x1:x2]
    darkened = np.zeros_like(region)
    canvas[y1:y2, x1:x2] = cv2.addWeighted(region, 1 - alpha, darkened, alpha, 0)


def make_yolo_label(box: Box, image_width: int, image_height: int) -> str:
    x_center = (box.x + box.width / 2) / image_width
    y_center = (box.y + box.height / 2) / image_height
    width = box.width / image_width
    height = box.height / image_height

    return f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def compose_training_image(
    slots: list[tuple[Path, np.ndarray]],
    canvas_width: int,
    canvas_height: int,
    min_slots: int,
    max_slots: int,
    rng: random.Random,
) -> tuple[np.ndarray, list[Box], list[str]]:
    canvas = make_background(canvas_width, canvas_height, rng)
    slot_count = rng.randint(min_slots, max_slots)
    chosen_slots = [rng.choice(slots) for _ in range(slot_count)]

    target_width = int(canvas_width * rng.uniform(0.54, 0.82))
    sample_height, sample_width = chosen_slots[0][1].shape[:2]
    base_slot_height = max(1, int(sample_height * (target_width / max(1, sample_width))))
    gap = int(base_slot_height * rng.uniform(0.18, 0.34))
    total_height = slot_count * base_slot_height + (slot_count - 1) * gap

    if total_height > canvas_height * 0.94:
        shrink = (canvas_height * 0.94) / total_height
        target_width = max(64, int(target_width * shrink))
        base_slot_height = max(1, int(base_slot_height * shrink))
        gap = max(4, int(gap * shrink))
        total_height = slot_count * base_slot_height + (slot_count - 1) * gap

    x_base = rng.randint(
        max(0, int(canvas_width * 0.10)),
        max(1, canvas_width - target_width - int(canvas_width * 0.04)),
    )
    y_base = rng.randint(
        max(0, int(canvas_height * 0.03)),
        max(1, canvas_height - total_height - int(canvas_height * 0.02)),
    )

    boxes: list[Box] = []
    source_names: list[str] = []

    for index, (path, slot) in enumerate(chosen_slots):
        width_jitter = int(target_width * rng.uniform(0.94, 1.05))
        patch = augment_slot(slot, width_jitter, rng)
        x = x_base + rng.randint(-int(canvas_width * 0.025), int(canvas_width * 0.025))
        y = y_base + index * (base_slot_height + gap) + rng.randint(
            -int(base_slot_height * 0.08),
            int(base_slot_height * 0.08),
        )

        paste_shadow(
            canvas,
            patch,
            x + rng.randint(3, 10),
            y + rng.randint(3, 10),
        )

        box = paste_image(canvas, patch, x, y)
        if box.width > 4 and box.height > 4:
            boxes.append(box)
            source_names.append(path.name)

    return add_camera_artifacts(canvas, rng), boxes, source_names


def draw_overlay(image: np.ndarray, boxes: list[Box]) -> np.ndarray:
    overlay = image.copy()
    for index, box in enumerate(boxes, start=1):
        x1, y1 = box.x, box.y
        x2, y2 = box.x + box.width, box.y + box.height
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 4)
        cv2.putText(
            overlay,
            str(index),
            (x1, max(32, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )
    return overlay


def write_dataset_yaml(output_dir: Path) -> None:
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_dir.as_posix()}",
                "train: images/train",
                "val: images/val",
                "names:",
                f"  0: {CLASS_NAME}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_manifest(output_dir: Path, manifest: list[dict], args: argparse.Namespace) -> None:
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "className": CLASS_NAME,
                "count": len(manifest),
                "settings": {
                    "count": args.count,
                    "valRatio": args.val_ratio,
                    "canvasWidth": args.canvas_width,
                    "canvasHeight": args.canvas_height,
                    "minSlots": args.min_slots,
                    "maxSlots": args.max_slots,
                    "seed": args.seed,
                },
                "images": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def build_dataset(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    input_dirs = [
        path if path.is_absolute() else BACKEND_DIR / path
        for path in args.input
    ]
    output_dir = args.output if args.output.is_absolute() else BACKEND_DIR / args.output

    slots = load_slot_images(input_dirs)
    reset_output_dir(output_dir)

    manifest: list[dict] = []
    val_count = int(round(args.count * args.val_ratio))
    if args.val_ratio > 0 and args.count > 1:
        val_count = max(1, min(args.count - 1, val_count))
    val_indices = set(rng.sample(range(1, args.count + 1), val_count))

    for index in range(1, args.count + 1):
        split = "val" if index in val_indices else "train"
        image, boxes, source_names = compose_training_image(
            slots=slots,
            canvas_width=args.canvas_width,
            canvas_height=args.canvas_height,
            min_slots=args.min_slots,
            max_slots=args.max_slots,
            rng=rng,
        )

        stem = f"slot_detector_{index:06d}"
        image_path = output_dir / "images" / split / f"{stem}.jpg"
        label_path = output_dir / "labels" / split / f"{stem}.txt"
        overlay_path = output_dir / "overlays" / split / f"{stem}_overlay.jpg"

        labels = [
            make_yolo_label(box, args.canvas_width, args.canvas_height)
            for box in boxes
        ]

        cv2.imwrite(str(image_path), image)
        label_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
        cv2.imwrite(str(overlay_path), draw_overlay(image, boxes))

        manifest.append(
            {
                "split": split,
                "image": str(image_path),
                "label": str(label_path),
                "overlay": str(overlay_path),
                "boxCount": len(boxes),
                "boxes": [
                    {
                        "x": box.x,
                        "y": box.y,
                        "width": box.width,
                        "height": box.height,
                    }
                    for box in boxes
                ],
                "sources": source_names,
            }
        )

        if index % 100 == 0 or index == args.count:
            print(f"Generated {index}/{args.count}")

    write_dataset_yaml(output_dir)
    write_manifest(output_dir, manifest, args)

    print()
    print(f"Loaded slot crops: {len(slots)}")
    print(f"YOLO dataset: {output_dir}")
    print(f"Data YAML: {output_dir / 'data.yaml'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build YOLO data for red slot detection.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=DEFAULT_INPUT_DIRS,
        help="Folder(s) containing unsorted full red slot crops.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="YOLO dataset output folder.",
    )
    parser.add_argument("--count", type=int, default=800, help="Synthetic images to generate.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--canvas-width", type=int, default=1215, help="Synthetic image width.")
    parser.add_argument("--canvas-height", type=int, default=2160, help="Synthetic image height.")
    parser.add_argument("--min-slots", type=int, default=6, help="Minimum slots per image.")
    parser.add_argument("--max-slots", type=int, default=6, help="Maximum slots per image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if not 0 <= args.val_ratio < 1:
        raise ValueError("--val-ratio must be in [0, 1)")
    if args.min_slots < 1 or args.max_slots < args.min_slots:
        raise ValueError("--min-slots and --max-slots are invalid")

    build_dataset(args)


if __name__ == "__main__":
    main()

