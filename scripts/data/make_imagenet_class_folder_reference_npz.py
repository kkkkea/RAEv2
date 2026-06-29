"""Build metadata or a reference NPZ from an ImageNet-style class-folder set.

Expected input layout:

    <input-root>/
      n01440764/
        ILSVRC2012_val_00000293.JPEG
        ...
      n01443537/
        ...

For validation references, the NPZ output follows this repository's eval contract:

    arr_0: uint8 RGB images in NHWC layout, range [0, 255]

No model normalization is applied. The only default image operation is the same
resize used by the repository's ImageNet dataloader.

For training-set bookkeeping, use --classes-output to write only the collected
class ids without loading images or creating an NPZ.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a RAEv2 reference NPZ from ImageNet nxxxxxxxx folders."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="Directory containing ImageNet class folders such as n01440764/.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz path. The file will contain arr_0. Not required with --classes-output.",
    )
    parser.add_argument(
        "--classes-output",
        type=Path,
        default=None,
        help="Optional text file to write collected WordNet ids, one per line. If --output is omitted, only this file is written.",
    )
    parser.add_argument(
        "--classes-file",
        type=Path,
        default=None,
        help="Optional text file with one WordNet id per line. Blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Output square image size. Use 0 to disable resizing.",
    )
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=None,
        help="Optional cap on images per class, useful for quick smoke tests.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional JSON manifest path recording classes and source files.",
    )
    return parser.parse_args()


def load_class_ids(input_root: Path, classes_file: Path | None) -> list[str]:
    if classes_file is None:
        class_ids = sorted(p.name for p in input_root.iterdir() if p.is_dir() and p.name.startswith("n"))
    else:
        class_ids = []
        for line in classes_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                class_ids.append(line)

    if not class_ids:
        raise ValueError("No class folders found or provided.")

    missing = [class_id for class_id in class_ids if not (input_root / class_id).is_dir()]
    if missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(f"{len(missing)} class folders are missing under {input_root}: {preview}")

    return class_ids


def list_images(class_dir: Path, limit: int | None) -> list[Path]:
    image_paths = sorted(
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if limit is not None:
        image_paths = image_paths[:limit]
    return image_paths


def load_image(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        if image_size > 0 and img.size != (image_size, image_size):
            img = img.resize((image_size, image_size), Image.Resampling.BICUBIC)
        return np.asarray(img, dtype=np.uint8)


def main() -> None:
    args = parse_args()
    input_root = args.input_root.expanduser().resolve()

    class_ids = load_class_ids(input_root, args.classes_file)

    if args.classes_output is not None:
        classes_output = args.classes_output.expanduser().resolve()
        classes_output.parent.mkdir(parents=True, exist_ok=True)
        classes_output.write_text("\n".join(class_ids) + "\n")
        print(f"Saved class ids to {classes_output}")
        print(f"  classes: {len(class_ids)}")

    if args.output is None:
        if args.classes_output is None:
            raise ValueError("Provide --output to create an NPZ or --classes-output to write class ids.")
        return

    output = args.output.expanduser().resolve()

    indexed_paths: list[tuple[str, Path]] = []
    per_class_counts: dict[str, int] = {}
    for class_id in class_ids:
        paths = list_images(input_root / class_id, args.limit_per_class)
        if not paths:
            raise FileNotFoundError(f"No images found in {input_root / class_id}")
        per_class_counts[class_id] = len(paths)
        indexed_paths.extend((class_id, path) for path in paths)

    images = [
        load_image(path, args.image_size)
        for _, path in tqdm(indexed_paths, desc="Loading images")
    ]
    arr = np.stack(images, axis=0)

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, arr_0=arr)

    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = output.with_suffix(".manifest.json")
    manifest_path = manifest_path.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "input_root": str(input_root),
        "output": str(output),
        "image_size": args.image_size,
        "num_classes": len(class_ids),
        "num_images": int(arr.shape[0]),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "classes": class_ids,
        "per_class_counts": per_class_counts,
        "files": [
            {"class_id": class_id, "path": str(path.relative_to(input_root))}
            for class_id, path in indexed_paths
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Saved {output}")
    print(f"  arr_0 shape: {arr.shape}")
    print(f"  arr_0 dtype: {arr.dtype}")
    print(f"  classes: {len(class_ids)}")
    print(f"  manifest: {manifest_path}")


if __name__ == "__main__":
    main()
