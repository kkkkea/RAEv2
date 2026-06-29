#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INPUT_ROOT="${1:-${IMAGENET100_TRAIN_DIR:-data/imagenet100/train}}"
OUTPUT="${2:-${IMAGENET100_CLASSES_FILE:-data/imagenet100_classes.txt}}"

cd "$REPO_ROOT"

python scripts/data/make_imagenet_class_folder_reference_npz.py \
  --input-root "$INPUT_ROOT" \
  --classes-output "$OUTPUT"

