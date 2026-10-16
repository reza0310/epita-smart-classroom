#!/usr/bin/env sh
set -eu
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 atlas-jeu-apprentissage.json [output-dir]" >&2
  exit 2
fi
INPUT="$1"
OUTPUT="${2:-./training/dataset}"
python training/prepare_coco.py "$INPUT" "$OUTPUT"
python training/write_tfrecords.py "$OUTPUT"
echo "Dataset préparé dans: $OUTPUT"
