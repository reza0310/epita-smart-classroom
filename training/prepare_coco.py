#!/usr/bin/env python3
"""Unpack an Atlas labelled-dataset-v1 JSON export into a COCO-style dataset.

Usage:
  python prepare_coco.py atlas-jeu-apprentissage.json ./dataset

The input JSON intentionally embeds image data URLs so the export is portable.
"""
from __future__ import annotations
import argparse
import base64
import json
import mimetypes
import random
import re
from pathlib import Path
from urllib.parse import unquote_to_bytes


def parse_data_url(data_url: str) -> tuple[bytes, str]:
    match = re.match(r"^data:([^;,]+)?(;base64)?,(.*)$", data_url, re.DOTALL)
    if not match:
        raise ValueError("data_url d'image invalide")
    mime = match.group(1) or "image/jpeg"
    payload = match.group(3)
    raw = base64.b64decode(payload) if match.group(2) else unquote_to_bytes(payload)
    ext = mimetypes.guess_extension(mime) or ".jpg"
    if ext == ".jpe":
        ext = ".jpg"
    return raw, ext


def safe_name(name: str, fallback: str) -> str:
    name = Path(name or fallback).name
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(name).stem).strip("._") or fallback
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--validation-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    payload = json.loads(args.export_json.read_text(encoding="utf-8"))
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise SystemExit("Aucun exemple dans le JSON. Exportez d'abord un jeu Atlas annoté.")

    categories = payload.get("categories") or [
        {"id": i + 1, "name": c[0], "label": c[1]}
        for i, c in enumerate([
            ("chair", "Chaises"), ("table", "Tables et bureaux"),
            ("laptop", "Ordinateurs portables"), ("screen", "Écrans"),
            ("board", "Tableaux"), ("window", "Fenêtres"), ("anomaly", "Anomalies visuelles"),
        ])
    ]
    category_ids = {str(c["name"]): int(c["id"]) for c in categories}
    category_ids.update({str(c.get("label")): int(c["id"]) for c in categories})

    out = args.output_dir.resolve()
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "annotations").mkdir(parents=True, exist_ok=True)

    prepared = []
    seen_names = set()
    errors = []
    for index, sample in enumerate(samples, start=1):
        if not sample.get("dataUrl") or not sample.get("width") or not sample.get("height"):
            errors.append(f"Exemple {index}: image ou dimensions manquantes")
            continue
        try:
            image_bytes, extension = parse_data_url(sample["dataUrl"])
        except Exception as exc:
            errors.append(f"Exemple {index}: {exc}")
            continue
        stem = safe_name(sample.get("name", ""), f"image_{index:05d}")
        file_name = f"{index:05d}_{stem}{extension}"
        while file_name.lower() in seen_names:
            file_name = f"{index:05d}_{stem}_{len(seen_names)}{extension}"
        seen_names.add(file_name.lower())
        anns = []
        for annotation in sample.get("annotations", []):
            category = category_ids.get(str(annotation.get("categoryId")))
            bbox = annotation.get("bbox")
            if category is None or not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x, y, width, height = [float(v) for v in bbox]
            if width <= 0 or height <= 0:
                continue
            anns.append({"category_id": category, "bbox": [x, y, width, height], "area": width * height, "iscrowd": 0})
        prepared.append({"source_index": index, "file_name": file_name, "bytes": image_bytes,
                         "width": int(sample["width"]), "height": int(sample["height"]), "annotations": anns})

    if not prepared:
        raise SystemExit("Aucune image valide n'a pu être extraite.")
    random.Random(args.seed).shuffle(prepared)
    val_count = round(len(prepared) * args.validation_ratio) if len(prepared) >= 2 else 0
    if len(prepared) >= 2:
        val_count = max(1, val_count)
    val_count = min(val_count, max(0, len(prepared) - 1))
    val_ids = {item["source_index"] for item in prepared[:val_count]}

    annotations_by_split = {
        "train": {"info": {"description": "Atlas visual audit labelled data"}, "images": [], "annotations": [], "categories": categories},
        "val": {"info": {"description": "Atlas visual audit validation data"}, "images": [], "annotations": [], "categories": categories},
    }
    next_annotation_id = 1
    for image_id, item in enumerate(prepared, start=1):
        split = "val" if item["source_index"] in val_ids else "train"
        target = out / "images" / split / item["file_name"]
        target.write_bytes(item["bytes"])
        coco = annotations_by_split[split]
        split_image_id = len(coco["images"]) + 1
        coco["images"].append({"id": split_image_id, "file_name": item["file_name"], "width": item["width"], "height": item["height"]})
        for ann in item["annotations"]:
            coco["annotations"].append({"id": next_annotation_id, "image_id": split_image_id, **ann})
            next_annotation_id += 1

    for split, document in annotations_by_split.items():
        path = out / "annotations" / f"instances_{split}.json"
        path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")

    label_map = []
    for category in categories:
        label_map.append(f'item {{\n  id: {int(category["id"])}\n  name: "{category["name"]}"\n}}')
    (out / "label_map.pbtxt").write_text("\n\n".join(label_map) + "\n", encoding="utf-8")
    manifest = {
        "images_total": len(prepared), "train_images": len(annotations_by_split["train"]["images"]),
        "validation_images": len(annotations_by_split["val"]["images"]),
        "train_boxes": len(annotations_by_split["train"]["annotations"]),
        "validation_boxes": len(annotations_by_split["val"]["annotations"]),
        "categories": categories, "warnings": errors,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if len(prepared) < 20:
        print("\nAVERTISSEMENT : moins de 20 images. Ce volume sert à démarrer, mais ne suffit généralement pas pour valider un détecteur robuste.")
    if not annotations_by_split["val"]["images"]:
        print("\nAVERTISSEMENT : aucune image de validation (un seul exemple exploitable). Ajoutez des images avant d'évaluer le modèle.")

if __name__ == "__main__":
    main()
