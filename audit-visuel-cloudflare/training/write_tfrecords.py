#!/usr/bin/env python3
"""Create TFOD-compatible TFRecord files from prepare_coco.py output."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import tensorflow as tf


def bytes_feature(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[v.encode("utf-8") if isinstance(v, str) else v for v in values]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def int64_list_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(v) for v in values]))


def float_list_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[float(v) for v in values]))


def write_split(root: Path, split: str, class_names: dict[int, str]) -> int:
    annotation_file = root / "annotations" / f"instances_{split}.json"
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    images_dir = root / "images" / split
    by_image = {}
    for annotation in data["annotations"]:
        by_image.setdefault(annotation["image_id"], []).append(annotation)
    output = root / f"{split}.record"
    count = 0
    with tf.io.TFRecordWriter(str(output)) as writer:
        for image in data["images"]:
            path = images_dir / image["file_name"]
            if not path.exists():
                raise FileNotFoundError(path)
            encoded = path.read_bytes()
            width, height = int(image["width"]), int(image["height"])
            xmins, xmaxs, ymins, ymaxs, classes, class_text, areas = [], [], [], [], [], [], []
            for annotation in by_image.get(image["id"], []):
                x, y, box_w, box_h = annotation["bbox"]
                xmins.append(max(0.0, x / width))
                xmaxs.append(min(1.0, (x + box_w) / width))
                ymins.append(max(0.0, y / height))
                ymaxs.append(min(1.0, (y + box_h) / height))
                classes.append(int(annotation["category_id"]))
                class_text.append(class_names[int(annotation["category_id"])])
                areas.append(float(annotation.get("area", box_w * box_h)))
            extension = path.suffix.lower().lstrip(".")
            if extension in ("jpg", "jpe"):
                extension = "jpeg"
            features = {
                "image/height": int64_feature(height),
                "image/width": int64_feature(width),
                "image/filename": bytes_feature(image["file_name"]),
                "image/source_id": bytes_feature(str(image["id"])),
                "image/encoded": bytes_feature(encoded),
                "image/format": bytes_feature(extension),
                "image/object/bbox/xmin": float_list_feature(xmins),
                "image/object/bbox/xmax": float_list_feature(xmaxs),
                "image/object/bbox/ymin": float_list_feature(ymins),
                "image/object/bbox/ymax": float_list_feature(ymaxs),
                "image/object/class/text": bytes_list_feature(class_text),
                "image/object/class/label": int64_list_feature(classes),
                "image/object/area": float_list_feature(areas),
                "image/object/is_crowd": int64_list_feature([0] * len(classes)),
                "image/object/difficult": int64_list_feature([0] * len(classes)),
                "image/object/truncated": int64_list_feature([0] * len(classes)),
            }
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())
            count += 1
    print(f"{output}: {count} image(s)")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    train_json = json.loads((args.dataset_dir / "annotations" / "instances_train.json").read_text(encoding="utf-8"))
    class_names = {int(category["id"]): str(category["name"]) for category in train_json["categories"]}
    write_split(args.dataset_dir, "train", class_names)
    validation_json = args.dataset_dir / "annotations" / "instances_val.json"
    if validation_json.exists() and json.loads(validation_json.read_text(encoding="utf-8")).get("images"):
        write_split(args.dataset_dir, "val", class_names)
    else:
        print("Pas de jeu de validation ; ajoutez plusieurs images et relancez prepare_coco.py avant l'entraînement.")

if __name__ == "__main__":
    main()
