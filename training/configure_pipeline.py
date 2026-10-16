#!/usr/bin/env python3
"""Configure a TensorFlow Object Detection API Faster R-CNN pipeline config."""
from __future__ import annotations
import argparse
from pathlib import Path
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True, help="pipeline.config shipped with a TF2 Faster R-CNN Model Zoo checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint prefix, without .index")
    parser.add_argument("--train-record", type=Path, required=True)
    parser.add_argument("--val-record", type=Path, required=True)
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    pipeline = pipeline_pb2.TrainEvalPipelineConfig()
    text_format.Merge(args.base_config.read_text(encoding="utf-8"), pipeline)
    if not pipeline.model.HasField("faster_rcnn"):
        raise SystemExit("Le pipeline fourni n'est pas un modèle Faster R-CNN. Choisissez un pipeline.config Faster R-CNN du TF2 Detection Model Zoo.")
    pipeline.model.faster_rcnn.num_classes = args.num_classes
    pipeline.train_config.batch_size = args.batch_size
    pipeline.train_config.fine_tune_checkpoint = args.checkpoint
    pipeline.train_config.fine_tune_checkpoint_type = "detection"
    pipeline.train_input_reader.label_map_path = str(args.label_map.resolve())
    del pipeline.train_input_reader.tf_record_input_reader.input_path[:]
    pipeline.train_input_reader.tf_record_input_reader.input_path.append(str(args.train_record.resolve()))
    if len(pipeline.eval_input_reader) == 0:
        pipeline.eval_input_reader.add()
    eval_reader = pipeline.eval_input_reader[0]
    eval_reader.label_map_path = str(args.label_map.resolve())
    del eval_reader.tf_record_input_reader.input_path[:]
    eval_reader.tf_record_input_reader.input_path.append(str(args.val_record.resolve()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text_format.MessageToString(pipeline), encoding="utf-8")
    print(f"Pipeline écrit : {args.output}")

if __name__ == "__main__":
    main()
