#!/usr/bin/env python3
"""
Export a YOLOv11 PyTorch .pt model to TensorRT .engine for Jetson.

Usage (from YOLOv11-finetuning directory):
  python3 export_to_engine.py \
    --weights runs/segment/yolov11n_seg_custom/weights/best.pt \
    --half \
    --workspace 2

Notes:
  - Use --half for FP16 (faster, slightly more memory). Omit for FP32 (safer).
  - workspace is in GB; keep small (2–4) on Jetson to avoid OOM.
"""

import argparse
import os
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Export YOLO .pt to TensorRT .engine")
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/segment/yolov11n_seg_custom/weights/best.pt",
        help="Path to the input .pt model",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Enable FP16 (faster, slightly higher memory). Default FP32.",
    )
    parser.add_argument(
        "--workspace",
        type=int,
        default=2,
        help="TensorRT workspace size in GB (keep low on Jetson to avoid OOM).",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        help="Simplify the ONNX graph before engine build (can speed up).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default 640). Match this in your inference code.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"Could not find weights file: {args.weights}")

    print(f"\n=== TensorRT Export ===")
    print(f"Input model: {args.weights}")
    print(f"Precision : {'FP16' if args.half else 'FP32'}")
    print(f"Workspace : {args.workspace} GB")
    print(f"Simplify  : {args.simplify}")
    print(f"Image Size: {args.imgsz}")
    print(f"=======================\n")

    model = YOLO(args.weights)

    engine_path = model.export(
        format="engine",
        half=args.half,
        workspace=args.workspace,
        simplify=args.simplify,
        imgsz=args.imgsz,
        verbose=True,
        device="cuda:0",
    )

    print(f"\n✅ Export complete!")
    print(f"Engine saved to: {engine_path}")
    print("Update your runtime to load the .engine for best Jetson performance.")


if __name__ == "__main__":
    main()

