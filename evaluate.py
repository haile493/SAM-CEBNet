"""
evaluate.py - Standalone Evaluation Script for Remote Sensing Models.

Computes comprehensive benchmark metrics:
  - Intersection over Union (IoU / mIoU)
  - F1-Score / Dice Coefficient
  - Precision & Recall
  - Overall Accuracy (OA)
  - Boundary F1-Score (Boundary IoU)
  - Latency (ms per tile) & Frames Per Second (FPS)
  - Parameter count (M)

Usage Examples:
---------------
1. Evaluate trained SAM-CEBNet checkpoint:
   python evaluate.py --model sam_cebnet --checkpoint outputs/sam_cebnet/best_model.pth --data_dir ./dataset

2. Evaluate on sample custom data:
   python evaluate.py --model cbrnet --data_dir ./sample_custom_data --img_size 128

3. Zero-shot inference & evaluation with BiSAM-CD:
   python evaluate.py --model bisam_cd --data_dir ./dataset --output_dir ./outputs/bisam_cd
"""

import argparse
import json
import os
import sys

from models import METHOD_CATALOG
from run_pipeline import run_method


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Remote Sensing Building Extraction and Change Detection Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "--method",
        dest="method",
        type=str,
        default="sam_cebnet",
        choices=list(METHOD_CATALOG.keys()),
        help="Architecture / model key to evaluate",
    )
    default_data = "./dataset" if os.path.exists("./dataset") else "./sample_custom_data"
    parser.add_argument(
        "--data_dir",
        type=str,
        default=default_data,
        help="Path to dataset folder",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained model weights (.pth). If None, runs model with default weights",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/evaluation",
        help="Directory to save evaluation report and sample predictions",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="unseen_test",
        choices=["train", "val", "unseen_test", "test", "all"],
        help="Split to evaluate on",
    )
    parser.add_argument(
        "--split_strategy",
        type=str,
        default="spatial_block",
        choices=["spatial_block", "stratified", "random"],
        help="Dataset split strategy",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Evaluation image resolution",
    )
    parser.add_argument(
        "--base_channels",
        type=int,
        default=32,
        help="Base feature channel width",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Inference batch size",
    )
    parser.add_argument(
        "--task_mode",
        type=str,
        default="bcd",
        choices=["bcd", "bda"],
        help="Task mode for GMFIMamba",
    )
    parser.add_argument(
        "--single_image",
        action="store_true",
        help="Force single-image evaluation for bi-temporal models",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max number of samples to evaluate on (for rapid testing)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 75)
    print(f"Starting Evaluation: {args.method.upper()}")
    print(f"Dataset Path       : {args.data_dir} (Split: {args.split})")
    print(f"Checkpoint         : {args.checkpoint or 'Default / Scratch'}")
    print(f"Resolution         : {args.img_size}x{args.img_size}")
    print(f"Output Directory   : {args.output_dir}")
    print("=" * 75)

    os.makedirs(args.output_dir, exist_ok=True)

    metrics = run_method(
        method=args.method,
        mode="eval",
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        img_size=args.img_size,
        base_channels=args.base_channels,
        batch_size=args.batch_size,
        task_mode=args.task_mode,
        single_image_mode=args.single_image,
        split=args.split,
        split_strategy=args.split_strategy,
        max_samples=args.max_samples,
    )

    # Save detailed metrics JSON
    metrics_path = os.path.join(args.output_dir, f"{args.method}_eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 75)
    print(f"Evaluation Summary for {args.method.upper()}:")
    print(f"  - Mean IoU (mIoU)    : {metrics.get('mean_iou', 0)*100:.2f}%")
    print(f"  - Dice / F1-Score    : {metrics.get('mean_dice', 0)*100:.2f}%")
    print(f"  - Precision          : {metrics.get('precision', 0)*100:.2f}%")
    print(f"  - Recall             : {metrics.get('recall', 0)*100:.2f}%")
    print(f"  - Overall Accuracy   : {metrics.get('overall_accuracy', 0)*100:.2f}%")
    if metrics.get("boundary_f1") is not None:
        print(f"  - Boundary F1 (B-F1) : {metrics.get('boundary_f1', 0)*100:.2f}%")
    print(f"  - Model Parameters   : {metrics.get('params_m', 0):.2f}M")
    print(f"  - Latency per Tile   : {metrics.get('latency_ms_per_tile', 0):.2f} ms")
    print(f"  - Inference FPS      : {metrics.get('fps', 0):.1f}")
    print("=" * 75)
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
