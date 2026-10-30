"""
train.py - Standalone Training Script for Remote Sensing Building Footprint Extraction & Change Detection.

Supports all benchmarked models (2022–2026):
  - SAM-CEBNet (Proposed SAM-2 Hiera Foundation + Corner BiFPN + Edge Refinement)
  - SAM3-CEBNet (Proposed SAM-3 Foundation + Concept Tokens + Edge Refinement)
  - CBRNet (Chen et al., IEEE TGRS 2026)
  - CGCCE-Net (Wang et al., Information Sciences 2026)
  - EdgeRefNet (Hima et al., IEEE TGRS 2026)
  - GMFIMamba (Xu et al., EAAI 2026)
  - SemiBCD (Liu et al., IEEE TGRS 2026)
  - KGBDCNet (Wang et al., ISPRS JPRS 2026)
  - SAM2-CD (Qin et al., IEEE JSTARS 2025)
  - BiSAM-CD (Qin et al., IEEE TGRS 2025)
  - ImprovedMaskRCNN (Han et al., Complex & Intell. Syst. 2022)

Usage Examples:
---------------
1. Train SAM-CEBNet on Hue City dataset (or custom dataset):
   python train.py --model sam_cebnet --data_dir ./dataset --epochs 50 --batch_size 4 --img_size 256

2. Quick train on sample custom data:
   python train.py --model cbrnet --data_dir ./sample_custom_data --epochs 5 --batch_size 2 --img_size 128

3. Train with pretrained backbone:
   python train.py --model sam_cebnet --data_dir ./dataset --pretrained --epochs 50
"""

import argparse
import os
import sys

from models import METHOD_CATALOG
from run_pipeline import run_method


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Remote Sensing Building Extraction and Change Detection Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "--method",
        dest="method",
        type=str,
        default="sam_cebnet",
        choices=list(METHOD_CATALOG.keys()),
        help="Architecture / model key to train",
    )
    default_data = "./dataset" if os.path.exists("./dataset") else "./sample_custom_data"
    parser.add_argument(
        "--data_dir",
        type=str,
        default=default_data,
        help="Path to Hue GeoTIFF/GeoJSON dataset or flat folder dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Output directory to save checkpoints and training logs",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional initial checkpoint to resume or fine-tune from",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Mini-batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate for AdamW optimizer",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Input spatial resolution (H=W)",
    )
    parser.add_argument(
        "--base_channels",
        type=int,
        default=32,
        help="Base feature channel width",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load ImageNet/Foundation pretrained backbone weights",
    )
    parser.add_argument(
        "--pretrained_arch",
        type=str,
        default="resnet50",
        choices=["resnet34", "resnet50"],
        help="Backbone architecture when --pretrained is enabled",
    )
    parser.add_argument(
        "--whu_pretrain",
        action="store_true",
        help="Run 2-stage transfer: Stage-1 WHU pretrain, Stage-2 Hue fine-tune",
    )
    parser.add_argument(
        "--whu_epochs",
        type=int,
        default=12,
        help="Stage-1 WHU pretraining epochs",
    )
    parser.add_argument(
        "--task_mode",
        type=str,
        default="bcd",
        choices=["bcd", "bda"],
        help="Task mode for GMFIMamba: bcd (binary change) or bda (building damage)",
    )
    parser.add_argument(
        "--split_strategy",
        type=str,
        default="spatial_block",
        choices=["spatial_block", "stratified", "random"],
        help="Dataset splitting strategy to guarantee zero spatial leakage",
    )
    parser.add_argument(
        "--single_image",
        action="store_true",
        help="Force single-image evaluation for bi-temporal architectures",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional limit on number of samples per split (useful for debugging)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 75)
    print(f"Starting Training: {args.method.upper()}")
    print(f"Dataset Path   : {args.data_dir}")
    print(f"Epochs         : {args.epochs} | Batch Size: {args.batch_size} | LR: {args.lr}")
    print(f"Resolution     : {args.img_size}x{args.img_size}")
    print(f"Output Path    : {args.output_dir}")
    print("=" * 75)

    os.makedirs(args.output_dir, exist_ok=True)

    metrics = run_method(
        method=args.method,
        mode="train",
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        img_size=args.img_size,
        base_channels=args.base_channels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        task_mode=args.task_mode,
        pretrained=args.pretrained,
        pretrained_arch=args.pretrained_arch,
        single_image_mode=args.single_image,
        whu_pretrain=args.whu_pretrain,
        whu_epochs=args.whu_epochs,
        split_strategy=args.split_strategy,
        max_samples=args.max_samples,
    )

    print("\nTraining completed successfully!")
    print(f"Best Validation IoU: {metrics.get('mean_iou', 0)*100:.2f}%")
    print(f"Best Validation F1 : {metrics.get('mean_dice', 0)*100:.2f}%")


if __name__ == "__main__":
    main()
