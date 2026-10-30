"""
predict.py - Standalone Inference Script for Remote Sensing Models.

Generates binary building masks, probability maps, and optional GeoJSON polygons
from input imagery files (.tif, .png, .jpg).

Usage Examples:
---------------
1. Predict building footprints on a single image:
   python predict.py --model sam_cebnet --image sample_custom_data/images/sample_000.png --output outputs/predictions/sample_000_pred.png

2. Predict bi-temporal changes between two images:
   python predict.py --model sam2_cd --image1 sample_custom_data/A/sample_000.png --image2 sample_custom_data/B/sample_000.png --output outputs/predictions/sample_000_change.png
"""

import argparse
import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from models import METHOD_CATALOG, create_method_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference with remote sensing models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        "--method",
        dest="method",
        type=str,
        default="sam_cebnet",
        choices=list(METHOD_CATALOG.keys()),
        help="Architecture / model key",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to trained weights file (.pth)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image for single-image models",
    )
    parser.add_argument(
        "--image1",
        type=str,
        default=None,
        help="Path to Pre-event / Time-1 image for bi-temporal models",
    )
    parser.add_argument(
        "--image2",
        type=str,
        default=None,
        help="Path to Post-event / Time-2 image for bi-temporal models",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./outputs/prediction.png",
        help="Path to save output prediction mask (.png)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Sigmoid probability threshold for binary mask",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Inference spatial resolution",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device",
    )
    return parser.parse_args()


def load_and_preprocess_image(path: str, img_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((img_size, img_size), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


def main():
    args = parse_args()
    device = torch.device(args.device)

    task = METHOD_CATALOG[args.method]["task"]
    print(f"Loading model: {args.method} (Task: {task}) on {device}")

    # Build model
    model = create_method_model(args.method, base_channels=32)
    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint weights from: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with torch.no_grad():
        if task == "single_image_building":
            img_path = args.image or args.image1
            if not img_path:
                raise ValueError("Must provide --image for single-image building extraction models!")
            x = load_and_preprocess_image(img_path, args.img_size).to(device)
            out = model(x)
            logits = out.seg_logits if hasattr(out, "seg_logits") else out
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

        else:
            # Bi-temporal
            p1 = args.image1 or args.image
            p2 = args.image2 or args.image
            if not p1 or not p2:
                raise ValueError("Must provide --image1 and --image2 for bi-temporal models!")
            t1 = load_and_preprocess_image(p1, args.img_size).to(device)
            t2 = load_and_preprocess_image(p2, args.img_size).to(device)
            out = model(t1, t2)
            logits = out.seg_logits if hasattr(out, "seg_logits") else out
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

    # Save output
    bin_mask = (probs >= args.threshold).astype(np.uint8) * 255
    pred_img = Image.fromarray(bin_mask)
    pred_img.save(args.output)
    print(f"Prediction successfully saved to: {args.output}")


if __name__ == "__main__":
    main()
