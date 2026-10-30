"""
Fair Comparison Suite for all 2026 Remote Sensing Models on the Hue Dataset.

Evaluates all 7 models from 2026:
1. CBRNet (Chen et al., IEEE TGRS 2026)
2. CGCCE-Net (Wang et al., Information Sciences 2026)
3. EdgeRefNet (Hima et al., IEEE TGRS 2026)
4. GMFIMamba (Xu et al., EAAI 2026)
5. SemiBCD (Liu et al., IEEE TGRS 2026)
6. KGBDCNet (Wang et al., ISPRS JPRS 2026)
7. SAM-CEBNet (Proposed 2026 Architecture)

Across the 3 requested settings:
- Table 1: All methods trained From Scratch and directly on Hue dataset
- Table 2: All methods with Pretrained Backbone and trained directly on Hue dataset
- Table 3: All methods with Pretrained Backbone and applying 2-Stage Training (WHU Pretrain -> Hue Fine-Tune)
"""

import argparse
import json
import os
import time
from typing import Any, Dict, List

import torch

from models import METHOD_CATALOG
from run_pipeline import run_method

MODELS_2026 = [
    {
        "key": "cbrnet",
        "name": "CBRNet",
        "authors_venue": "Chen et al., IEEE TGRS 2026",
        "task": "Single-Image Building",
        "arch_desc": "Corner-Guided Boundary Refinement + BiFPN + DSBR",
    },
    {
        "key": "cgcce_net",
        "name": "CGCCE-Net",
        "authors_venue": "Wang et al., Information Sciences 2026",
        "task": "Change Detection / Footprint",
        "arch_desc": "Global Cross-Correlation + Semantic Cognitive Enhancement",
    },
    {
        "key": "edgerefnet",
        "name": "EdgeRefNet",
        "authors_venue": "Hima et al., IEEE TGRS 2026",
        "task": "Change Detection / Footprint",
        "arch_desc": "Sobel Edge Guidance + EDB/EEM Blocks + Cross-Attention",
    },
    {
        "key": "gmfimamba",
        "name": "GMFIMamba",
        "authors_venue": "Xu et al., EAAI 2026",
        "task": "Change Detection / Footprint",
        "arch_desc": "Grouped Mamba Feature Interaction (GMBFIM) + MCFEM",
    },
    {
        "key": "semibcd",
        "name": "SemiBCD",
        "authors_venue": "Liu et al., IEEE TGRS 2026",
        "task": "Semi-Supervised CD / Footprint",
        "arch_desc": "VLM Pseudo-Supervision + Task-Homogeneous Alignment",
    },
    {
        "key": "kgbdcnet",
        "name": "KGBDCNet",
        "authors_venue": "Wang et al., ISPRS JPRS 2026",
        "task": "Damage Captioning / Footprint",
        "arch_desc": "Keyword-Guided STMEM + Space-Scan Mamba + LoRA",
    },
    {
        "key": "sam_cebnet",
        "name": "SAM2-CEBNet",
        "authors_venue": "Proposed Architecture (SAM 2)",
        "task": "Single-Image Building (Foundation)",
        "arch_desc": "SAM-2 Hiera Foundation + Corner BiFPN + Edge Cross-Attention",
    },
    {
        "key": "sam3_cebnet",
        "name": "SAM3-CEBNet",
        "authors_venue": "Proposed Architecture (SAM 3)",
        "task": "Single-Image Building (Foundation)",
        "arch_desc": "SAM 3 Perception Encoder + PCS Concept Tokens + Presence Gating",
    },
]


def format_table(
    title: str,
    setting_desc: str,
    results: List[Dict[str, Any]],
) -> str:
    lines = []
    lines.append(f"### {title}")
    lines.append(f"*{setting_desc}*\n")
    lines.append(
        "| # | Model / Architecture | Published Venue / Authors | Backbone / Pretrain | Params (M) | Latency (ms) | FPS | Test IoU (%) | Test Dice (%) | Precision (%) | Recall (%) | Overall Acc (%) | Boundary F1 (%) |"
    )
    lines.append(
        "| :-: | :--- | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |"
    )

    # Sort results by Test IoU descending
    sorted_res = sorted(results, key=lambda x: x.get("mean_iou", 0.0), reverse=True)
    for idx, item in enumerate(sorted_res, 1):
        m_name = item.get("name", item.get("method"))
        venue = item.get("authors_venue", "2026")
        bb = item.get("backbone_desc", "-")
        params = f"{item.get('params_m', 0.0):.2f}"
        lat = f"{item.get('latency_ms_per_tile', 0.0):.2f}"
        fps = f"{item.get('fps', 0.0):.1f}"
        iou = f"{item.get('mean_iou', 0.0) * 100:.2f}%"
        dice = f"{item.get('mean_dice', 0.0) * 100:.2f}%"
        prec = f"{item.get('precision', 0.0) * 100:.2f}%"
        rec = f"{item.get('recall', 0.0) * 100:.2f}%"
        oa = f"{item.get('overall_accuracy', 0.0) * 100:.2f}%"
        bnd = f"{item.get('boundary_f1', 0.0) * 100:.2f}%" if item.get("boundary_f1") is not None else "-"
        lines.append(
            f"| **{idx}** | **{m_name}** | {venue} | {bb} | {params} | {lat} | {fps} | **{iou}** | **{dice}** | {prec} | {rec} | {oa} | {bnd} |"
        )

    # Captioning extra metrics footnote if KGBDCNet present
    kgb = next((r for r in results if r.get("method") == "kgbdcnet"), None)
    if kgb and "caption_token_f1" in kgb:
        lines.append("\n> [!NOTE]")
        lines.append(
            f"> **KGBDCNet Bi-Temporal Damage Captioning Metrics**: "
            f"Token F1 = **{kgb.get('caption_token_f1', 0.0)*100:.2f}%**, "
            f"BLEU-4 = **{kgb.get('caption_bleu4', 0.0)*100:.2f}%**, "
            f"ROUGE-L = **{kgb.get('caption_rouge_l', 0.0)*100:.2f}%**, "
            f"Exact Match = **{kgb.get('caption_exact_match', 0.0)*100:.2f}%**."
        )
    lines.append("\n")
    return "\n".join(lines)


def run_all_fair_comparisons(
    data_dir: str = "./dataset",
    output_root: str = "./outputs/fair_comparison_2026",
    epochs: int = 20,
    whu_epochs: int = 10,
    img_size: int = 128,
    batch_size: int = 2,
    lr: float = 1e-3,
    reuse_existing: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    os.makedirs(output_root, exist_ok=True)
    all_experiments: Dict[str, List[Dict[str, Any]]] = {
        "scratch_direct_hue": [],
        "pretrained_direct_hue": [],
        "pretrained_whu_hue_2stage": [],
    }

    settings_meta = [
        (
            "scratch_direct_hue",
            "Table 1: All 2026 Methods Trained From Scratch on Direct Hue Dataset",
            "Full training directly on Hue dataset (109 train tiles, zero pretraining, randomly initialized weights).",
            False,  # pretrained
            False,  # whu_pretrain
        ),
        (
            "pretrained_direct_hue",
            "Table 2: All 2026 Methods with Pretrained Backbone on Direct Hue Dataset",
            "Standard ImageNet-pretrained pyramid backbone (ResNet-34/50 / SAM-2 Hiera) fine-tuned directly on Hue dataset.",
            True,  # pretrained
            False,  # whu_pretrain
        ),
        (
            "pretrained_whu_hue_2stage",
            "Table 3: All 2026 Methods with Pretrained Backbone & 2-Stage Training (WHU Datasets -> Hue Dataset)",
            "Stage 1: Multi-source pretraining on WHU Aerial + Wuhan East Asian Satellite + Hue train quadrants; Stage 2: Target fine-tuning on Hue dataset.",
            True,  # pretrained
            True,  # whu_pretrain
        ),
    ]

    for setting_key, setting_title, setting_desc, is_pretrained, is_whu in settings_meta:
        print("=" * 90, flush=True)
        print(f"Executing {setting_title}...", flush=True)
        print("=" * 90, flush=True)
        setting_out_dir = os.path.join(output_root, setting_key)
        os.makedirs(setting_out_dir, exist_ok=True)

        for m_info in MODELS_2026:
            m_key = m_info["key"]
            m_name = m_info["name"]
            m_out = os.path.join(setting_out_dir, m_key)
            metrics_file = os.path.join(m_out, f"{m_key}_metrics.json")
            best_ckpt = os.path.join(m_out, f"{m_key}_best.pt")

            # Check existing result in current directory
            if reuse_existing and os.path.isfile(metrics_file):
                print(f"[{setting_key}] Reusing existing cached metrics for {m_name}...", flush=True)
                with open(metrics_file, "r", encoding="utf-8") as f:
                    res = json.load(f)["summary"]
                res.update(m_info)
                res["setting"] = setting_key
                if is_pretrained:
                    res["backbone_desc"] = (
                        "ResNet-50 + SAM-3 PCS" if m_key == "sam3_cebnet"
                        else "ResNet-50 + SAM-2 Hiera" if m_key == "sam_cebnet"
                        else "ResNet-50 Pretrained" if m_key == "cbrnet"
                        else "ResNet-34 Pretrained"
                    )
                else:
                    res["backbone_desc"] = "From Scratch (Random Init)"
                all_experiments[setting_key].append(res)
                continue

            # Check if an equivalent run exists in single_image_9methods
            existing_match = None
            if reuse_existing and setting_key == "scratch_direct_hue":
                alt_p = os.path.join("outputs", "single_image_9methods", m_key, f"{m_key}_metrics.json")
                if os.path.isfile(alt_p):
                    with open(alt_p, "r", encoding="utf-8") as f:
                        cached = json.load(f)["summary"]
                    if not cached.get("pretrained_backbone", False) and not cached.get("whu_pretrain", False):
                        existing_match = cached
            elif reuse_existing and setting_key == "pretrained_direct_hue":
                if m_key == "cbrnet":
                    alt_p = os.path.join("outputs", "single_image_9methods", "cbrnet_resnet50", "cbrnet_metrics.json")
                    if os.path.isfile(alt_p):
                        with open(alt_p, "r", encoding="utf-8") as f:
                            existing_match = json.load(f)["summary"]
                elif m_key == "sam_cebnet":
                    alt_p = os.path.join("outputs", "single_image_9methods", "sam_cebnet_resnet50", "sam_cebnet_metrics.json")
                    if os.path.isfile(alt_p):
                        with open(alt_p, "r", encoding="utf-8") as f:
                            existing_match = json.load(f)["summary"]
            elif reuse_existing and setting_key == "pretrained_whu_hue_2stage":
                if m_key == "sam_cebnet":
                    alt_p = os.path.join("outputs", "single_image_9methods", "sam_cebnet_resnet50_whu_pretrain", "sam_cebnet_metrics.json")
                    if os.path.isfile(alt_p):
                        with open(alt_p, "r", encoding="utf-8") as f:
                            existing_match = json.load(f)["summary"]

            if existing_match is not None:
                print(f"[{setting_key}] Imported existing certified run for {m_name}...", flush=True)
                existing_match.update(m_info)
                existing_match["setting"] = setting_key
                if is_pretrained:
                    existing_match["backbone_desc"] = (
                        "ResNet-50 + SAM-2 Hiera" if m_key == "sam_cebnet"
                        else "ResNet-50 Pretrained" if m_key == "cbrnet"
                        else "ResNet-34 Pretrained"
                    )
                else:
                    existing_match["backbone_desc"] = "From Scratch (Random Init)"
                all_experiments[setting_key].append(existing_match)
                continue

            # Execute Training & Evaluation
            print(
                f"[{setting_key} | {m_name}] Training: pretrained={is_pretrained}, whu_pretrain={is_whu}, "
                f"epochs={epochs}, whu_epochs={whu_epochs if is_whu else 0}...",
                flush=True,
            )
            bb_arch = "resnet50" if m_key in ("cbrnet", "sam_cebnet", "sam3_cebnet") else "resnet34"
            stats = run_method(
                method=m_key,
                mode="train",
                data_dir=data_dir,
                output_dir=m_out,
                img_size=img_size,
                base_channels=16,
                batch_size=batch_size,
                epochs=epochs,
                lr=lr,
                split="unseen_test",
                split_strategy="spatial_block",
                pretrained=is_pretrained,
                pretrained_arch=bb_arch,
                single_image_mode=True,
                whu_pretrain=is_whu,
                whu_epochs=whu_epochs if is_whu else 0,
            )
            stats.update(m_info)
            stats["setting"] = setting_key
            if is_pretrained:
                stats["backbone_desc"] = (
                    "ResNet-50 + SAM-3 PCS" if m_key == "sam3_cebnet"
                    else "ResNet-50 + SAM-2 Hiera" if m_key == "sam_cebnet"
                    else f"{bb_arch.capitalize()} Pretrained"
                )
            else:
                stats["backbone_desc"] = "From Scratch (Random Init)"
            all_experiments[setting_key].append(stats)

    # Save comprehensive results JSON
    summary_path = os.path.join(output_root, "fair_comparison_2026_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_experiments, f, indent=2)

    return all_experiments


def format_sam_ablation(results: Dict[str, List[Dict[str, Any]]]) -> str:
    lines = []
    lines.append("### Head-to-Head Ablation: SAM-2 vs. SAM-3 on the Proposed SAM-CEBNet Architecture")
    lines.append("*Direct side-by-side evaluation across all 3 training regimes on the locked 21 unseen Hue test tiles:*\n")
    lines.append("| Training Regime | Model Variant | Foundation Backbone | Key Architectural Mechanism | Params (M) | Latency (ms) | FPS | Test IoU (%) | Test Dice (%) | Precision (%) | Recall (%) | Boundary F1 (%) |")
    lines.append("| :--- | :--- | :--- | :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |")
    
    regime_names = [
        ("scratch_direct_hue", "From Scratch (Direct Hue)"),
        ("pretrained_direct_hue", "Pretrained (Direct Hue)"),
        ("pretrained_whu_hue_2stage", "2-Stage (WHU -> Hue)"),
    ]
    for reg_key, reg_label in regime_names:
        reg_items = {item["key"]: item for item in results.get(reg_key, [])}
        for k, v_name, v_mech in [
            ("sam_cebnet", "SAM2-CEBNet", "Hiera Hierarchical Tokens + Geometric Priors"),
            ("sam3_cebnet", "SAM3-CEBNet", "PCS Concept Tokens + Decoupled Presence Gating"),
        ]:
            if k in reg_items:
                m = reg_items[k]
                p_m = m.get("model_params", 0) / 1e6
                lat = m.get("inference_time_ms_per_tile", 0.0)
                fps = m.get("fps", 0.0)
                iou = m.get("mean_iou", 0.0) * 100.0
                dice = m.get("mean_dice", 0.0) * 100.0
                prec = m.get("mean_precision", 0.0) * 100.0
                rec = m.get("mean_recall", 0.0) * 100.0
                b_f1 = m.get("mean_boundary_f1", 0.0) * 100.0
                bb = m.get("backbone_desc", "Pretrained")
                lines.append(
                    f"| {reg_label} | **{v_name}** | {bb} | {v_mech} | {p_m:.2f} | {lat:.2f} | {fps:.1f} | **{iou:.2f}%** | **{dice:.2f}%** | {prec:.2f}% | {rec:.2f}% | **{b_f1:.2f}%** |"
                )
    lines.append("\n")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fair Comparison Suite for 2026 Remote Sensing Models")
    parser.add_argument("--data_dir", type=str, default="./dataset")
    parser.add_argument("--output_root", type=str, default="./outputs/fair_comparison_2026")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--whu_epochs", type=int, default=10)
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--no_reuse", action="store_true")
    args = parser.parse_args()

    results = run_all_fair_comparisons(
        data_dir=args.data_dir,
        output_root=args.output_root,
        epochs=args.epochs,
        whu_epochs=args.whu_epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        reuse_existing=not args.no_reuse,
    )
    print("\n" + "=" * 90)
    print("FAIR COMPARISON 2026 - COMPLETE BENCHMARK REPORT")
    print("=" * 90 + "\n")
    t1 = format_table(
        "Table 1: 2026 Models - From Scratch on Direct Hue Dataset",
        "Direct Hue dataset training (109 train tiles, zero pretraining, randomly initialized weights, evaluated on 21 held-out unseen test tiles).",
        results["scratch_direct_hue"],
    )
    t2 = format_table(
        "Table 2: 2026 Models - With Pretrained Backbone on Direct Hue Dataset",
        "Standard ImageNet-pretrained backbone (ResNet-34/50 / SAM-2 Hiera / SAM-3 PCS) fine-tuned directly on Hue dataset.",
        results["pretrained_direct_hue"],
    )
    t3 = format_table(
        "Table 3: 2026 Models - With Pretrained Backbone & 2-Stage Training (WHU Datasets -> Hue Dataset)",
        "Stage 1: Multi-source pretraining on WHU Aerial + Wuhan East Asian Satellite + Hue train quadrants; Stage 2: Target fine-tuning on Hue dataset.",
        results["pretrained_whu_hue_2stage"],
    )
    t_sam = format_sam_ablation(results)
    print(t1)
    print(t2)
    print(t3)
    print(t_sam)
