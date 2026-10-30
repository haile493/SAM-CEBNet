"""
Unified Training, Evaluation, and Inference Pipeline for the 9 Remote Sensing Studies (2022–2026).

Usage Examples with Custom Data:
--------------------------------
1. Single-Image Building Detection / Extraction (`cbrnet` or `maskrcnn_building`):
   python run_pipeline.py --method cbrnet --mode train --data_dir ./my_data --epochs 20 --img_size 256
   python run_pipeline.py --method maskrcnn_building --mode infer --data_dir ./my_data --output_dir ./outputs

2. Bi-Temporal Change Detection (`bisam_cd`, `sam2_cd`, `gmfimamba`, `semibcd`, `cgcce_net`, `edgerefnet`):
   python run_pipeline.py --method bisam_cd --mode infer --data_dir ./my_bitemporal_data --output_dir ./outputs
   python run_pipeline.py --method sam2_cd --mode train --data_dir ./my_bitemporal_data --epochs 20
   python run_pipeline.py --method gmfimamba --mode train --data_dir ./my_bitemporal_data --task_mode bcd
   python run_pipeline.py --method semibcd --mode train --data_dir ./my_bitemporal_data --epochs 20
   python run_pipeline.py --method cgcce_net --mode train --data_dir ./my_bitemporal_data --epochs 20
   python run_pipeline.py --method edgerefnet --mode train --data_dir ./my_bitemporal_data --epochs 20

3. Bi-Temporal Building Damage Captioning (`kgbdcnet`):
   python run_pipeline.py --method kgbdcnet --mode train --data_dir ./my_caption_data --epochs 20
   python run_pipeline.py --method kgbdcnet --mode infer --data_dir ./my_caption_data --output_dir ./outputs

4. End-to-End Demo & Verification across all 9 methods (auto-generates sample dataset if needed):
   python run_pipeline.py --mode demo_all
"""

import argparse
import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ag_models import (
    BinaryDiceLoss,
    CBRNetCompositeLoss,
    EdgeGuidedChangeLoss,
    SAMCEBLoss,
    SemiBCDLoss,
    compute_binary_segmentation_metrics,
)
from datasets.custom_datasets import (
    BiTemporalCaptionDataset,
    BiTemporalChangeDataset,
    MultiSourceBuildingPretrainDataset,
    SingleImageBuildingDataset,
    create_or_load_hue_splits,
    generate_synthetic_custom_data,
)
from models import METHOD_CATALOG, create_method_model


def build_dataloader_for_method(
    method: str,
    data_dir: str,
    img_size: Tuple[int, int] = (256, 256),
    batch_size: int = 2,
    augment: bool = False,
    task_mode: str = "bcd",
    max_samples: Optional[int] = None,
    split: str = "all",
    split_strategy: str = "spatial_block",
    tokenizer: Optional[Any] = None,
    single_image_mode: bool = False,
) -> Tuple[DataLoader, Any]:
    task = METHOD_CATALOG[method]["task"]
    if task == "single_image_building":
        ds = SingleImageBuildingDataset(
            data_dir=data_dir,
            img_size=img_size,
            augment=augment,
            max_samples=max_samples,
            split=split,
            split_strategy=split_strategy,
        )
    elif task == "bitemporal_semi_cd":
        ds = BiTemporalChangeDataset(
            data_dir=data_dir,
            img_size=img_size,
            augment=augment,
            return_strong_aug=True,
            max_samples=max_samples,
            split=split,
            split_strategy=split_strategy,
            single_image_mode=single_image_mode,
        )
    elif task == "bitemporal_cd":
        ds = BiTemporalChangeDataset(
            data_dir=data_dir,
            img_size=img_size,
            augment=augment,
            multi_class_damage=(method == "gmfimamba" and task_mode == "bda"),
            max_samples=max_samples,
            split=split,
            split_strategy=split_strategy,
            single_image_mode=single_image_mode,
        )
    elif task == "bitemporal_captioning":
        ds = BiTemporalCaptionDataset(
            data_dir=data_dir,
            img_size=img_size,
            max_samples=max_samples,
            split=split,
            split_strategy=split_strategy,
            tokenizer=tokenizer,
            single_image_mode=single_image_mode,
        )
    else:
        raise ValueError(f"Unknown task '{task}' for method '{method}'")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=augment, num_workers=0)
    return loader, ds


def train_one_epoch(
    method: str,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    task_mode: str = "bcd",
) -> Dict[str, float]:
    model.train()
    dice_loss_fn = BinaryDiceLoss()
    cbr_loss_fn = CBRNetCompositeLoss()
    sam_ceb_loss_fn = SAMCEBLoss()
    semi_loss_fn = SemiBCDLoss()
    edge_loss_fn = EdgeGuidedChangeLoss()

    total_loss_accum = 0.0
    num_batches = 0

    for batch in loader:
        optimizer.zero_grad()
        if method == "cbrnet":
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            bnd = batch["boundary_gt"].to(device)
            cor = batch["corner_gt"].to(device)
            out = model(img)
            loss, _ = cbr_loss_fn(out, mask, bnd, cor)

        elif method in ("sam_cebnet", "sam_cbr_edgenet", "sam3_cebnet"):
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            bnd = batch["boundary_gt"].to(device)
            cor = batch["corner_gt"].to(device)
            out = model(img)
            loss, _ = sam_ceb_loss_fn(out, mask, bnd, cor)

        elif method == "maskrcnn_building":
            img = batch["image"].to(device)
            mask = batch["mask"].to(device)
            out = model(img)
            l_seg = F.binary_cross_entropy_with_logits(out.seg_logits, mask) + dice_loss_fn(out.seg_logits, mask)
            l_rpn = F.binary_cross_entropy_with_logits(out.rpn_logits, mask)
            loss = l_seg + 0.3 * l_rpn

        elif method == "edgerefnet":
            t1 = batch.get("img_t1", batch.get("image")).to(device)
            t2 = batch.get("img_t2", batch.get("image")).to(device)
            mask = batch["mask"].to(device)
            edge_gt = batch.get("edge_gt", batch.get("boundary_gt", mask)).to(device)
            out = model(t1, t2)
            loss, _ = edge_loss_fn(out, mask, edge_gt)

        elif method == "semibcd":
            t1 = batch.get("img_t1", batch.get("image")).to(device)
            t2 = batch.get("img_t2", batch.get("image")).to(device)
            t1_s = batch.get("img_t1_strong", t1).to(device)
            t2_s = batch.get("img_t2_strong", t2).to(device)
            mask = batch["mask"].to(device)
            vlm_pseudo = batch.get("vlm_pseudo", mask).to(device)
            sup_out = model(t1, t2)
            unsup_out = model(t1, t2, need_fp=True, img_t1_strong=t1_s, img_t2_strong=t2_s)
            loss, _ = semi_loss_fn(sup_out, mask, unsup_outputs=unsup_out, vlm_pseudo_targets=vlm_pseudo)

        elif method == "sam2_cd":
            t1 = batch.get("img_t1", batch.get("image")).to(device)
            t2 = batch.get("img_t2", batch.get("image")).to(device)
            mask = batch["mask"].to(device)
            out = model(t1, t2)
            l_main = F.binary_cross_entropy_with_logits(out.seg_logits, mask) + dice_loss_fn(out.seg_logits, mask)
            l_p2 = F.binary_cross_entropy_with_logits(out.mask_p2, mask) + dice_loss_fn(out.mask_p2, mask)
            l_p3 = F.binary_cross_entropy_with_logits(out.mask_p3, mask) + dice_loss_fn(out.mask_p3, mask)
            loss = l_main + 0.5 * l_p2 + 0.25 * l_p3

        elif method == "gmfimamba" and task_mode == "bda":
            t1 = batch.get("img_t1", batch.get("image")).to(device)
            t2 = batch.get("img_t2", batch.get("image")).to(device)
            dmg_mask = batch["mask"].to(device).long()
            loc_mask = batch.get("loc_mask", batch["mask"]).to(device).float()
            out = model(t1, t2)
            l_dmg = F.cross_entropy(out.seg_logits, dmg_mask)
            l_loc = F.binary_cross_entropy_with_logits(out.loc_logits, loc_mask) + dice_loss_fn(
                out.loc_logits, loc_mask
            )
            loss = l_dmg + l_loc

        elif method in ("bisam_cd", "gmfimamba", "cgcce_net"):
            t1 = batch.get("img_t1", batch.get("image")).to(device)
            t2 = batch.get("img_t2", batch.get("image")).to(device)
            mask = batch["mask"].to(device).float()
            out = model(t1, t2)
            loss = F.binary_cross_entropy_with_logits(out.seg_logits, mask) + dice_loss_fn(out.seg_logits, mask)

        elif method == "kgbdcnet":
            t1 = batch.get("img_t1", batch.get("image")).to(device)
            t2 = batch.get("img_t2", batch.get("image")).to(device)
            cap_tokens = batch.get("caption_tokens")
            if cap_tokens is not None:
                cap_tokens = cap_tokens.to(device)
            kw_ids = batch.get("keyword_ids")
            if kw_ids is not None:
                kw_ids = kw_ids.to(device)
            chg_mask = batch.get("mask")
            if chg_mask is not None:
                chg_mask = chg_mask.to(device)
            out = model(t1, t2, caption_targets=cap_tokens, keyword_ids=kw_ids, change_mask=chg_mask)
            if out.loss is not None:
                loss = out.loss
            elif getattr(out, "seg_logits", None) is not None and chg_mask is not None:
                loss = F.binary_cross_entropy_with_logits(out.seg_logits, chg_mask) + dice_loss_fn(out.seg_logits, chg_mask)
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

        else:
            raise ValueError(f"Unsupported method: {method}")

        loss.backward()
        optimizer.step()
        total_loss_accum += float(loss.item())
        num_batches += 1

    return {"train_loss": total_loss_accum / max(num_batches, 1)}


def _compute_caption_metrics(pred_text: str, ref_text: str) -> Dict[str, float]:
    """Computes Token F1, smoothed BLEU-4, ROUGE-L, and Exact Match for building damage captions."""
    pred_toks = [w for w in pred_text.lower().split() if w]
    ref_toks = [w for w in ref_text.lower().split() if w]
    if not pred_toks and not ref_toks:
        return {"token_f1": 1.0, "bleu4": 1.0, "rouge_l": 1.0, "exact_match": 1.0}
    if not pred_toks or not ref_toks:
        return {"token_f1": 0.0, "bleu4": 0.0, "rouge_l": 0.0, "exact_match": 0.0}

    exact = 1.0 if pred_toks == ref_toks else 0.0

    # Bag-of-words overlap for Token F1
    from collections import Counter

    c_pred = Counter(pred_toks)
    c_ref = Counter(ref_toks)
    overlap = sum((c_pred & c_ref).values())
    prec_1 = overlap / len(pred_toks)
    rec_1 = overlap / len(ref_toks)
    token_f1 = (2.0 * prec_1 * rec_1 / (prec_1 + rec_1)) if (prec_1 + rec_1) > 0 else 0.0

    # Smoothed BLEU-4 (n=1..4)
    import math

    log_p_sum = 0.0
    for n in range(1, 5):
        if len(pred_toks) < n or len(ref_toks) < n:
            p_n = 1.0 / (len(pred_toks) + 1.0)
        else:
            p_ngrams = Counter(tuple(pred_toks[i : i + n]) for i in range(len(pred_toks) - n + 1))
            r_ngrams = Counter(tuple(ref_toks[i : i + n]) for i in range(len(ref_toks) - n + 1))
            matches = sum((p_ngrams & r_ngrams).values())
            total = sum(p_ngrams.values())
            p_n = (matches + 1.0) / (total + 1.0)
        log_p_sum += 0.25 * math.log(max(p_n, 1e-9))
    bp = 1.0 if len(pred_toks) >= len(ref_toks) else math.exp(1.0 - len(ref_toks) / max(len(pred_toks), 1))
    bleu4 = float(bp * math.exp(log_p_sum))

    # ROUGE-L via Longest Common Subsequence (LCS)
    m, n_len = len(pred_toks), len(ref_toks)
    dp = [[0] * (n_len + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n_len + 1):
            if pred_toks[i - 1] == ref_toks[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n_len]
    r_prec = lcs / m
    r_rec = lcs / n_len
    rouge_l = (2.0 * r_prec * r_rec / (r_prec + r_rec)) if (r_prec + r_rec) > 0 else 0.0

    return {
        "token_f1": float(token_f1),
        "bleu4": float(bleu4),
        "rouge_l": float(rouge_l),
        "exact_match": float(exact),
    }


def _save_cbrnet_panel(
    rgb_tensor: torch.Tensor,
    gt_mask: torch.Tensor,
    pred_prob: torch.Tensor,
    bnd_prob: Optional[torch.Tensor],
    cor_prob: Optional[torch.Tensor],
    save_path: str,
) -> None:
    """Saves a 4-panel horizontal image: [RGB | GT Building Mask | Pred Mask | Boundary+Corner Overlay]."""
    img_np = (rgb_tensor.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    gt_np = (gt_mask[0].cpu().numpy() >= 0.5).astype(np.uint8) * 255
    pred_bin = (pred_prob[0].cpu().numpy() >= 0.5).astype(np.uint8) * 255

    gt_rgb = np.stack([gt_np, gt_np, gt_np], axis=-1)
    pred_rgb = np.stack([pred_bin, pred_bin, pred_bin], axis=-1)

    overlay = (img_np.astype(np.float32) * 0.65).astype(np.uint8)
    # Green tint for True Positive, Red for False Positive, Blue for False Negative
    tp = (pred_bin > 0) & (gt_np > 0)
    fp = (pred_bin > 0) & (gt_np == 0)
    fn = (pred_bin == 0) & (gt_np > 0)
    overlay[tp] = np.clip(overlay[tp] * 0.4 + np.array([40, 220, 80]) * 0.6, 0, 255).astype(np.uint8)
    overlay[fp] = np.clip(overlay[fp] * 0.4 + np.array([235, 60, 60]) * 0.6, 0, 255).astype(np.uint8)
    overlay[fn] = np.clip(overlay[fn] * 0.4 + np.array([60, 120, 240]) * 0.6, 0, 255).astype(np.uint8)

    if bnd_prob is not None:
        bnd_mask = bnd_prob[0].cpu().numpy() >= 0.45
        overlay[bnd_mask] = [255, 230, 40]  # Yellow refined boundaries
    if cor_prob is not None:
        cor_mask = cor_prob[0].cpu().numpy() >= 0.45
        overlay[cor_mask] = [255, 40, 220]  # Magenta building corners

    sep = np.full((img_np.shape[0], 3, 3), 255, dtype=np.uint8)
    panel = np.concatenate([img_np, sep, gt_rgb, sep, pred_rgb, sep, overlay], axis=1)
    Image.fromarray(panel).save(save_path)


@torch.no_grad()
def evaluate_or_infer(
    method: str,
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: Any,
    device: torch.device,
    output_dir: Optional[str] = None,
    task_mode: str = "bcd",
) -> Dict[str, Any]:
    import time

    model.eval()
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    task = METHOD_CATALOG[method]["task"]
    dice_scores = []
    iou_scores = []
    prec_scores = []
    rec_scores = []
    bnd_f1_scores = []
    generated_captions = []
    cap_token_f1s = []
    cap_bleu4s = []
    cap_rougels = []
    cap_exacts = []

    tp_sum = 0.0
    fp_sum = 0.0
    fn_sum = 0.0
    tn_sum = 0.0
    total_infer_sec = 0.0
    total_tiles_timed = 0

    panel_dir = os.path.join(output_dir, "panels") if output_dir else None
    if panel_dir:
        os.makedirs(panel_dir, exist_ok=True)

    for batch in loader:
        ids = batch["id"]
        bsz = len(ids)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_start = time.perf_counter()

        if task == "single_image_building":
            img = batch["image"].to(device)
            out = model(img)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_infer_sec += time.perf_counter() - t_start
            total_tiles_timed += bsz

            probs = torch.sigmoid(out.seg_logits)
            bnd_probs = torch.sigmoid(out.boundary_logits) if getattr(out, "boundary_logits", None) is not None else None
            cor_probs = torch.sigmoid(out.corner_logits) if getattr(out, "corner_logits", None) is not None else None

            if batch["has_label"].any():
                probs_cpu = probs.cpu()
                mask_cpu = batch["mask"].cpu()
                m = compute_binary_segmentation_metrics(probs_cpu, mask_cpu)
                dice_scores.append(m["dice"])
                iou_scores.append(m["iou"])
                prec_scores.append(m["precision"])
                rec_scores.append(m["recall"])

                pred_bin = (probs_cpu >= 0.5).float()
                tgt_bin = (mask_cpu >= 0.5).float()
                tp_sum += float((pred_bin * tgt_bin).sum().item())
                fp_sum += float((pred_bin * (1.0 - tgt_bin)).sum().item())
                fn_sum += float(((1.0 - pred_bin) * tgt_bin).sum().item())
                tn_sum += float(((1.0 - pred_bin) * (1.0 - tgt_bin)).sum().item())

                if "boundary_gt" in batch:
                    pred_bnd = (
                        bnd_probs.cpu()
                        if bnd_probs is not None
                        else EdgeGuidedChangeLoss.extract_edge_gt(pred_bin)
                    )
                    bm = compute_binary_segmentation_metrics(pred_bnd, batch["boundary_gt"].cpu(), threshold=0.45)
                    bnd_f1_scores.append(bm["dice"])

            if output_dir:
                for i, sample_id in enumerate(ids):
                    pred_np = (probs[i, 0].cpu().numpy() >= 0.5).astype(np.uint8) * 255
                    Image.fromarray(pred_np).save(os.path.join(output_dir, f"{method}_{sample_id}"))
                    if panel_dir:
                        _save_cbrnet_panel(
                            img[i],
                            batch["mask"][i],
                            probs[i],
                            bnd_probs[i] if bnd_probs is not None else None,
                            cor_probs[i] if cor_probs is not None else None,
                            os.path.join(panel_dir, f"panel_{sample_id}"),
                        )

        elif task in ("bitemporal_cd", "bitemporal_semi_cd"):
            t1 = batch["img_t1"].to(device)
            t2 = batch["img_t2"].to(device)
            out = model(t1, t2)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_infer_sec += time.perf_counter() - t_start
            total_tiles_timed += bsz

            if method == "gmfimamba" and task_mode == "bda":
                pred_cls = out.seg_logits.argmax(dim=1)
                if output_dir:
                    for i, sample_id in enumerate(ids):
                        pred_np = (pred_cls[i].cpu().numpy() * 50).clip(0, 255).astype(np.uint8)
                        Image.fromarray(pred_np).save(os.path.join(output_dir, f"{method}_bda_{sample_id}"))
            else:
                probs = torch.sigmoid(out.seg_logits)
                edge_probs = torch.sigmoid(out.edge_logits) if getattr(out, "edge_logits", None) is not None else None
                if batch["has_label"].any():
                    probs_cpu = probs.cpu()
                    mask_cpu = batch["mask"].cpu()
                    m = compute_binary_segmentation_metrics(probs_cpu, mask_cpu)
                    dice_scores.append(m["dice"])
                    iou_scores.append(m["iou"])
                    prec_scores.append(m["precision"])
                    rec_scores.append(m["recall"])

                    pred_bin = (probs_cpu >= 0.5).float()
                    tgt_bin = (mask_cpu >= 0.5).float()
                    tp_sum += float((pred_bin * tgt_bin).sum().item())
                    fp_sum += float((pred_bin * (1.0 - tgt_bin)).sum().item())
                    fn_sum += float(((1.0 - pred_bin) * tgt_bin).sum().item())
                    tn_sum += float(((1.0 - pred_bin) * (1.0 - tgt_bin)).sum().item())

                    if "boundary_gt" in batch:
                        pred_bnd = (
                            edge_probs.cpu()
                            if edge_probs is not None
                            else EdgeGuidedChangeLoss.extract_edge_gt(pred_bin)
                        )
                        bm = compute_binary_segmentation_metrics(pred_bnd, batch["boundary_gt"].cpu(), threshold=0.45)
                        bnd_f1_scores.append(bm["dice"])

                if output_dir:
                    for i, sample_id in enumerate(ids):
                        pred_np = (probs[i, 0].cpu().numpy() >= 0.5).astype(np.uint8) * 255
                        Image.fromarray(pred_np).save(os.path.join(output_dir, f"{method}_{sample_id}"))
                        if panel_dir:
                            _save_cbrnet_panel(
                                t2[i],
                                batch["mask"][i],
                                probs[i],
                                edge_probs[i] if edge_probs is not None else None,
                                None,
                                os.path.join(panel_dir, f"panel_{sample_id}"),
                            )

        elif task == "bitemporal_captioning":
            t1 = batch["img_t1"].to(device)
            t2 = batch["img_t2"].to(device)
            kw_ids = batch["keyword_ids"].to(device)
            out = model(t1, t2, keyword_ids=kw_ids, max_gen_len=20)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            total_infer_sec += time.perf_counter() - t_start
            total_tiles_timed += bsz

            gen_tokens = out.generated_tokens.cpu().tolist()
            ref_texts = batch.get("caption_text", [""] * bsz)
            for i, sample_id in enumerate(ids):
                text = dataset.tokenizer.decode(gen_tokens[i])
                ref_t = ref_texts[i] if i < len(ref_texts) else ""
                cm = _compute_caption_metrics(text, ref_t)
                cap_token_f1s.append(cm["token_f1"])
                cap_bleu4s.append(cm["bleu4"])
                cap_rougels.append(cm["rouge_l"])
                cap_exacts.append(cm["exact_match"])
                generated_captions.append(
                    {"id": sample_id, "generated_caption": text, "reference_caption": ref_t, **cm}
                )

            if getattr(out, "seg_logits", None) is not None and "mask" in batch and batch["has_label"].any():
                probs = torch.sigmoid(out.seg_logits)
                probs_cpu = probs.cpu()
                mask_cpu = batch["mask"].cpu()
                m = compute_binary_segmentation_metrics(probs_cpu, mask_cpu)
                dice_scores.append(m["dice"])
                iou_scores.append(m["iou"])
                prec_scores.append(m["precision"])
                rec_scores.append(m["recall"])

                pred_bin = (probs_cpu >= 0.5).float()
                tgt_bin = (mask_cpu >= 0.5).float()
                tp_sum += float((pred_bin * tgt_bin).sum().item())
                fp_sum += float((pred_bin * (1.0 - tgt_bin)).sum().item())
                fn_sum += float(((1.0 - pred_bin) * tgt_bin).sum().item())
                tn_sum += float(((1.0 - pred_bin) * (1.0 - tgt_bin)).sum().item())

                if "boundary_gt" in batch:
                    pred_bnd = EdgeGuidedChangeLoss.extract_edge_gt(pred_bin)
                    bm = compute_binary_segmentation_metrics(pred_bnd, batch["boundary_gt"].cpu(), threshold=0.45)
                    bnd_f1_scores.append(bm["dice"])

                if output_dir:
                    for i, sample_id in enumerate(ids):
                        pred_np = (probs[i, 0].cpu().numpy() >= 0.5).astype(np.uint8) * 255
                        Image.fromarray(pred_np).save(os.path.join(output_dir, f"{method}_{sample_id}"))
                        if panel_dir:
                            _save_cbrnet_panel(
                                t2[i],
                                batch["mask"][i],
                                probs[i],
                                None,
                                None,
                                os.path.join(panel_dir, f"panel_{sample_id}"),
                            )

    num_params = sum(p.numel() for p in model.parameters())
    latency_ms = (total_infer_sec * 1000.0) / max(total_tiles_timed, 1)

    results: Dict[str, Any] = {
        "method": method,
        "year": METHOD_CATALOG[method]["year"],
        "task": task,
        "params_m": round(num_params / 1e6, 3),
        "latency_ms_per_tile": round(latency_ms, 2),
        "fps": round(1000.0 / max(latency_ms, 1e-3), 1),
    }
    if dice_scores:
        eps = 1e-7
        global_iou = (tp_sum + eps) / (tp_sum + fp_sum + fn_sum + eps)
        global_dice = (2.0 * tp_sum + eps) / (2.0 * tp_sum + fp_sum + fn_sum + eps)
        global_prec = (tp_sum + eps) / (tp_sum + fp_sum + eps)
        global_rec = (tp_sum + eps) / (tp_sum + fn_sum + eps)
        global_acc = (tp_sum + tn_sum + eps) / (tp_sum + tn_sum + fp_sum + fn_sum + eps)

        results["mean_dice"] = float(global_dice)
        results["mean_iou"] = float(global_iou)
        results["precision"] = float(global_prec)
        results["recall"] = float(global_rec)
        results["overall_accuracy"] = float(global_acc)
        results["tile_avg_dice"] = float(np.mean(dice_scores))
        results["tile_avg_iou"] = float(np.mean(iou_scores))
        if bnd_f1_scores:
            results["boundary_f1"] = float(np.mean(bnd_f1_scores))
    if generated_captions:
        results["caption_token_f1"] = float(np.mean(cap_token_f1s))
        results["caption_bleu4"] = float(np.mean(cap_bleu4s))
        results["caption_rouge_l"] = float(np.mean(cap_rougels))
        results["caption_exact_match"] = float(np.mean(cap_exacts))
        results["sample_captions"] = generated_captions[:3]
        if output_dir:
            with open(os.path.join(output_dir, f"{method}_captions.json"), "w", encoding="utf-8") as f:
                json.dump(generated_captions, f, indent=2)
    return results


def run_method(
    method: str,
    mode: str,
    data_dir: str,
    output_dir: str = "./outputs",
    checkpoint: Optional[str] = None,
    img_size: int = 128,
    base_channels: int = 16,
    batch_size: int = 2,
    epochs: int = 2,
    lr: float = 1e-3,
    task_mode: str = "bcd",
    max_samples: Optional[int] = None,
    split: str = "unseen_test",
    split_strategy: str = "spatial_block",
    pretrained: bool = False,
    pretrained_arch: str = "resnet50",
    single_image_mode: bool = False,
    whu_pretrain: bool = False,
    whu_epochs: int = 12,
    device_str: Optional[str] = None,
) -> Dict[str, Any]:
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    create_or_load_hue_splits(data_dir, strategy=split_strategy)

    train_split = "train" if mode == "train" else split
    train_loader, train_dataset = build_dataloader_for_method(
        method=method,
        data_dir=data_dir,
        img_size=(img_size, img_size),
        batch_size=batch_size,
        augment=(mode == "train"),
        task_mode=task_mode,
        max_samples=max_samples,
        split=train_split,
        split_strategy=split_strategy,
        single_image_mode=single_image_mode,
    )

    model_kwargs: Dict[str, Any] = {"in_channels": 3}
    if method == "bisam_cd":
        model_kwargs["embed_dim"] = base_channels
        model_kwargs["num_interp_frames"] = 4
    elif method == "kgbdcnet":
        model_kwargs["vocab_size"] = train_dataset.tokenizer.vocab_size
        model_kwargs["embed_dim"] = max(base_channels * 2, 32)
        model_kwargs["num_heads"] = 4
        model_kwargs["pretrained_backbone"] = pretrained
        model_kwargs["pretrained_arch"] = pretrained_arch
    elif method == "gmfimamba":
        model_kwargs["base_channels"] = base_channels
        model_kwargs["task_mode"] = task_mode
        model_kwargs["pretrained_backbone"] = pretrained
        model_kwargs["pretrained_arch"] = pretrained_arch
    elif method in ("cbrnet", "sam_cebnet", "sam_cbr_edgenet", "sam3_cebnet"):
        model_kwargs["base_channels"] = base_channels
        model_kwargs["fpn_channels"] = base_channels * 2
        model_kwargs["pretrained_backbone"] = pretrained
        model_kwargs["pretrained_arch"] = pretrained_arch
        if method in ("sam_cebnet", "sam_cbr_edgenet"):
            model_kwargs["pretrained_sam"] = pretrained
            model_kwargs["sam_arch"] = "sam2_hiera_tiny"
            model_kwargs["sam_version"] = "sam2"
        elif method == "sam3_cebnet":
            model_kwargs["pretrained_sam"] = pretrained
            model_kwargs["sam_arch"] = "sam2_hiera_tiny"
            model_kwargs["sam_version"] = "sam3"
    elif method in ("maskrcnn_building", "sam2_cd", "semibcd", "cgcce_net", "edgerefnet"):
        model_kwargs["base_channels"] = base_channels
        model_kwargs["pretrained_backbone"] = pretrained
        model_kwargs["pretrained_arch"] = pretrained_arch
    else:
        model_kwargs["base_channels"] = base_channels

    model = create_method_model(method, **model_kwargs).to(device)
    if checkpoint and os.path.isfile(checkpoint):
        model.load_checkpoint(checkpoint, map_location=device)

    os.makedirs(output_dir, exist_ok=True)
    history = []
    val_stats_best: Optional[Dict[str, Any]] = None

    if mode == "train":
        val_loader, val_dataset = build_dataloader_for_method(
            method=method,
            data_dir=data_dir,
            img_size=(img_size, img_size),
            batch_size=batch_size,
            augment=False,
            task_mode=task_mode,
            max_samples=max_samples,
            split="val",
            split_strategy=split_strategy,
            tokenizer=getattr(train_dataset, "tokenizer", None),
            single_image_mode=single_image_mode,
        )
        if hasattr(model, "sam_adapter") and hasattr(model, "cnn_backbone"):
            sam_enc_ids = {id(p) for p in model.sam_adapter.sam_encoder.parameters()}
            cnn_ids = {id(p) for p in model.cnn_backbone.parameters()}
            sam_enc_params = [p for p in model.sam_adapter.sam_encoder.parameters() if p.requires_grad]
            cnn_params = [p for p in model.cnn_backbone.parameters() if p.requires_grad]
            head_params = [
                p for p in model.parameters() if p.requires_grad and id(p) not in sam_enc_ids and id(p) not in cnn_ids
            ]
            optimizer = torch.optim.AdamW(
                [
                    {"params": sam_enc_params, "lr": lr * 0.08},
                    {"params": cnn_params, "lr": lr * 0.25},
                    {"params": head_params, "lr": lr},
                ],
                weight_decay=1e-4,
            )
        elif (pretrained or method in ("sam_cebnet", "sam_cbr_edgenet", "sam3_cebnet")) and hasattr(model, "backbone"):
            backbone_ids = {id(p) for p in model.backbone.parameters()}
            backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
            head_params = [p for p in model.parameters() if p.requires_grad and id(p) not in backbone_ids]
            optimizer = torch.optim.AdamW(
                [
                    {"params": backbone_params, "lr": lr * 0.25},
                    {"params": head_params, "lr": lr},
                ],
                weight_decay=1e-4,
            )
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1), eta_min=lr * 0.05)
        best_val_iou = -1.0
        best_ckpt_path = os.path.join(output_dir, f"{method}_best.pt")

        # Optional Stage-1 Multi-Source Pretraining (WHU Aerial + Wuhan East Asian Satellite + Hue Train Quadrants)
        if whu_pretrain and whu_epochs > 0:
            print(
                f"[{method} | Stage-1 Pretrain] Curating WHU Aerial + Wuhan East Asian Satellite + Hue Train Quadrants...",
                flush=True,
            )
            pretrain_ds = MultiSourceBuildingPretrainDataset(
                hue_data_dir=data_dir,
                whu_cache_dir="./dataset_whu_pretrain",
                img_size=(img_size, img_size),
                augment=True,
                split_strategy=split_strategy,
                target_whu_aerial=160,
                target_east_asia_sat=160,
            )
            pretrain_loader = DataLoader(pretrain_ds, batch_size=batch_size, shuffle=True, num_workers=0)
            print(
                f"[{method} | Stage-1 Pretrain] Loaded {len(pretrain_ds)} multi-source building tiles for {whu_epochs} epochs.",
                flush=True,
            )
            if hasattr(model, "sam_adapter") and hasattr(model, "cnn_backbone"):
                sam_enc_ids = {id(p) for p in model.sam_adapter.sam_encoder.parameters()}
                cnn_ids = {id(p) for p in model.cnn_backbone.parameters()}
                sam_enc_params = [p for p in model.sam_adapter.sam_encoder.parameters() if p.requires_grad]
                cnn_params = [p for p in model.cnn_backbone.parameters() if p.requires_grad]
                head_params = [
                    p
                    for p in model.parameters()
                    if p.requires_grad and id(p) not in sam_enc_ids and id(p) not in cnn_ids
                ]
                pre_opt = torch.optim.AdamW(
                    [
                        {"params": sam_enc_params, "lr": lr * 0.08},
                        {"params": cnn_params, "lr": lr * 0.25},
                        {"params": head_params, "lr": lr},
                    ],
                    weight_decay=1e-4,
                )
            else:
                pre_opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

            pre_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                pre_opt, T_max=max(whu_epochs, 1), eta_min=lr * 0.1
            )
            for pep in range(1, whu_epochs + 1):
                p_stats = train_one_epoch(method, model, pretrain_loader, pre_opt, device, task_mode=task_mode)
                pre_sched.step()
                if pep == whu_epochs or pep % max(1, whu_epochs // 3) == 0:
                    val_cur = evaluate_or_infer(
                        method=method,
                        model=model,
                        loader=val_loader,
                        dataset=val_dataset,
                        device=device,
                        output_dir=None,
                        task_mode=task_mode,
                    )
                    cur_iou = val_cur.get("mean_iou", 0.0)
                    if cur_iou >= best_val_iou:
                        best_val_iou = cur_iou
                        val_stats_best = val_cur
                        model.save_checkpoint(best_ckpt_path, epoch=0, metrics=val_cur)
                    print(
                        f"[{method} | Stage-1 Pretrain {pep:02d}/{whu_epochs:02d}] "
                        f"pretrain_loss={p_stats['train_loss']:.4f} | "
                        f"hue_val_IoU={cur_iou:.4f} | hue_val_Dice={val_cur.get('mean_dice', 0.0):.4f}",
                        flush=True,
                    )
            stage1_ckpt = os.path.join(output_dir, f"{method}_stage1_whu_pretrain.pt")
            model.save_checkpoint(stage1_ckpt, epoch=whu_epochs, metrics={"stage": "whu_pretrain"})
            # Scale Stage-2 learning rate by 0.65 for smooth Hue fine-tuning
            for pg in optimizer.param_groups:
                pg["lr"] = pg["lr"] * 0.65
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(epochs, 1), eta_min=lr * 0.03
            )

        val_interval = 2 if whu_pretrain else max(1, epochs // 5)
        for ep in range(1, epochs + 1):
            train_stats = train_one_epoch(method, model, train_loader, optimizer, device, task_mode=task_mode)
            scheduler.step()
            # Validate on the buffer `val` split
            if ep == epochs or ep % val_interval == 0:
                val_cur = evaluate_or_infer(
                    method=method,
                    model=model,
                    loader=val_loader,
                    dataset=val_dataset,
                    device=device,
                    output_dir=None,
                    task_mode=task_mode,
                )
                cur_iou = val_cur.get("mean_iou", val_cur.get("caption_token_f1", 0.0))
                if cur_iou >= best_val_iou:
                    best_val_iou = cur_iou
                    val_stats_best = val_cur
                    model.save_checkpoint(best_ckpt_path, epoch=ep, metrics=val_cur)
                extra_str = (
                    f" | val_CapF1={val_cur['caption_token_f1']:.4f}"
                    if "caption_token_f1" in val_cur
                    else ""
                )
                print(
                    f"[{method} | Epoch {ep:02d}/{epochs:02d}] train_loss={train_stats['train_loss']:.4f} | "
                    f"val_IoU={val_cur.get('mean_iou', 0.0):.4f} | val_Dice={val_cur.get('mean_dice', 0.0):.4f}"
                    f"{extra_str}",
                    flush=True,
                )
            history.append({"epoch": ep, **train_stats})

        ckpt_path = os.path.join(output_dir, f"{method}_latest.pt")
        model.save_checkpoint(ckpt_path, epoch=epochs, metrics=history[-1] if history else {})
        if os.path.isfile(best_ckpt_path):
            model.load_checkpoint(best_ckpt_path, map_location=device)

    # Always evaluate on the held-out `unseen_test` split after training (or on the requested `split` in eval/infer)
    eval_split = "unseen_test" if mode == "train" else split
    eval_loader, eval_dataset = build_dataloader_for_method(
        method=method,
        data_dir=data_dir,
        img_size=(img_size, img_size),
        batch_size=batch_size,
        augment=False,
        task_mode=task_mode,
        max_samples=max_samples,
        split=eval_split,
        split_strategy=split_strategy,
        tokenizer=getattr(train_dataset, "tokenizer", None),
        single_image_mode=single_image_mode,
    )

    eval_stats = evaluate_or_infer(
        method=method,
        model=model,
        loader=eval_loader,
        dataset=eval_dataset,
        device=device,
        output_dir=output_dir,
        task_mode=task_mode,
    )
    eval_stats["single_image_mode"] = single_image_mode
    eval_stats["whu_pretrain"] = whu_pretrain
    eval_stats["pretrained_backbone"] = pretrained
    if pretrained:
        eval_stats["pretrained_arch"] = pretrained_arch
    eval_stats["eval_split"] = eval_split
    eval_stats["eval_tiles"] = len(eval_dataset)
    if mode == "train":
        eval_stats["train_tiles"] = len(train_dataset)
        if val_stats_best is not None:
            eval_stats["val_metrics"] = {
                "val_tiles": len(val_dataset),
                "mean_iou": val_stats_best.get("mean_iou"),
                "mean_dice": val_stats_best.get("mean_dice"),
                "precision": val_stats_best.get("precision"),
                "recall": val_stats_best.get("recall"),
                "overall_accuracy": val_stats_best.get("overall_accuracy"),
                "boundary_f1": val_stats_best.get("boundary_f1"),
                "caption_token_f1": val_stats_best.get("caption_token_f1"),
                "caption_bleu4": val_stats_best.get("caption_bleu4"),
                "caption_rouge_l": val_stats_best.get("caption_rouge_l"),
                "caption_exact_match": val_stats_best.get("caption_exact_match"),
            }
    if history:
        eval_stats["train_loss_initial"] = history[0]["train_loss"]
        eval_stats["train_loss_final"] = history[-1]["train_loss"]
        with open(os.path.join(output_dir, f"{method}_metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": eval_stats, "history": history}, f, indent=2)
    return eval_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Runner for 9 Remote Sensing Studies (2022–2026)")
    parser.add_argument(
        "--method",
        type=str,
        default="cbrnet",
        choices=list(METHOD_CATALOG.keys()),
        help="Study method to run",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="demo_all",
        choices=["train", "eval", "infer", "demo_all", "benchmark_all"],
        help="Execution mode",
    )
    parser.add_argument("--data_dir", type=str, default="./dataset", help="Path to Hue dataset or custom dataset folder")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Directory for checkpoints & predictions")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint path to load")
    parser.add_argument("--img_size", type=int, default=128, help="Input spatial resolution H=W")
    parser.add_argument("--base_channels", type=int, default=16, help="Base feature channel width")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--task_mode", type=str, default="bcd", choices=["bcd", "bda"], help="BCD or BDA for GMFIMamba")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use ImageNet-pretrained ResNet pyramid backbone and fine-tune on the target dataset",
    )
    parser.add_argument(
        "--pretrained_arch",
        type=str,
        default="resnet50",
        choices=["resnet34", "resnet50"],
        help="Pretrained ResNet backbone architecture when --pretrained is set (default: resnet50 per CBRNet paper)",
    )
    parser.add_argument(
        "--single_image",
        action="store_true",
        help="Run all methods (including bi-temporal architectures) on pure single-image Hue building extraction (100%% footprints, zero GT leakage into T1)",
    )
    parser.add_argument(
        "--whu_pretrain",
        action="store_true",
        help="Run Stage-1 pretraining on WHU Aerial + Wuhan East Asian Satellite + Hue Train Quadrants before Stage-2 Hue fine-tuning",
    )
    parser.add_argument(
        "--whu_epochs",
        type=int,
        default=12,
        help="Number of Stage-1 WHU+Satellite pretraining epochs when --whu_pretrain is set",
    )
    parser.add_argument(
        "--reuse_existing",
        action="store_true",
        help="In benchmark_all mode, reuse existing trained checkpoints/metrics if already present",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="unseen_test",
        choices=["train", "val", "unseen_test", "test", "all"],
        help="Dataset split for eval/infer (train mode automatically trains on 'train' and evaluates on 'unseen_test')",
    )
    parser.add_argument(
        "--split_strategy",
        type=str,
        default="spatial_block",
        choices=["spatial_block", "stratified", "random"],
        help="Splitting strategy to guarantee zero data leakage into unseen_test",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Max number of tiles to load per split (defaults to 4 in demo_all mode, or all tiles in train/eval/infer)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir) or not os.listdir(args.data_dir):
        print(f"[INFO] Creating synthetic sample dataset at '{args.data_dir}'...")
        generate_synthetic_custom_data(args.data_dir, num_samples=6, img_size=args.img_size)

    split_info = create_or_load_hue_splits(args.data_dir, strategy=args.split_strategy)

    if args.mode == "demo_all":
        demo_samples = args.max_samples if args.max_samples is not None else 4
        print("=" * 80)
        print(
            f"Running End-to-End Verification across all 9 Chronological Methods on Hue Dataset "
            f"(strategy={args.split_strategy}, counts={split_info['counts']}, single_image={args.single_image})"
        )
        print("=" * 80)
        summary = []
        for key, meta in sorted(METHOD_CATALOG.items(), key=lambda kv: kv[1]["order"]):
            run_mode = "infer" if key == "bisam_cd" else "train"
            stats = run_method(
                method=key,
                mode=run_mode,
                data_dir=args.data_dir,
                output_dir=os.path.join(args.output_dir, key),
                img_size=args.img_size,
                base_channels=args.base_channels,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                max_samples=demo_samples,
                split="unseen_test",
                split_strategy=args.split_strategy,
                pretrained=args.pretrained,
                pretrained_arch=args.pretrained_arch,
                single_image_mode=args.single_image,
            )
            print(f"[{meta['order']}/9] ({meta['year']}) {key:18s} | {stats}")
            summary.append(stats)
        print("=" * 80)
        print("All 9 methods trained on 'train' and evaluated on locked 'unseen_test':", os.path.abspath(args.output_dir))
    elif args.mode == "benchmark_all":
        print("=" * 95, flush=True)
        print(
            f"Full Hue Benchmark Across All 9 Chronological Methods (2022–2026) "
            f"| Split={args.split_strategy} {split_info['counts']} | img_size={args.img_size} | epochs={args.epochs} "
            f"| single_image={args.single_image}",
            flush=True,
        )
        print("=" * 95, flush=True)
        summary = []
        for key, meta in sorted(METHOD_CATALOG.items(), key=lambda kv: kv[1]["order"]):
            method_out_dir = os.path.join(args.output_dir, key)
            metrics_path = os.path.join(method_out_dir, f"{key}_metrics.json")
            best_ckpt = os.path.join(method_out_dir, f"{key}_best.pt")

            # For BiSAM-CD, also record zero-shot performance before fine-tuning
            if key == "bisam_cd":
                zs_stats = run_method(
                    method=key,
                    mode="infer",
                    data_dir=args.data_dir,
                    output_dir=os.path.join(args.output_dir, "bisam_cd_zeroshot"),
                    img_size=args.img_size,
                    base_channels=args.base_channels,
                    batch_size=args.batch_size,
                    epochs=0,
                    lr=args.lr,
                    max_samples=args.max_samples,
                    split="unseen_test",
                    split_strategy=args.split_strategy,
                    single_image_mode=args.single_image,
                )
                zs_val = run_method(
                    method=key,
                    mode="infer",
                    data_dir=args.data_dir,
                    output_dir=os.path.join(args.output_dir, "bisam_cd_zeroshot_val"),
                    img_size=args.img_size,
                    base_channels=args.base_channels,
                    batch_size=args.batch_size,
                    epochs=0,
                    lr=args.lr,
                    max_samples=args.max_samples,
                    split="val",
                    split_strategy=args.split_strategy,
                    single_image_mode=args.single_image,
                )
                zs_stats["mode_tag"] = "zero_shot"
                zs_stats["val_metrics"] = {
                    "val_tiles": zs_val.get("eval_tiles"),
                    "mean_iou": zs_val.get("mean_iou"),
                    "mean_dice": zs_val.get("mean_dice"),
                    "precision": zs_val.get("precision"),
                    "recall": zs_val.get("recall"),
                    "overall_accuracy": zs_val.get("overall_accuracy"),
                    "boundary_f1": zs_val.get("boundary_f1"),
                }
                print(
                    f"[2a/9] (2025) bisam_cd (Zero-Shot) | val_Dice={zs_val.get('mean_dice', 0):.4f} | "
                    f"test_Dice={zs_stats.get('mean_dice', 0):.4f} | test_IoU={zs_stats.get('mean_iou', 0):.4f}",
                    flush=True,
                )

            if args.reuse_existing and os.path.isfile(metrics_path) and os.path.isfile(best_ckpt):
                with open(metrics_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)["summary"]
                eval_only = run_method(
                    method=key,
                    mode="eval",
                    data_dir=args.data_dir,
                    output_dir=method_out_dir,
                    checkpoint=best_ckpt,
                    img_size=args.img_size,
                    base_channels=args.base_channels,
                    batch_size=args.batch_size,
                    epochs=args.epochs,
                    lr=args.lr,
                    max_samples=args.max_samples,
                    split="unseen_test",
                    split_strategy=args.split_strategy,
                    pretrained=bool(cached.get("pretrained_backbone", False)),
                    pretrained_arch=str(cached.get("pretrained_arch", "resnet34")),
                    single_image_mode=args.single_image,
                )
                cached.update(
                    {
                        "params_m": eval_only.get("params_m"),
                        "latency_ms_per_tile": eval_only.get("latency_ms_per_tile"),
                        "fps": eval_only.get("fps"),
                        "boundary_f1": eval_only.get("boundary_f1", cached.get("boundary_f1")),
                        "single_image_mode": args.single_image,
                    }
                )
                stats = cached
                if key == "bisam_cd":
                    stats["zero_shot_unseen_test"] = zs_stats
                print(
                    f"[{meta['order']}/9] ({meta['year']}) {key:18s} (Reused Checkpoint) | "
                    f"val_Dice={stats.get('val_metrics', {}).get('mean_dice', 0):.4f} | "
                    f"test_Dice={stats.get('mean_dice', 0):.4f} | test_IoU={stats.get('mean_iou', 0):.4f}",
                    flush=True,
                )
                summary.append(stats)
                continue

            stats = run_method(
                method=key,
                mode="train",
                data_dir=args.data_dir,
                output_dir=method_out_dir,
                img_size=args.img_size,
                base_channels=args.base_channels,
                batch_size=args.batch_size,
                epochs=args.epochs,
                lr=args.lr,
                max_samples=args.max_samples,
                split="unseen_test",
                split_strategy=args.split_strategy,
                pretrained=args.pretrained,
                pretrained_arch=args.pretrained_arch,
                single_image_mode=args.single_image,
            )
            if key == "bisam_cd":
                stats["zero_shot_unseen_test"] = zs_stats
            print(
                f"--> [{meta['order']}/9] ({meta['year']}) {key:18s} DONE | "
                f"val_Dice={stats.get('val_metrics', {}).get('mean_dice', 0):.4f} | "
                f"test_Dice={stats.get('mean_dice', 0):.4f} | test_IoU={stats.get('mean_iou', 0):.4f}",
                flush=True,
            )
            summary.append(stats)

        master_json = os.path.join(args.output_dir, "hue_9methods_comparison.json")
        with open(master_json, "w", encoding="utf-8") as f:
            json.dump({"split_info": split_info["counts"], "single_image_mode": args.single_image, "methods": summary}, f, indent=2)
        print("=" * 95, flush=True)
        print("Saved complete 9-method comparison to:", os.path.abspath(master_json), flush=True)
    else:
        stats = run_method(
            method=args.method,
            mode=args.mode,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            checkpoint=args.checkpoint,
            img_size=args.img_size,
            base_channels=args.base_channels,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            task_mode=args.task_mode,
            max_samples=args.max_samples,
            split=args.split,
            split_strategy=args.split_strategy,
            pretrained=args.pretrained,
            pretrained_arch=args.pretrained_arch,
            single_image_mode=args.single_image,
            whu_pretrain=args.whu_pretrain,
            whu_epochs=args.whu_epochs,
        )
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

