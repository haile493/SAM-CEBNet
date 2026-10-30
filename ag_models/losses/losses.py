"""
Standardized, Dimension-Agnostic (2D & 3D) Loss Functions for Single-Task and Multi-Task Models.

Unifies:
- `MultiClassDiceLoss` (works seamlessly on 2D `(B, C, H, W)` and 3D `(B, C, D, H, W)`)
- `BinaryDiceLoss` & `BinaryTumorDiceLoss`
- `FocalLoss` (2D & 3D multi-class or binary)
- `BoundaryEdgeLoss` (for SOPSeg / WaveAttn multi-scale edge supervision)
- `UnifiedMultiTaskLoss` / `TumorAwareMultiTaskLoss` (Segmentation + Deep Supervision + Classification + Regression)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ag_models.core.registry import LOSS_REGISTRY


@LOSS_REGISTRY.register("multiclass_dice_loss", aliases=["dice_loss"])
class MultiClassDiceLoss(nn.Module):
    """
    Dimension-agnostic (2D or 3D) Multi-Class Generalized Dice Loss.
    Automatically detects spatial dimensions (`(2, 3)` for 2D or `(2, 3, 4)` for 3D).
    """

    def __init__(self, smooth: float = 1e-5, include_background: bool = False):
        super().__init__()
        self.smooth = smooth
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        targets_clamped = targets.long().clamp(0, num_classes - 1)
        one_hot = F.one_hot(targets_clamped, num_classes=num_classes)
        # Permute class axis to dim 1: (B, C, ...)
        perm = [0, one_hot.ndim - 1] + list(range(1, one_hot.ndim - 1))
        targets_one_hot = one_hot.permute(*perm).float()

        reduce_dims = (0,) + tuple(range(2, logits.ndim))
        intersection = torch.sum(probs * targets_one_hot, dim=reduce_dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=reduce_dims)

        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        if not self.include_background and num_classes > 1:
            return 1.0 - torch.mean(dice_score[1:])
        return 1.0 - torch.mean(dice_score)


@LOSS_REGISTRY.register("binary_dice_loss")
class BinaryDiceLoss(nn.Module):
    """Dimension-agnostic Binary Soft Dice Loss on logits."""

    def __init__(self, smooth: float = 1e-5, from_logits: bool = True):
        super().__init__()
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits) if self.from_logits else logits
        probs_flat = probs.view(probs.shape[0], -1)
        targets_flat = targets.float().view(targets.shape[0], -1)

        inter = (probs_flat * targets_flat).sum(dim=1)
        card = (probs_flat + targets_flat).sum(dim=1)
        dice = (2.0 * inter + self.smooth) / (card + self.smooth)
        return 1.0 - dice.mean()


@LOSS_REGISTRY.register("binary_tumor_dice_loss")
class BinaryTumorDiceLoss(nn.Module):
    """
    Dedicated Foreground/Lesion Dice Loss (for class indices >= `tumor_start_idx`, or channel 1 in 2-channel logits).
    Handles empty-lesion volumes gracefully by penalizing false-positive probability mass.
    """

    def __init__(self, smooth: float = 1e-5, tumor_start_idx: int = 2):
        super().__init__()
        self.smooth = smooth
        self.tumor_start_idx = tumor_start_idx

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_c = logits.shape[1]
        if num_c > self.tumor_start_idx:
            probs = F.softmax(logits, dim=1)
            pred_tumor_prob = torch.sum(probs[:, self.tumor_start_idx :], dim=1)
            true_tumor_mask = (targets >= self.tumor_start_idx).float()
        elif num_c == 2:
            pred_tumor_prob = torch.sigmoid(logits[:, 1])
            true_tumor_mask = (targets[:, 1] if targets.ndim == logits.ndim else (targets >= 1)).float()
        else:
            pred_tumor_prob = torch.sigmoid(logits[:, 0])
            true_tumor_mask = targets.float().squeeze(1)

        spatial_dims = tuple(range(1, pred_tumor_prob.ndim))
        intersection = torch.sum(pred_tumor_prob * true_tumor_mask, dim=spatial_dims)
        cardinality = torch.sum(pred_tumor_prob + true_tumor_mask, dim=spatial_dims)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        has_tumor = (torch.sum(true_tumor_mask, dim=spatial_dims) > 0).float()
        case_loss = has_tumor * (1.0 - dice) + (1.0 - has_tumor) * torch.mean(pred_tumor_prob, dim=spatial_dims)
        return torch.mean(case_loss)


@LOSS_REGISTRY.register("focal_loss", aliases=["focal_loss_3d", "focal_loss_2d"])
class FocalLoss(nn.Module):
    """Dimension-agnostic Multi-Class Focal Loss."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets.long(), weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal = ((1.0 - pt) ** self.gamma) * ce_loss
        return torch.mean(focal)


@LOSS_REGISTRY.register("unified_multitask_loss", aliases=["tumor_aware_multitask_loss", "multitask_loss"])
class UnifiedMultiTaskLoss(nn.Module):
    """
    Standardized Multi-Task Loss Function supporting:
    - Multi-Class or Binary Segmentation (`seg_logits`)
    - Deep Supervision Auxiliary Segmentation (`aux_seg_logits`)
    - Multi-Label or Multi-Class Classification (`cls_logits`)
    - Quantitative Continuous Regression (`tumor_burden`)
    """

    def __init__(
        self,
        lambda_seg: float = 1.0,
        lambda_cls: float = 0.5,
        lambda_reg: float = 0.2,
        beta_tumor: float = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        cls_pos_weight: Optional[torch.Tensor] = None,
        multi_label_cls: bool = True,
    ):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_cls = lambda_cls
        self.lambda_reg = lambda_reg
        self.beta_tumor = beta_tumor
        self.multi_label_cls = multi_label_cls

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None

        self.dice_loss = MultiClassDiceLoss()
        self.bin_dice_loss = BinaryDiceLoss()
        self.tumor_dice_loss = BinaryTumorDiceLoss()
        self.focal_loss = FocalLoss(gamma=2.0, weight=self.class_weights)
        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=cls_pos_weight)
        self.ce_cls = nn.CrossEntropyLoss()
        self.reg_loss = nn.SmoothL1Loss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        seg_targets: Optional[torch.Tensor] = None,
        cls_targets: Optional[torch.Tensor] = None,
        reg_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        losses: Dict[str, float] = {}
        device = next(iter(outputs.values())).device
        total_loss = torch.tensor(0.0, device=device)

        if "seg_logits" in outputs and seg_targets is not None:
            seg_logits = outputs["seg_logits"]
            # Binary/Multi-label segmentation (e.g., 2 channels: Liver, Tumor) vs Multi-class index target
            if seg_targets.ndim == seg_logits.ndim and seg_targets.shape[1] == seg_logits.shape[1]:
                l_bce = F.binary_cross_entropy_with_logits(seg_logits, seg_targets.float())
                l_dice = self.bin_dice_loss(seg_logits, seg_targets.float())
                l_seg = l_bce + l_dice
                losses["loss_seg_bce"] = l_bce.item()
                losses["loss_seg_dice"] = l_dice.item()
            else:
                cw = self.class_weights
                if cw is not None and cw.shape[0] != seg_logits.shape[1]:
                    cw = None
                l_ce = F.cross_entropy(seg_logits, seg_targets.long(), weight=cw)
                l_mc_dice = self.dice_loss(seg_logits, seg_targets)
                l_tumor_dice = self.tumor_dice_loss(seg_logits, seg_targets)
                l_focal = F.cross_entropy(seg_logits, seg_targets.long(), reduction="none")
                l_focal = (((1.0 - torch.exp(-l_focal)) ** 2.0) * l_focal).mean()
                l_seg = l_ce + l_mc_dice + self.beta_tumor * l_tumor_dice + 0.5 * l_focal
                losses["loss_ce"] = l_ce.item()
                losses["loss_mc_dice"] = l_mc_dice.item()
                losses["loss_tumor_dice"] = l_tumor_dice.item()

            if "aux_seg_logits" in outputs and outputs["aux_seg_logits"] is not None:
                aux_logits = outputs["aux_seg_logits"]
                if seg_targets.ndim == aux_logits.ndim and seg_targets.shape[1] == aux_logits.shape[1]:
                    l_aux = self.bin_dice_loss(aux_logits, seg_targets.float())
                else:
                    l_aux = self.tumor_dice_loss(aux_logits, seg_targets) + self.dice_loss(aux_logits, seg_targets)
                l_seg = l_seg + 0.5 * l_aux
                losses["loss_aux"] = l_aux.item()

            losses["loss_seg"] = l_seg.item()
            total_loss = total_loss + self.lambda_seg * l_seg

        if "cls_logits" in outputs and cls_targets is not None:
            cls_logits = outputs["cls_logits"]
            if self.multi_label_cls or cls_targets.ndim == 2:
                l_cls = self.bce_cls(cls_logits, cls_targets.float())
            else:
                l_cls = self.ce_cls(cls_logits, cls_targets.long())
            losses["loss_cls"] = l_cls.item()
            total_loss = total_loss + self.lambda_cls * l_cls

        if "tumor_burden" in outputs and outputs["tumor_burden"] is not None and reg_targets is not None:
            l_reg = self.reg_loss(outputs["tumor_burden"].squeeze(-1), reg_targets.float().squeeze(-1))
            losses["loss_reg"] = l_reg.item()
            total_loss = total_loss + self.lambda_reg * l_reg

        losses["loss_total"] = total_loss.item()
        return total_loss, losses


TumorAwareMultiTaskLoss = UnifiedMultiTaskLoss
MultiTaskLoss = UnifiedMultiTaskLoss
FocalLoss3D = FocalLoss


@LOSS_REGISTRY.register("cbrnet_composite_loss")
class CBRNetCompositeLoss(nn.Module):
    """
    Composite Multi-Task Loss from CBRNet (Huang et al., IEEE TGRS 2026):
        L_total = L_sem (BCE/CE + Lovasz/IoU) + lambda_b * L_bnd (Boundary IoU + BCE) + lambda_c * L_cor (BCE + MSE)

    Automatically derives morphological boundary masks and Harris corner ground-truth maps
    from binary building masks if explicit boundary/corner ground truths are not passed.
    """

    def __init__(self, lambda_b: float = 0.5, lambda_c: float = 0.3, lambda_mse: float = 0.5):
        super().__init__()
        self.lambda_b = lambda_b
        self.lambda_c = lambda_c
        self.lambda_mse = lambda_mse
        self.dice_loss = BinaryDiceLoss()

    @staticmethod
    def extract_boundary_and_corner_gt(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extracts morphological boundary GT (dilation - erosion) and corner GT
        from a binary mask (B, 1, H, W) using pure PyTorch tensor operations.
        """
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        m = mask.float()
        dilated = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-m, kernel_size=3, stride=1, padding=1)
        boundary_gt = (dilated - eroded).clamp(0.0, 1.0)

        # Corner response on ground-truth mask via horizontal & vertical gradient product
        gx = m[:, :, :, 1:] - m[:, :, :, :-1]
        gx = F.pad(gx, (0, 1, 0, 0)).abs()
        gy = m[:, :, 1:, :] - m[:, :, :-1, :]
        gy = F.pad(gy, (0, 0, 0, 1)).abs()
        ix2 = F.avg_pool2d(gx * gx, kernel_size=3, stride=1, padding=1)
        iy2 = F.avg_pool2d(gy * gy, kernel_size=3, stride=1, padding=1)
        ixy = F.avg_pool2d(gx * gy, kernel_size=3, stride=1, padding=1)
        det = ix2 * iy2 - ixy * ixy
        trace = ix2 + iy2
        r_gt = (det - 0.04 * trace * trace).clamp(min=0.0)
        r_max = r_gt.amax(dim=(-2, -1), keepdim=True) + 1e-6
        corner_gt = (r_gt / r_max > 0.15).float()
        corner_gt = F.max_pool2d(corner_gt, kernel_size=3, stride=1, padding=1)
        return boundary_gt, corner_gt

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        mask_targets: torch.Tensor,
        boundary_targets: Optional[torch.Tensor] = None,
        corner_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if mask_targets.ndim == 3:
            mask_targets = mask_targets.unsqueeze(1)
        mask_f = mask_targets.float()

        if boundary_targets is None or corner_targets is None:
            auto_bnd, auto_cor = self.extract_boundary_and_corner_gt(mask_f)
            if boundary_targets is None:
                boundary_targets = auto_bnd
            if corner_targets is None:
                corner_targets = auto_cor

        seg_logits = outputs["seg_logits"]
        if seg_logits.shape[1] == 1:
            l_sem = F.binary_cross_entropy_with_logits(seg_logits, mask_f) + self.dice_loss(seg_logits, mask_f)
        else:
            l_sem = F.cross_entropy(seg_logits, mask_f.squeeze(1).long())

        l_bnd = torch.tensor(0.0, device=seg_logits.device)
        if "boundary_logits" in outputs and outputs["boundary_logits"] is not None:
            bnd_logits = outputs["boundary_logits"]
            l_bnd = F.binary_cross_entropy_with_logits(bnd_logits, boundary_targets.float()) + self.dice_loss(
                bnd_logits, boundary_targets.float()
            )
            if "coarse_boundary_logits" in outputs and outputs["coarse_boundary_logits"] is not None:
                l_bnd = l_bnd + 0.5 * F.binary_cross_entropy_with_logits(
                    outputs["coarse_boundary_logits"], boundary_targets.float()
                )

        l_cor = torch.tensor(0.0, device=seg_logits.device)
        if "corner_attentions" in outputs and outputs["corner_attentions"]:
            attn_list = outputs["corner_attentions"]
            rnorm_list = outputs.get("corner_responses", attn_list)
            for attn_map, r_norm in zip(attn_list, rnorm_list):
                c_gt = F.interpolate(corner_targets.float(), size=attn_map.shape[-2:], mode="nearest")
                a_clamped = attn_map.clamp(1e-5, 1.0 - 1e-5)
                r_clamped = r_norm.clamp(1e-5, 1.0 - 1e-5)
                l_cor = l_cor + F.binary_cross_entropy(a_clamped, c_gt) + self.lambda_mse * F.mse_loss(r_clamped, c_gt)
            l_cor = l_cor / float(len(attn_list))

        total = l_sem + self.lambda_b * l_bnd + self.lambda_c * l_cor
        return total, {
            "loss_total": total.item(),
            "loss_sem": l_sem.item(),
            "loss_bnd": l_bnd.item(),
            "loss_cor": l_cor.item(),
        }


@LOSS_REGISTRY.register("semibcd_loss")
class SemiBCDLoss(nn.Module):
    """
    Semi-Supervised Building Change Detection Loss from SemiBCD (Liu et al., IEEE TGRS 2026).

    Combines:
    - Supervised BCE + Dice Loss (L_s)
    - UAPLR uncertainty-weighted VLM pseudo-label loss (L_ws)
    - Multi-View Consistency Regularization (L_mvc = L_p2w + L_w2i + L_w2f)
    - Task-Homogeneous Alignment (L_tha = L_fc + L_sc):
      * L_fc: Feature-level L2 distance loss between SS representations (rep_t1, rep_t2)
      * L_sc: Semantic-level cosine alignment loss between SS building probabilities (ss_t1, ss_t2)
    """

    def __init__(
        self,
        margin_fc: float = 1.0,
        margin_sc: float = 0.2,
        conf_thresh: float = 0.85,
        lambda_mvc: float = 0.5,
        lambda_tha: float = 0.2,
    ):
        super().__init__()
        self.margin_fc = margin_fc
        self.margin_sc = margin_sc
        self.conf_thresh = conf_thresh
        self.lambda_mvc = lambda_mvc
        self.lambda_tha = lambda_tha
        self.dice = BinaryDiceLoss()

    def compute_lfc_loss(
        self, rep_t1: torch.Tensor, rep_t2: torch.Tensor, change_mask: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        f1 = F.normalize(rep_t1, p=2, dim=1)
        f2 = F.normalize(rep_t2, p=2, dim=1)
        y = F.interpolate(change_mask.float(), size=f1.shape[-2:], mode="nearest").squeeze(1)
        dist_sq = torch.sum((f1 - f2) ** 2, dim=1)  # (B, H, W)
        dist = torch.sqrt(dist_sq + 1e-8)
        loss_map = (1.0 - y) * dist_sq + y * (F.relu(self.margin_fc - dist) ** 2)
        if valid_mask is not None:
            v = F.interpolate(valid_mask.float(), size=f1.shape[-2:], mode="nearest").squeeze(1)
            return (loss_map * v).sum() / (v.sum() + 1e-6)
        return loss_map.mean()

    def compute_lsc_loss(
        self, ss_logits_t1: torch.Tensor, ss_logits_t2: torch.Tensor, change_mask: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if ss_logits_t1.shape[1] == 1:
            p1 = torch.sigmoid(ss_logits_t1)
            v1 = torch.cat([1.0 - p1, p1], dim=1)
            p2 = torch.sigmoid(ss_logits_t2)
            v2 = torch.cat([1.0 - p2, p2], dim=1)
        else:
            v1 = F.softmax(ss_logits_t1, dim=1)
            v2 = F.softmax(ss_logits_t2, dim=1)

        y = F.interpolate(change_mask.float(), size=v1.shape[-2:], mode="nearest").squeeze(1)
        cos_sim = F.cosine_similarity(v1, v2, dim=1, eps=1e-6)
        loss_map = (1.0 - y) * (1.0 - cos_sim) + y * F.relu(cos_sim - self.margin_sc)
        if valid_mask is not None:
            v = F.interpolate(valid_mask.float(), size=v1.shape[-2:], mode="nearest").squeeze(1)
            return (loss_map * v).sum() / (v.sum() + 1e-6)
        return loss_map.mean()

    def forward(
        self,
        sup_outputs: Dict[str, torch.Tensor],
        sup_targets: torch.Tensor,
        unsup_outputs: Optional[Dict[str, torch.Tensor]] = None,
        vlm_pseudo_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if sup_targets.ndim == 3:
            sup_targets = sup_targets.unsqueeze(1)
        sup_f = sup_targets.float()
        cd_logits = sup_outputs["seg_logits"]
        l_sup = F.binary_cross_entropy_with_logits(cd_logits, sup_f) + self.dice(cd_logits, sup_f)

        l_tha = torch.tensor(0.0, device=cd_logits.device)
        if "ss_rep_t1" in sup_outputs and "ss_rep_t2" in sup_outputs:
            l_fc = self.compute_lfc_loss(sup_outputs["ss_rep_t1"], sup_outputs["ss_rep_t2"], sup_f)
            l_sc = self.compute_lsc_loss(sup_outputs["ss_logits_t1"], sup_outputs["ss_logits_t2"], sup_f)
            l_tha = l_fc + l_sc

        l_mvc = torch.tensor(0.0, device=cd_logits.device)
        if unsup_outputs is not None:
            weak_logits = unsup_outputs["seg_logits"].detach()
            weak_prob = torch.sigmoid(weak_logits)
            pseudo_hard = (weak_prob >= 0.5).float()
            conf_mask = ((weak_prob >= self.conf_thresh) | (weak_prob <= (1.0 - self.conf_thresh))).float()

            if "fp_logits" in unsup_outputs and unsup_outputs["fp_logits"] is not None:
                l_w2f = (
                    F.binary_cross_entropy_with_logits(unsup_outputs["fp_logits"], pseudo_hard, reduction="none")
                    * conf_mask
                ).mean()
                l_mvc = l_mvc + l_w2f

            if "strong_logits" in unsup_outputs and unsup_outputs["strong_logits"] is not None:
                l_w2i = (
                    F.binary_cross_entropy_with_logits(unsup_outputs["strong_logits"], pseudo_hard, reduction="none")
                    * conf_mask
                ).mean()
                l_mvc = l_mvc + l_w2i

            if vlm_pseudo_targets is not None:
                if vlm_pseudo_targets.ndim == 3:
                    vlm_pseudo_targets = vlm_pseudo_targets.unsqueeze(1)
                l_p2w = F.binary_cross_entropy_with_logits(unsup_outputs["seg_logits"], vlm_pseudo_targets.float())
                l_mvc = l_mvc + l_p2w

            if "ss_rep_t1" in unsup_outputs and "ss_rep_t2" in unsup_outputs:
                l_tha_u = self.compute_lfc_loss(
                    unsup_outputs["ss_rep_t1"], unsup_outputs["ss_rep_t2"], pseudo_hard, conf_mask
                ) + self.compute_lsc_loss(
                    unsup_outputs["ss_logits_t1"], unsup_outputs["ss_logits_t2"], pseudo_hard, conf_mask
                )
                l_tha = 0.5 * (l_tha + l_tha_u)

        total = l_sup + self.lambda_mvc * l_mvc + self.lambda_tha * l_tha
        return total, {
            "loss_total": total.item(),
            "loss_sup": l_sup.item(),
            "loss_mvc": l_mvc.item(),
            "loss_tha": l_tha.item(),
        }


@LOSS_REGISTRY.register("edge_guided_change_loss")
class EdgeGuidedChangeLoss(nn.Module):
    """
    Joint Semantic Change & Edge Supervision Loss from EdgeRefNet (Hussin et al., IEEE TGRS 2026).
    Automatically extracts Canny/morphological gradient edge targets from change masks if edge_targets is None.
    """

    def __init__(self, edge_weight: float = 0.5):
        super().__init__()
        self.edge_weight = edge_weight
        self.dice = BinaryDiceLoss()

    @staticmethod
    def extract_edge_gt(mask: torch.Tensor) -> torch.Tensor:
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        m = mask.float()
        dilated = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
        eroded = -F.max_pool2d(-m, kernel_size=3, stride=1, padding=1)
        return (dilated - eroded).clamp(0.0, 1.0)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        change_targets: torch.Tensor,
        edge_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if change_targets.ndim == 3:
            change_targets = change_targets.unsqueeze(1)
        target_f = change_targets.float()
        if edge_targets is None:
            edge_targets = self.extract_edge_gt(target_f)

        seg_logits = outputs["seg_logits"]
        if seg_logits.shape[1] == 1:
            l_cd = F.binary_cross_entropy_with_logits(seg_logits, target_f) + self.dice(seg_logits, target_f)
        else:
            l_cd = F.cross_entropy(seg_logits, target_f.squeeze(1).long())

        l_edge = torch.tensor(0.0, device=seg_logits.device)
        if "edge_logits" in outputs and outputs["edge_logits"] is not None:
            edge_logits = outputs["edge_logits"]
            if edge_logits.shape[1] == 1:
                l_edge = F.binary_cross_entropy_with_logits(edge_logits, edge_targets.float()) + self.dice(
                    edge_logits, edge_targets.float()
                )
            else:
                l_edge = F.cross_entropy(edge_logits, edge_targets.squeeze(1).long())

        total = l_cd + self.edge_weight * l_edge
        return total, {
            "loss_total": total.item(),
            "loss_cd": l_cd.item(),
            "loss_edge": l_edge.item(),
        }


@LOSS_REGISTRY.register("sam_ceb_loss", aliases=["sam_cbr_edge_loss"])
class SAMCEBLoss(nn.Module):
    """
    Unified Joint Loss for SAM-CEBNet (`SAMCEBNet`) combining:
    - CBRNetCompositeLoss (semantic footprint BCE+Dice + DSBR fine/coarse boundary supervision + Harris corner supervision)
    - High-frequency edge branch supervision (`edge_ref_logits` BCE+Dice)
    """

    def __init__(
        self,
        lambda_b: float = 0.5,
        lambda_c: float = 0.25,
        lambda_edge_ref: float = 0.4,
        lambda_mse: float = 1.0,
    ):
        super().__init__()
        self.cbr_loss = CBRNetCompositeLoss(lambda_b=lambda_b, lambda_c=lambda_c, lambda_mse=lambda_mse)
        self.lambda_edge_ref = lambda_edge_ref
        self.dice = BinaryDiceLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        mask_targets: torch.Tensor,
        boundary_targets: Optional[torch.Tensor] = None,
        corner_targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if mask_targets.ndim == 3:
            mask_targets = mask_targets.unsqueeze(1)
        mask_f = mask_targets.float()

        if boundary_targets is None or corner_targets is None:
            auto_bnd, auto_cor = CBRNetCompositeLoss.extract_boundary_and_corner_gt(mask_f)
            if boundary_targets is None:
                boundary_targets = auto_bnd
            if corner_targets is None:
                corner_targets = auto_cor

        base_loss, stats = self.cbr_loss(outputs, mask_f, boundary_targets, corner_targets)

        l_edge_ref = torch.tensor(0.0, device=mask_f.device)
        if "edge_ref_logits" in outputs and outputs["edge_ref_logits"] is not None:
            er_logits = outputs["edge_ref_logits"]
            bnd_f = boundary_targets.float()
            l_edge_ref = F.binary_cross_entropy_with_logits(er_logits, bnd_f) + self.dice(er_logits, bnd_f)

        total = base_loss + self.lambda_edge_ref * l_edge_ref
        stats["loss_edge_ref"] = float(l_edge_ref.item())
        stats["loss_total"] = float(total.item())
        return total, stats


SAMCBREdgeLoss = SAMCEBLoss


@LOSS_REGISTRY.register("dstcd_loss", aliases=["dstcd_mls_loss"])
class DSTCDMultiLoss(nn.Module):
    """
    Multi-Loss Supervision (MLS) from DSTCD (Wang et al., Int. J. Appl. Earth Obs. Geoinf. 146, 105100, 2026).
    Follows Eq. (9) and `BoundaryLoss` in `DSTCD/utils/loss.py`:
        MLS(p1, p2, y) = 1/3 * [ L_bce(p1, y) + L_bce(p2, y) + L_dice(p2, y) ]
    where `p1` (`aux_seg_logits`) is the CDFE prediction before transfer and `p2` (`seg_logits`)
    is the prediction after Task-Guided Feature Transfer.
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth
        self.dice = BinaryDiceLoss(smooth=smooth, from_logits=True)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        y = targets.float()

        p2_logits = outputs["seg_logits"]
        p1_logits = outputs.get("aux_seg_logits")
        if p1_logits is None:
            p1_logits = p2_logits

        l_bce1 = F.binary_cross_entropy_with_logits(p1_logits, y)
        l_bce2 = F.binary_cross_entropy_with_logits(p2_logits, y)
        l_dice2 = self.dice(p2_logits, y)

        total = (l_bce1 + l_bce2 + l_dice2) / 3.0
        return total, {
            "loss_total": float(total.item()),
            "loss_bce1": float(l_bce1.item()),
            "loss_bce2": float(l_bce2.item()),
            "loss_dice2": float(l_dice2.item()),
        }


@LOSS_REGISTRY.register("seg2change_loss", aliases=["cach_loss"])
class Seg2ChangeLoss(nn.Module):
    """
    Composite Loss for Seg2Change CACH (Su et al., arXiv:2604.11231, 2026).
    Follows Eqs. (11)–(14):
        L_total = alpha * L_cd + beta * L_ups + upsilon * L_sim
    where:
    - `L_cd`: Binary/Cross-Entropy + Dice loss on main category-agnostic change logits (`seg_logits`)
    - `L_ups`: Multi-scale upsampling supervision on intermediate decoder paths (`multi_scale_logits`)
    - `L_sim`: Unchanged-region cosine similarity constraint `(1 - cos(sem_rep_t1, sem_rep_t2)) * (1 - y_l)`
    """

    def __init__(
        self,
        alpha: float = 0.8,
        beta: float = 0.1,
        upsilon: float = 0.1,
        temperature: float = 3.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.upsilon = upsilon
        self.temperature = temperature
        self.dice = BinaryDiceLoss()

    def _seg_criterion(self, logits: torch.Tensor, target_f: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] == 1:
            return F.binary_cross_entropy_with_logits(logits, target_f) + self.dice(logits, target_f)
        return F.cross_entropy(logits, target_f.squeeze(1).long())

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        y = targets.float()

        seg_logits = outputs["seg_logits"]
        l_cd = self._seg_criterion(seg_logits, y)

        l_ups = torch.tensor(0.0, device=seg_logits.device)
        ms_list = outputs.get("multi_scale_logits")
        if ms_list:
            for m_logit in ms_list:
                l_ups = l_ups + self._seg_criterion(m_logit, y)
            l_ups = l_ups / float(len(ms_list))

        l_sim = torch.tensor(0.0, device=seg_logits.device)
        sem_a = outputs.get("sem_rep_t1")
        sem_b = outputs.get("sem_rep_t2")
        if sem_a is not None and sem_b is not None:
            pa = F.softmax(sem_a / self.temperature, dim=1)
            pb = F.softmax(sem_b / self.temperature, dim=1)
            cos_sim = F.cosine_similarity(pa, pb, dim=1, eps=1e-6).unsqueeze(1)
            unchanged_mask = (1.0 - y).clamp(0.0, 1.0)
            l_sim = ((1.0 - cos_sim) * unchanged_mask).sum() / (unchanged_mask.sum() + 1e-6)

        total = self.alpha * l_cd + self.beta * l_ups + self.upsilon * l_sim
        return total, {
            "loss_total": float(total.item()),
            "loss_cd": float(l_cd.item()),
            "loss_ups": float(l_ups.item()),
            "loss_sim": float(l_sim.item()),
        }





@LOSS_REGISTRY.register("fapmnet_loss", aliases=["fapm_loss"])
class FAPMNetLoss(nn.Module):
    """
    FAPMNet multi-task loss, Eq. (8): L = l1*L_cd + l2*L_sem(T1) + l3*L_sem(T2) with (1, 0.5, 0.5).
    Adaptation: the letter uses cross-entropy + Lovasz-Softmax on multi-class maps; for binary maps BCE + Dice is used.
    Expects `seg_logits`, and optionally `sem_logits_t1/t2` with targets `sem_t1`, `sem_t2`.
    """

    def __init__(self, lambdas=(1.0, 0.5, 0.5), pos_weight: float = 1.0):
        super().__init__()
        self.lambdas = lambdas
        self.pos_weight = pos_weight
        self.dice = BinaryDiceLoss(from_logits=True)

    def _bce_dice(self, logits, y):
        pw = torch.tensor([self.pos_weight], device=logits.device)
        return F.binary_cross_entropy_with_logits(logits, y, pos_weight=pw) + self.dice(logits, y)

    def forward(self, outputs, targets, sem_t1=None, sem_t2=None):
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        l_cd = self._bce_dice(outputs["seg_logits"], targets.float())
        zero = torch.zeros((), device=l_cd.device)
        l1 = self._bce_dice(outputs["sem_logits_t1"], sem_t1.float()) if sem_t1 is not None and outputs.get("sem_logits_t1") is not None else zero
        l2 = self._bce_dice(outputs["sem_logits_t2"], sem_t2.float()) if sem_t2 is not None and outputs.get("sem_logits_t2") is not None else zero
        total = self.lambdas[0] * l_cd + self.lambdas[1] * l1 + self.lambdas[2] * l2
        return total, {"loss_total": float(total.item()), "loss_cd": float(l_cd.item()), "loss_sem_t1": float(l1.item()), "loss_sem_t2": float(l2.item())}


@LOSS_REGISTRY.register("farmcd_loss", aliases=["farmland_cd_loss"])
class FarmCDLoss(nn.Module):
    """
    FarmCD objective, Eqs. (17)-(21): L = lam_s*(L_farm,1 + L_farm,2) + lam_c*L_cd with
    L_farm = beta*BCE + (1-beta)*Dice (beta=0.75) and a weighted BCE for change with factor W=0.25
    (defaults lam_s=0.5, lam_c=1.0).
    The typeset WBCE is ambiguous about which class W multiplies; here W weights the unchanged (negative) class
    and (1-W) the changed class, which counteracts the class imbalance.
    """

    def __init__(self, lambda_s: float = 0.5, lambda_c: float = 1.0, beta: float = 0.75, w: float = 0.25):
        super().__init__()
        self.lambda_s, self.lambda_c, self.beta, self.w = lambda_s, lambda_c, beta, w
        self.dice = BinaryDiceLoss(from_logits=True)

    def wbce(self, logits, y):
        p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        return -((1 - self.w) * y * torch.log(p) + self.w * (1 - y) * torch.log(1 - p)).mean()

    def farm(self, logits, y):
        return self.beta * F.binary_cross_entropy_with_logits(logits, y) + (1 - self.beta) * self.dice(logits, y)

    def forward(self, outputs, targets, roi_t1=None, roi_t2=None, phase: int = 3):
        if targets.ndim == 3:
            targets = targets.unsqueeze(1)
        l_cd = self.wbce(outputs["seg_logits"], targets.float())
        zero = torch.zeros((), device=l_cd.device)
        lf = zero
        if roi_t1 is not None and roi_t2 is not None and outputs.get("roi_logits_t1") is not None:
            lf = self.farm(outputs["roi_logits_t1"], roi_t1.float()) + self.farm(outputs["roi_logits_t2"], roi_t2.float())
        if phase == 1:
            total = self.lambda_s * lf
        elif phase == 2:
            total = self.lambda_c * l_cd
        else:
            total = self.lambda_s * lf + self.lambda_c * l_cd
        return total, {"loss_total": float(total.item()), "loss_cd": float(l_cd.item()), "loss_farm": float(lf.item())}
