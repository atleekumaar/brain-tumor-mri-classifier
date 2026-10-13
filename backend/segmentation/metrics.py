"""
NEUROSCAN AI — 3D Brain Tumor Segmentation Evaluation Metrics
Rigorously implements Dice coefficient, IoU, sensitivity, specificity,
and volume agreement with explicit empty mask and edge-case handling.
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import numpy as np
import torch


def compute_binary_metrics(
    pred_binary: np.ndarray,
    target_binary: np.ndarray,
    voxel_spacing: Optional[Tuple[float, float, float]] = None
) -> Dict[str, float]:
    """
    Computes segmentation metrics between two binary boolean/0-1 volumes.
    Handles empty masks explicitly:
      - If both pred and target are empty: Dice = 1.0, IoU = 1.0, Sensitivity = 1.0.
      - If one is empty and the other is not: Dice = 0.0, IoU = 0.0, Sensitivity = 0.0.
    """
    pred_b = (pred_binary > 0).astype(np.uint8)
    target_b = (target_binary > 0).astype(np.uint8)

    tp = int(np.sum((pred_b == 1) & (target_b == 1)))
    fp = int(np.sum((pred_b == 1) & (target_b == 0)))
    fn = int(np.sum((pred_b == 0) & (target_b == 1)))
    tn = int(np.sum((pred_b == 0) & (target_b == 0)))

    pred_voxels = int(np.sum(pred_b))
    target_voxels = int(np.sum(target_b))

    # Empty mask edge cases
    if pred_voxels == 0 and target_voxels == 0:
        dice = 1.0
        iou = 1.0
        sensitivity = 1.0
        specificity = 1.0
        vol_diff_pct = 0.0
    elif pred_voxels == 0 or target_voxels == 0:
        dice = 0.0
        iou = 0.0
        sensitivity = 0.0 if target_voxels > 0 else 1.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        vol_diff_pct = 100.0
    else:
        dice = float((2.0 * tp) / (2.0 * tp + fp + fn))
        iou = float(tp / (tp + fp + fn))
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        vol_diff_pct = float(abs(pred_voxels - target_voxels) / max(target_voxels, 1) * 100.0)

    # Physical volume if spacing provided (dx, dy, dz) in mm
    voxel_vol_mm3 = 1.0
    if voxel_spacing is not None and len(voxel_spacing) >= 3:
        voxel_vol_mm3 = float(voxel_spacing[0] * voxel_spacing[1] * voxel_spacing[2])

    pred_vol_cm3 = round((pred_voxels * voxel_vol_mm3) / 1000.0, 3)
    target_vol_cm3 = round((target_voxels * voxel_vol_mm3) / 1000.0, 3)

    return {
        "dice": round(dice, 4),
        "iou": round(iou, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "pred_voxel_count": pred_voxels,
        "target_voxel_count": target_voxels,
        "pred_volume_cm3": pred_vol_cm3,
        "target_volume_cm3": target_vol_cm3,
        "volume_diff_pct": round(vol_diff_pct, 2)
    }


Tuple_like = Union[List[float], tuple, np.ndarray]


def compute_case_subregion_metrics(
    pred_mask: Union[np.ndarray, torch.Tensor],
    target_mask: Union[np.ndarray, torch.Tensor],
    voxel_spacing: Optional[Tuple_like] = None
) -> Dict[str, Any]:
    """
    Evaluates multi-class segmentation mask according to BraTS subregion guidelines:
      - WT (Whole Tumor): labels {1, 2, 3} (or {1, 2, 4} mapped to {1, 2, 3})
      - TC (Tumor Core): labels {1, 3}
      - ET (Enhancing Tumor): label {3}
      - NCR/NET: label {1}
      - ED (Edema): label {2}
    """
    if isinstance(pred_mask, torch.Tensor):
        pred_np = pred_mask.detach().cpu().numpy()
    else:
        pred_np = np.asarray(pred_mask)

    if isinstance(target_mask, torch.Tensor):
        target_np = target_mask.detach().cpu().numpy()
    else:
        target_np = np.asarray(target_mask)

    # Map potential raw BraTS label 4 to 3 if present in masks
    pred_clean = np.where(pred_np == 4, 3, pred_np)
    target_clean = np.where(target_np == 4, 3, target_np)

    # Whole Tumor (WT): 1 + 2 + 3
    pred_wt = np.isin(pred_clean, [1, 2, 3])
    target_wt = np.isin(target_clean, [1, 2, 3])
    wt_metrics = compute_binary_metrics(pred_wt, target_wt, voxel_spacing)

    # Tumor Core (TC): 1 + 3
    pred_tc = np.isin(pred_clean, [1, 3])
    target_tc = np.isin(target_clean, [1, 3])
    tc_metrics = compute_binary_metrics(pred_tc, target_tc, voxel_spacing)

    # Enhancing Tumor (ET): 3
    pred_et = (pred_clean == 3)
    target_et = (target_clean == 3)
    et_metrics = compute_binary_metrics(pred_et, target_et, voxel_spacing)

    # Peritumoral Edema (ED): 2
    pred_ed = (pred_clean == 2)
    target_ed = (target_clean == 2)
    ed_metrics = compute_binary_metrics(pred_ed, target_ed, voxel_spacing)

    # Necrotic / Non-enhancing Core (NCR): 1
    pred_ncr = (pred_clean == 1)
    target_ncr = (target_clean == 1)
    ncr_metrics = compute_binary_metrics(pred_ncr, target_ncr, voxel_spacing)

    # Mean BraTS Dice across the 3 primary reporting benchmarks (WT, TC, ET)
    mean_dice = round((wt_metrics["dice"] + tc_metrics["dice"] + et_metrics["dice"]) / 3.0, 4)
    mean_iou = round((wt_metrics["iou"] + tc_metrics["iou"] + et_metrics["iou"]) / 3.0, 4)

    return {
        "mean_dice": mean_dice,
        "mean_iou": mean_iou,
        "subregions": {
            "WT": {
                "name": "Whole Tumor",
                "description": "Necrosis + Edema + Enhancing",
                **wt_metrics
            },
            "TC": {
                "name": "Tumor Core",
                "description": "Necrosis + Enhancing Tumor",
                **tc_metrics
            },
            "ET": {
                "name": "Enhancing Tumor",
                "description": "Active Enhancing Rim",
                **et_metrics
            },
            "ED": {
                "name": "Peritumoral Edema",
                "description": "Fluid Infiltration Margin",
                **ed_metrics
            },
            "NCR": {
                "name": "Necrotic Core",
                "description": "Hypointense Core Component",
                **ncr_metrics
            }
        }
    }


def aggregate_dataset_metrics(cases_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes aggregate metrics across multiple validation cases.
    Returns mean, std, median for WT, TC, ET Dice.
    """
    if not cases_metrics:
        return {"error": "No cases provided for aggregation"}

    agg = {}
    for sub in ["WT", "TC", "ET"]:
        dices = [c["subregions"][sub]["dice"] for c in cases_metrics if "subregions" in c]
        ious = [c["subregions"][sub]["iou"] for c in cases_metrics if "subregions" in c]

        agg[sub] = {
            "dice_mean": round(float(np.mean(dices)), 4) if dices else 0.0,
            "dice_std": round(float(np.std(dices)), 4) if dices else 0.0,
            "dice_median": round(float(np.median(dices)), 4) if dices else 0.0,
            "iou_mean": round(float(np.mean(ious)), 4) if ious else 0.0
        }

    overall_mean = round(float(np.mean([c["mean_dice"] for c in cases_metrics])), 4)
    return {
        "total_cases_evaluated": len(cases_metrics),
        "overall_mean_dice": overall_mean,
        "subregions": agg
    }
