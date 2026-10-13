"""
Unit tests for segmentation metrics, empty mask edge cases, and BraTS subregion calculations.
"""

import numpy as np
import pytest

from backend.segmentation.metrics import (
    compute_binary_metrics,
    compute_case_subregion_metrics,
    aggregate_dataset_metrics
)


def test_perfect_overlap_metrics():
    mask = np.zeros((32, 32, 32), dtype=np.uint8)
    mask[10:20, 10:20, 10:20] = 1

    res = compute_binary_metrics(mask, mask, voxel_spacing=(1.0, 1.0, 1.0))
    assert res["dice"] == 1.0
    assert res["iou"] == 1.0
    assert res["sensitivity"] == 1.0
    assert res["specificity"] == 1.0
    assert res["volume_diff_pct"] == 0.0
    assert res["pred_voxel_count"] == 1000
    assert res["target_voxel_count"] == 1000
    assert res["pred_volume_cm3"] == 1.0


def test_empty_mask_handling():
    empty_pred = np.zeros((32, 32, 32), dtype=np.uint8)
    empty_target = np.zeros((32, 32, 32), dtype=np.uint8)

    # 1. Both empty -> True negative agreement -> Dice = 1.0
    res_both_empty = compute_binary_metrics(empty_pred, empty_target)
    assert res_both_empty["dice"] == 1.0
    assert res_both_empty["iou"] == 1.0
    assert res_both_empty["pred_voxel_count"] == 0
    assert res_both_empty["target_voxel_count"] == 0

    # 2. Target has lesion, pred is empty -> False negative -> Dice = 0.0
    non_empty = np.zeros((32, 32, 32), dtype=np.uint8)
    non_empty[5:15, 5:15, 5:15] = 1
    res_pred_empty = compute_binary_metrics(empty_pred, non_empty)
    assert res_pred_empty["dice"] == 0.0
    assert res_pred_empty["iou"] == 0.0
    assert res_pred_empty["sensitivity"] == 0.0

    # 3. Pred has lesion, target is empty -> False positive -> Dice = 0.0
    res_target_empty = compute_binary_metrics(non_empty, empty_target)
    assert res_target_empty["dice"] == 0.0
    assert res_target_empty["iou"] == 0.0


def test_subregion_hierarchy():
    # Construct synthetic ground truth
    # 1: Necrotic (100 voxels), 2: Edema (300 voxels), 3: Enhancing (200 voxels)
    gt = np.zeros((30, 30, 30), dtype=np.uint8)
    gt[5:10, 5:10, 5:9] = 1   # 5*5*4 = 100
    gt[10:20, 10:20, 5:8] = 2  # 10*10*3 = 300
    gt[5:15, 20:25, 5:9] = 3   # 10*5*4 = 200

    # Identical prediction
    pred = gt.copy()

    metrics = compute_case_subregion_metrics(pred, gt, voxel_spacing=(1.0, 1.0, 1.0))
    assert metrics["mean_dice"] == 1.0
    assert metrics["subregions"]["WT"]["dice"] == 1.0
    assert metrics["subregions"]["WT"]["target_voxel_count"] == 600  # 100 + 300 + 200
    assert metrics["subregions"]["TC"]["target_voxel_count"] == 300  # 100 + 200
    assert metrics["subregions"]["ET"]["target_voxel_count"] == 200  # 200


def test_aggregate_dataset_metrics():
    case1 = {
        "mean_dice": 0.85,
        "subregions": {
            "WT": {"dice": 0.90, "iou": 0.81},
            "TC": {"dice": 0.82, "iou": 0.70},
            "ET": {"dice": 0.83, "iou": 0.72}
        }
    }
    case2 = {
        "mean_dice": 0.75,
        "subregions": {
            "WT": {"dice": 0.80, "iou": 0.67},
            "TC": {"dice": 0.72, "iou": 0.56},
            "ET": {"dice": 0.73, "iou": 0.57}
        }
    }

    agg = aggregate_dataset_metrics([case1, case2])
    assert agg["total_cases_evaluated"] == 2
    assert agg["overall_mean_dice"] == 0.80
    assert agg["subregions"]["WT"]["dice_mean"] == 0.85
    assert agg["subregions"]["TC"]["dice_mean"] == 0.77
    assert agg["subregions"]["ET"]["dice_mean"] == 0.78
