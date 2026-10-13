"""
NEUROSCAN AI — 3D Segmentation Evaluation and Validation Script
Evaluates 3D U-Net checkpoints on multi-case NIfTI test sets, reporting
per-case and aggregate Dice (WT, TC, ET), IoU, sensitivity, and volume agreement.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch

from backend.segmentation.model import load_segmentation_model
from backend.segmentation.pipeline import (
    load_nifti_file,
    preprocess_volume_for_model,
    run_sliding_window_inference
)
from backend.segmentation.metrics import (
    compute_case_subregion_metrics,
    aggregate_dataset_metrics
)


def evaluate_dataset(
    checkpoint_path: str,
    test_dir: str,
    output_json: str = None,
    device_name: str = "auto"
) -> Dict[str, Any]:
    """
    Runs evaluation across all cases in test_dir, strictly reporting measured results.
    """
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "auto" else device_name
    )

    model, is_loaded, meta = load_segmentation_model(checkpoint_path, device)
    if not is_loaded:
        raise RuntimeError(f"Cannot evaluate: model checkpoint not loaded from {checkpoint_path}")

    test_path = Path(test_dir)
    mri_files = list(test_path.glob("*_T1CE.nii*")) + list(test_path.glob("*_t1ce.nii*"))
    if not mri_files:
        mri_files = [p for p in test_path.glob("*.nii*") if "seg" not in p.name.lower()]

    case_results = []
    print(f"\n=======================================================")
    print(f"NEUROSCAN AI — Volumetric Evaluation Benchmark")
    print(f"Model Checkpoint: {checkpoint_path}")
    print(f"Device: {device} | Total Cases: {len(mri_files)}")
    print(f"=======================================================\n")

    for mri_p in mri_files:
        seg_name = mri_p.name.replace("_T1CE", "_seg").replace("_t1ce", "_seg")
        seg_p = mri_p.parent / seg_name
        if not seg_p.exists():
            print(f"[SKIP] Ground truth mask not found for {mri_p.name}")
            continue

        # Load volumes
        mri_vol = load_nifti_file(mri_p)
        gt_vol = load_nifti_file(seg_p)

        # Inference
        input_tensor = preprocess_volume_for_model(mri_vol, target_channels=4)
        pred_mask = run_sliding_window_inference(
            model=model,
            inputs=input_tensor,
            roi_size=(64, 64, 64),
            overlap=0.5,
            device=device
        )

        gt_data = gt_vol.data[0] if gt_vol.data.ndim == 4 else gt_vol.data
        case_metric = compute_case_subregion_metrics(
            pred_mask=pred_mask,
            target_mask=gt_data,
            voxel_spacing=mri_vol.voxel_spacing
        )

        case_summary = {
            "case_id": mri_p.stem,
            "mean_dice": case_metric["mean_dice"],
            "wt_dice": case_metric["subregions"]["WT"]["dice"],
            "tc_dice": case_metric["subregions"]["TC"]["dice"],
            "et_dice": case_metric["subregions"]["ET"]["dice"],
            "subregions": case_metric["subregions"]
        }
        case_results.append(case_summary)

        print(f"Case {mri_p.stem:<24} | Mean Dice: {case_summary['mean_dice']:.4f} | "
              f"WT: {case_summary['wt_dice']:.4f} | TC: {case_summary['tc_dice']:.4f} | ET: {case_summary['et_dice']:.4f}")

    if not case_results:
        print("[WARNING] No test cases evaluated.")
        return {"error": "No valid evaluated cases"}

    aggregated = aggregate_dataset_metrics(case_results)

    print(f"\n-------------------------------------------------------")
    print(f"AGGREGATE VALIDATION SUMMARY ({len(case_results)} Cases)")
    print(f"-------------------------------------------------------")
    print(f"Overall Mean Dice: {aggregated['overall_mean_dice']:.4f}")
    for sub in ["WT", "TC", "ET"]:
        sub_info = aggregated["subregions"][sub]
        print(f"  {sub} — Mean Dice: {sub_info['dice_mean']:.4f} (±{sub_info['dice_std']:.4f}) | Median: {sub_info['dice_median']:.4f} | IoU: {sub_info['iou_mean']:.4f}")
    print(f"=======================================================\n")

    report = {
        "checkpoint": checkpoint_path,
        "total_cases": len(case_results),
        "aggregate_metrics": aggregated,
        "cases": case_results
    }

    if output_json:
        out_p = Path(output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[INFO] Saved evaluation report to {output_json}")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NeuroScan 3D Tumor Segmentation Model")
    parser.add_argument("--checkpoint", type=str, default="models/NeuroScan_Seg3D_UNet.pth")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to folder containing test NIfTI volumes")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    evaluate_dataset(args.checkpoint, args.test_dir, args.output_json)
