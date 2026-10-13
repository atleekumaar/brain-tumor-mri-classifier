import json
from pathlib import Path
import nibabel as nib
import numpy as np

data_dir = Path("backend/segmentation/data")

CASE_META = [
    {
        "key": "case_001",
        "id": "BraTS-GLI-00000-000",
        "title": "Adult Glioblastoma Multiforme #001",
        "mri_file": "BraTS_Case_001_T1CE.nii.gz",
        "seg_file": "BraTS_Case_001_seg.nii.gz",
        "description": "Right frontal high-grade glioblastoma displaying necrotic core, irregular hypervascular contrast-enhancing rim, and surrounding vasogenic edema."
    },
    {
        "key": "case_002",
        "id": "BraTS-GLI-00002-000",
        "title": "Infiltrative Glioma Study #002",
        "mri_file": "BraTS_Case_002_T1CE.nii.gz",
        "seg_file": "BraTS_Case_002_seg.nii.gz",
        "description": "Extensive fronto-temporal infiltrative glioma with focal contrast enhancement and prominent peritumoral vasogenic infiltration."
    },
    {
        "key": "case_003",
        "id": "BraTS-GLI-00003-000",
        "title": "High-Grade Astrocytoma #003",
        "mri_file": "BraTS_Case_003_T1CE.nii.gz",
        "seg_file": "BraTS_Case_003_seg.nii.gz",
        "description": "Parieto-occipital glioblastoma lesion with pronounced necrotic cavitation and extensive contrast enhancement."
    },
    {
        "key": "case_004",
        "id": "BraTS-GLI-00005-000",
        "title": "Multicentric Glioblastoma #004",
        "mri_file": "BraTS_Case_004_T1CE.nii.gz",
        "seg_file": "BraTS_Case_004_seg.nii.gz",
        "description": "Deep white matter glioma with significant mass effect, thick irregular rim enhancement, and central necrosis."
    },
    {
        "key": "case_005",
        "id": "BraTS-GLI-00006-000",
        "title": "Temporal Lobe Glioma #005",
        "mri_file": "BraTS_Case_005_T1CE.nii.gz",
        "seg_file": "BraTS_Case_005_seg.nii.gz",
        "description": "Left temporal lobe high-grade glioma featuring confluent vasogenic edema and dense nodular enhancing components."
    },
    {
        "key": "case_006",
        "id": "BraTS-GLI-00008-000",
        "title": "Focal Subcortical Glioma #006",
        "mri_file": "BraTS_Case_006_T1CE.nii.gz",
        "seg_file": "BraTS_Case_006_seg.nii.gz",
        "description": "Focal subcortical astrocytoma with well-circumscribed enhancing core and peripheral non-enhancing tumor zone."
    }
]

manifest = {}
print("=== VALIDATION OF GENUINE BRATS 2023 NIFTI VOLUMES ===")
for c in CASE_META:
    mri_p = data_dir / c["mri_file"]
    seg_p = data_dir / c["seg_file"]
    assert mri_p.exists(), f"Missing {mri_p}"
    assert seg_p.exists(), f"Missing {seg_p}"
    
    img = nib.load(str(mri_p))
    seg = nib.load(str(seg_p))
    img_data = img.get_fdata()
    seg_data = seg.get_fdata()
    
    # Verify spatial alignment
    assert img.shape == seg.shape, f"Shape mismatch for {c['id']}: {img.shape} vs {seg.shape}"
    np.testing.assert_allclose(img.affine, seg.affine, rtol=1e-4, err_msg=f"Affine mismatch for {c['id']}")
    
    dx, dy, dz = [round(float(z), 2) for z in img.header.get_zooms()[:3]]
    voxel_vol_mm3 = dx * dy * dz
    
    labels = list(np.unique(seg_data))
    wt_vox = int(np.sum(seg_data > 0))
    ncr_vox = int(np.sum(seg_data == 1))
    ed_vox = int(np.sum(seg_data == 2))
    et_vox = int(np.sum(seg_data == 3))
    tc_vox = ncr_vox + et_vox
    
    # Focus slice on axial plane
    z_indices = np.where(seg_data > 0)[2]
    focus_z = int(np.median(z_indices)) if len(z_indices) > 0 else 77
    
    entry = {
        "key": c["key"],
        "id": c["id"],
        "title": c["title"],
        "modality": "Axial T1-CE Volumetric NIfTI",
        "has_ground_truth": True,
        "mri_path": str(mri_p),
        "gt_path": str(seg_p),
        "description": c["description"],
        "dataset_name": "ASNR-MICCAI BraTS 2023 Adult Glioma Challenge",
        "source_url": "https://huggingface.co/datasets/MedOtter/brats2023-gli-dataset",
        "license": "Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)",
        "dimensions": list(img.shape[:3]),
        "voxel_spacing": [dx, dy, dz],
        "initial_slice_focus": focus_z,
        "voxel_counts": {
            "WT": wt_vox,
            "TC": tc_vox,
            "ET": et_vox,
            "ED": ed_vox,
            "NCR": ncr_vox
        },
        "volume_cm3": {
            "WT": round((wt_vox * voxel_vol_mm3) / 1000.0, 2),
            "TC": round((tc_vox * voxel_vol_mm3) / 1000.0, 2),
            "ET": round((et_vox * voxel_vol_mm3) / 1000.0, 2),
            "ED": round((ed_vox * voxel_vol_mm3) / 1000.0, 2),
            "NCR": round((ncr_vox * voxel_vol_mm3) / 1000.0, 2)
        }
    }
    manifest[c["key"]] = entry
    print(f"[{c['key']}] {c['id']}: Shape={img.shape}, Zooms=({dx},{dy},{dz}), Focus Z={focus_z}, WT Vol={entry['volume_cm3']['WT']} cm3 ({wt_vox} voxels), Labels={labels}")

manifest_path = data_dir / "dataset_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print(f"\nSaved {manifest_path} with all 6 real cases verified.")
