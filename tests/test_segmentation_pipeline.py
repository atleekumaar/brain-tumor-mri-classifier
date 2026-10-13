"""
Unit tests for NIfTI loading, intensity normalization and sliding window inference.
"""

from pathlib import Path
import numpy as np
import torch
import pytest

from backend.segmentation.pipeline import (
    load_nifti_file,
    normalize_intensity,
    preprocess_volume_for_model,
    run_sliding_window_inference,
    extract_slice_rgba
)
from backend.segmentation.model import build_segmentation_model


@pytest.fixture
def sample_data_dir():
    return Path(__file__).resolve().parent.parent / "backend" / "segmentation" / "data"


def test_nifti_loading_and_dimensions(sample_data_dir):
    mri_file = sample_data_dir / "BraTS_Case_001_T1CE.nii.gz"
    assert mri_file.exists(), "Sample file BraTS_Case_001_T1CE.nii.gz should exist"

    vol = load_nifti_file(mri_file)
    assert vol.data.ndim == 4, "Data must be standardized to 4D (C, H, W, D)"
    assert vol.shape == (1, 240, 240, 155)
    assert len(vol.voxel_spacing) == 3
    assert vol.voxel_spacing == (1.0, 1.0, 1.0)
    assert vol.affine.shape == (4, 4)


def test_intensity_normalization():
    # Synthetic array with zeros and active tissue
    arr = np.zeros((20, 20, 20), dtype=np.float32)
    arr[5:15, 5:15, 5:15] = 100.0 + np.random.randn(10, 10, 10).astype(np.float32) * 10.0

    normalized = normalize_intensity(arr)
    # Background must remain exactly 0
    assert np.all(normalized[0, 0, :] == 0.0)
    # Foreground must have approximately zero mean and unit variance
    fg = normalized[5:15, 5:15, 5:15]
    assert abs(np.mean(fg)) < 1e-4
    assert abs(np.std(fg) - 1.0) < 0.1


def test_preprocess_volume_tensor_shape(sample_data_dir):
    mri_file = sample_data_dir / "BraTS_Case_001_T1CE.nii.gz"
    vol = load_nifti_file(mri_file)

    tensor = preprocess_volume_for_model(vol, target_channels=4)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (1, 4, 240, 240, 155)
    assert tensor.dtype == torch.float32


def test_sliding_window_inference_output(sample_data_dir):
    mri_file = sample_data_dir / "BraTS_Case_001_T1CE.nii.gz"
    vol = load_nifti_file(mri_file)
    tensor = preprocess_volume_for_model(vol, target_channels=4)

    model = build_segmentation_model(in_channels=4, out_channels=4)
    model.eval()

    device = torch.device("cpu")
    pred_mask = run_sliding_window_inference(
        model=model,
        inputs=tensor,
        roi_size=(48, 48, 48),
        overlap=0.25,
        device=device
    )

    assert isinstance(pred_mask, np.ndarray)
    assert pred_mask.shape == (240, 240, 155)
    # Predicted classes must be in range [0, 3]
    unique_labels = np.unique(pred_mask)
    for lbl in unique_labels:
        assert lbl in [0, 1, 2, 3]


def test_extract_slice_rgba():
    mri_slice = np.random.rand(64, 64).astype(np.float32) * 500.0
    mask_slice = np.zeros((64, 64), dtype=np.uint8)
    mask_slice[20:30, 20:30] = 3  # Enhancing tumor

    mri_b64, mask_b64 = extract_slice_rgba(mri_slice, mask_slice)
    assert mri_b64.startswith("data:image/png;base64,")
    assert mask_b64.startswith("data:image/png;base64,")
