"""
NEUROSCAN AI — NIfTI Processing, Spatial Alignment & Inference Pipeline
Handles .nii/.nii.gz volumes, non-zero intensity normalization,
patch-based sliding window inference, and multi-planar slice generation.
"""

import os
import io
import base64
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List, Union
import numpy as np
import torch
from PIL import Image

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    from monai.inferers import sliding_window_inference as monai_sw_inference
    MONAI_INFERER_AVAILABLE = True
except ImportError:
    MONAI_INFERER_AVAILABLE = False


class NiftiVolume:
    """Encapsulates a loaded NIfTI 3D/4D volume with spatial geometry and sanitized headers."""
    def __init__(
        self,
        data: np.ndarray,
        affine: np.ndarray,
        voxel_spacing: Tuple[float, float, float],
        shape: Tuple[int, ...],
        modality_channels: int
    ):
        self.data = data  # Shape (C, H, W, D)
        self.affine = affine
        self.voxel_spacing = voxel_spacing  # (dx, dy, dz) in mm
        self.shape = shape
        self.modality_channels = modality_channels


def load_nifti_file(file_path: Union[str, Path]) -> NiftiVolume:
    """
    Safely loads a .nii or .nii.gz file, extracting spatial affine,
    voxel zooms, and channel structure while anonymizing patient metadata.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"NIfTI file not found: {path}")

    if not (path.name.endswith(".nii") or path.name.endswith(".nii.gz")):
        raise ValueError(f"Unsupported file format. Must be .nii or .nii.gz: {path.name}")

    if not NIBABEL_AVAILABLE:
        raise RuntimeError("nibabel is required to load NIfTI volumes. Please ensure it is installed.")

    img = nib.load(str(path))
    # Conform volume to canonical RAS+ spatial coordinate system
    img = nib.as_closest_canonical(img)
    # Conform data to float32
    raw_data = np.asarray(img.dataobj, dtype=np.float32)
    affine = np.asarray(img.affine, dtype=np.float32)

    # Voxel spacing from affine or zooms
    try:
        zooms = img.header.get_zooms()
        dx = float(zooms[0]) if len(zooms) > 0 else 1.0
        dy = float(zooms[1]) if len(zooms) > 1 else 1.0
        dz = float(zooms[2]) if len(zooms) > 2 else 1.0
    except Exception:
        # Compute from affine
        dx = float(np.sqrt(np.sum(affine[:3, 0] ** 2)))
        dy = float(np.sqrt(np.sum(affine[:3, 1] ** 2)))
        dz = float(np.sqrt(np.sum(affine[:3, 2] ** 2)))

    voxel_spacing = (round(dx, 3), round(dy, 3), round(dz, 3))

    # Standardize shape to (C, H, W, D)
    if raw_data.ndim == 3:
        # Single modality (e.g. T1ce or FLAIR)
        data = raw_data[np.newaxis, ...]
        modality_channels = 1
    elif raw_data.ndim == 4:
        # Could be (H, W, D, C) or (C, H, W, D)
        if raw_data.shape[3] in [1, 2, 3, 4] and raw_data.shape[0] > 10:
            # Transpose (H, W, D, C) -> (C, H, W, D)
            data = np.transpose(raw_data, (3, 0, 1, 2))
        else:
            data = raw_data
        modality_channels = data.shape[0]
    else:
        raise ValueError(f"Invalid volume dimensions: {raw_data.ndim}D. Expected 3D or 4D NIfTI.")

    return NiftiVolume(
        data=data,
        affine=affine,
        voxel_spacing=voxel_spacing,
        shape=data.shape,
        modality_channels=modality_channels
    )


def normalize_intensity(channel_data: np.ndarray) -> np.ndarray:
    """
    Z-score intensity normalization restricted to non-zero intracranial voxels.
    Prevents background zero-padding from skewing brain tissue contrast.
    """
    normalized = np.zeros_like(channel_data, dtype=np.float32)
    non_zero_mask = channel_data > 0

    if np.any(non_zero_mask):
        mean = float(np.mean(channel_data[non_zero_mask]))
        std = float(np.std(channel_data[non_zero_mask]))
        if std > 1e-5:
            normalized[non_zero_mask] = (channel_data[non_zero_mask] - mean) / std
        else:
            normalized[non_zero_mask] = channel_data[non_zero_mask] - mean
    return normalized


def preprocess_volume_for_model(
    nifti_vol: NiftiVolume,
    target_channels: int = 4
) -> torch.Tensor:
    """
    Applies per-channel normalization and conforms channels to target_channels.
    If input has 1 channel and target is 4, replicates across modalities.
    Returns torch.Tensor of shape (1, C, H, W, D).
    """
    channels_data = []
    for c in range(nifti_vol.modality_channels):
        norm_c = normalize_intensity(nifti_vol.data[c])
        channels_data.append(norm_c)

    # Replicate or slice to match target_channels
    if len(channels_data) == 1 and target_channels == 4:
        processed_data = np.repeat(channels_data[0][np.newaxis, ...], 4, axis=0)
    elif len(channels_data) >= target_channels:
        processed_data = np.stack(channels_data[:target_channels], axis=0)
    else:
        # Pad remaining channels with zeros
        padded = np.zeros((target_channels, *nifti_vol.data.shape[1:]), dtype=np.float32)
        for i, c_data in enumerate(channels_data):
            padded[i] = c_data
        processed_data = padded

    tensor = torch.from_numpy(processed_data).float().unsqueeze(0)  # (1, C, H, W, D)
    return tensor


def native_sliding_window_inference(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    roi_size: Tuple[int, int, int] = (64, 64, 64),
    overlap: float = 0.5,
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Native sliding-window 3D patch inference with overlap blending.
    Allows large 3D NIfTI inference on constrained memory hardware without CUDA OOM.
    """
    b, c, h, w, d = inputs.shape
    rh, rw, rd = roi_size

    # Ensure inputs fit roi_size with symmetric padding if volume is smaller
    pad_h = max(0, rh - h)
    pad_w = max(0, rw - w)
    pad_d = max(0, rd - d)
    if pad_h > 0 or pad_w > 0 or pad_d > 0:
        inputs = torch.nn.functional.pad(
            inputs,
            (0, pad_d, 0, pad_w, 0, pad_h),
            mode="constant",
            value=0
        )
        _, _, h, w, d = inputs.shape

    step_h = max(1, int(rh * (1 - overlap)))
    step_w = max(1, int(rw * (1 - overlap)))
    step_d = max(1, int(rd * (1 - overlap)))

    h_starts = list(range(0, max(1, h - rh + 1), step_h))
    if h_starts[-1] + rh < h:
        h_starts.append(h - rh)

    w_starts = list(range(0, max(1, w - rw + 1), step_w))
    if w_starts[-1] + rw < w:
        w_starts.append(w - rw)

    d_starts = list(range(0, max(1, d - rd + 1), step_d))
    if d_starts[-1] + rd < d:
        d_starts.append(d - rd)

    # Accumulator tensors
    out_channels = getattr(model, "out_channels", 4)
    output_accum = torch.zeros((b, out_channels, h, w, d), dtype=torch.float32, device=device)
    count_accum = torch.zeros((b, 1, h, w, d), dtype=torch.float32, device=device)

    # Cosine/Hanning patch weight window to prevent seam artifacts
    window_h = torch.hann_window(rh, periodic=False).view(rh, 1, 1)
    window_w = torch.hann_window(rw, periodic=False).view(1, rw, 1)
    window_d = torch.hann_window(rd, periodic=False).view(1, 1, rd)
    patch_weight = (window_h * window_w * window_d).to(device)
    patch_weight = torch.clamp(patch_weight, min=1e-3).unsqueeze(0).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        for hs in h_starts:
            for ws in w_starts:
                for ds in d_starts:
                    patch = inputs[:, :, hs:hs+rh, ws:ws+rw, ds:ds+rd].to(device)
                    # Forward pass
                    with torch.amp.autocast(device_type=device.type if device.type in ["cuda", "cpu"] else "cpu", enabled=(device.type == "cuda")):
                        logits = model(patch)
                    # Accumulate
                    output_accum[:, :, hs:hs+rh, ws:ws+rw, ds:ds+rd] += logits * patch_weight
                    count_accum[:, :, hs:hs+rh, ws:ws+rw, ds:ds+rd] += patch_weight

    output = output_accum / torch.clamp(count_accum, min=1e-5)
    # Crop back if padded
    orig_h = h - pad_h
    orig_w = w - pad_w
    orig_d = d - pad_d
    return output[:, :, :orig_h, :orig_w, :orig_d]


def run_sliding_window_inference(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    roi_size: Tuple[int, int, int] = (64, 64, 64),
    overlap: float = 0.5,
    device: torch.device = torch.device("cpu")
) -> np.ndarray:
    """
    Executes sliding window inference using MONAI when available or native patch engine.
    Returns argmax discrete mask numpy array of shape (H, W, D).
    """
    model.eval()
    inputs = inputs.to(device)

    if MONAI_INFERER_AVAILABLE:
        try:
            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")):
                    logits = monai_sw_inference(
                        inputs=inputs,
                        roi_size=roi_size,
                        sw_batch_size=2,
                        predictor=model,
                        overlap=overlap,
                        mode="gaussian"
                    )
        except Exception:
            logits = native_sliding_window_inference(model, inputs, roi_size, overlap, device)
    else:
        logits = native_sliding_window_inference(model, inputs, roi_size, overlap, device)

    pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    return pred_mask


def extract_slice_rgba(
    mri_slice: np.ndarray,
    mask_slice: Optional[np.ndarray] = None,
    color_map: Optional[Dict[int, List[int]]] = None
) -> Tuple[str, Optional[str]]:
    """
    Converts a single 2D MRI slice (and optional segmentation mask)
    into base64 encoded PNG data URLs for fast frontend canvas rendering.
    """
    # Robust percentile-based intensity normalization for authentic clinical MRI contrast
    non_zero = mri_slice[mri_slice > 0]
    if len(non_zero) > 10:
        p_min = float(np.percentile(non_zero, 1.0))
        p_max = float(np.percentile(non_zero, 99.5))
        if p_max - p_min > 1e-3:
            scaled = np.clip((mri_slice - p_min) / (p_max - p_min) * 255.0, 0, 255).astype(np.uint8)
        else:
            scaled = np.zeros_like(mri_slice, dtype=np.uint8)
    else:
        p_min = float(np.min(mri_slice))
        p_max = float(np.max(mri_slice))
        if p_max - p_min > 1e-4:
            scaled = np.clip((mri_slice - p_min) / (p_max - p_min) * 255.0, 0, 255).astype(np.uint8)
        else:
            scaled = np.zeros_like(mri_slice, dtype=np.uint8)

    img_pil = Image.fromarray(scaled, mode="L").convert("RGB")
    buf_mri = io.BytesIO()
    img_pil.save(buf_mri, format="PNG")
    mri_b64 = "data:image/png;base64," + base64.b64encode(buf_mri.getvalue()).decode("utf-8")

    mask_b64 = None
    if mask_slice is not None:
        h, w = mask_slice.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        default_colors = {
            1: [255, 69, 58, 210],    # NCR (Red)
            2: [57, 213, 255, 190],   # ED (Cyan)
            3: [255, 214, 10, 225],   # ET (Yellow)
            4: [255, 214, 10, 225]    # ET alternative
        }
        cmap = color_map or default_colors
        for label, rgba_val in cmap.items():
            locs = (mask_slice == label)
            if np.any(locs):
                rgba[locs] = rgba_val

        mask_pil = Image.fromarray(rgba, mode="RGBA")
        buf_mask = io.BytesIO()
        mask_pil.save(buf_mask, format="PNG")
        mask_b64 = "data:image/png;base64," + base64.b64encode(buf_mask.getvalue()).decode("utf-8")

    return mri_b64, mask_b64


def generate_volume_slice_pack(
    mri_vol: np.ndarray,
    pred_mask: Optional[np.ndarray] = None,
    gt_mask: Optional[np.ndarray] = None,
    plane: str = "axial"  # "axial" (Z), "coronal" (Y), "sagittal" (X)
) -> Dict[str, Any]:
    """
    Generates structured slice metadata and image representations along specified plane
    under canonical RAS+ radiological coordinate convention.
    """
    # Assuming mri_vol shape (X, Y, Z)
    if plane == "axial":
        total_slices = mri_vol.shape[2]
        get_mri_slice = lambda idx: np.rot90(mri_vol[:, :, idx])
        get_pred_slice = (lambda idx: np.rot90(pred_mask[:, :, idx])) if pred_mask is not None else None
        get_gt_slice = (lambda idx: np.rot90(gt_mask[:, :, idx])) if gt_mask is not None else None
    elif plane == "coronal":
        total_slices = mri_vol.shape[1]
        get_mri_slice = lambda idx: np.rot90(mri_vol[:, idx, :])
        get_pred_slice = (lambda idx: np.rot90(pred_mask[:, idx, :])) if pred_mask is not None else None
        get_gt_slice = (lambda idx: np.rot90(gt_mask[:, idx, :])) if gt_mask is not None else None
    else:  # sagittal
        total_slices = mri_vol.shape[0]
        get_mri_slice = lambda idx: np.rot90(mri_vol[idx, :, :])
        get_pred_slice = (lambda idx: np.rot90(pred_mask[idx, :, :])) if pred_mask is not None else None
        get_gt_slice = (lambda idx: np.rot90(gt_mask[idx, :, :])) if gt_mask is not None else None

    # Identify slice with largest tumor cross-section as initial focus slice
    focus_slice = total_slices // 2
    max_tumor_pixels = 0
    active_eval_slice = get_gt_slice if get_gt_slice is not None else get_pred_slice
    if active_eval_slice is not None:
        for idx in range(total_slices):
            slice_m = active_eval_slice(idx)
            count = int(np.sum(slice_m > 0))
            if count > max_tumor_pixels:
                max_tumor_pixels = count
                focus_slice = idx

    return {
        "plane": plane,
        "total_slices": total_slices,
        "initial_focus_slice": focus_slice,
        "has_tumor_in_plane": max_tumor_pixels > 0
    }
