"""
NEUROSCAN AI — 3D Brain Tumor Segmentation Training Script
Supports BraTS NIfTI datasets with patient-level deterministic splitting,
Dice + Cross-Entropy loss, mixed-precision (AMP), and checkpoint persistence.
"""

import os
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from backend.segmentation.model import build_segmentation_model
from backend.segmentation.pipeline import load_nifti_file, normalize_intensity
from backend.segmentation.metrics import compute_case_subregion_metrics

try:
    from monai.losses import DiceCELoss
    MONAI_LOSS_AVAILABLE = True
except ImportError:
    MONAI_LOSS_AVAILABLE = False


class NativeDiceCELoss(nn.Module):
    """Fallback Dice + Cross-Entropy loss for multi-class 3D segmentation"""
    def __init__(self, num_classes: int = 4, smooth: float = 1e-5):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C, H, W, D), targets: (B, H, W, D) long
        ce_loss = self.ce(logits, targets)

        # One-hot encode targets for Dice
        probs = torch.softmax(logits, dim=1)
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes)
        targets_onehot = targets_onehot.permute(0, 4, 1, 2, 3).float()

        dims = (2, 3, 4)
        intersection = torch.sum(probs * targets_onehot, dim=dims)
        cardinality = torch.sum(probs + targets_onehot, dim=dims)
        dice_score = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        # Average across classes 1..C-1 (foreground)
        dice_loss = 1.0 - torch.mean(dice_score[:, 1:])

        return 0.5 * ce_loss + 0.5 * dice_loss


class NiftiSegmentationDataset(Dataset):
    """
    Patient-level dataset for 3D MRI volumes and ground-truth segmentation masks.
    Extracts random or centered patches during training.
    """
    def __init__(
        self,
        case_pairs: List[Tuple[Path, Path]],
        patch_size: Tuple[int, int, int] = (64, 64, 64),
        is_train: bool = True
    ):
        self.case_pairs = case_pairs
        self.patch_size = patch_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.case_pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        mri_path, seg_path = self.case_pairs[idx]
        mri_vol = load_nifti_file(mri_path)
        seg_vol = load_nifti_file(seg_path)

        # Normalize MRI per channel
        channels = []
        for c in range(mri_vol.modality_channels):
            channels.append(normalize_intensity(mri_vol.data[c]))

        if len(channels) == 1:
            mri_data = np.repeat(channels[0][np.newaxis, ...], 4, axis=0)
        else:
            mri_data = np.stack(channels[:4], axis=0)

        # Clean segmentation mask labels
        seg_data = seg_vol.data[0] if seg_vol.data.ndim == 4 else seg_vol.data
        seg_data = np.where(seg_data == 4, 3, seg_data).astype(np.int64)

        # Crop patch
        c, h, w, d = mri_data.shape
        ph, pw, pd = self.patch_size

        if self.is_train and h > ph and w > pw and d > pd:
            # If training, sample with 50% probability around tumor foreground
            tumor_indices = np.argwhere(seg_data > 0)
            if len(tumor_indices) > 0 and np.random.rand() > 0.4:
                center = tumor_indices[np.random.choice(len(tumor_indices))]
                sh = max(0, min(h - ph, center[0] - ph // 2))
                sw = max(0, min(w - pw, center[1] - pw // 2))
                sd = max(0, min(d - pd, center[2] - pd // 2))
            else:
                sh = np.random.randint(0, h - ph + 1)
                sw = np.random.randint(0, w - pw + 1)
                sd = np.random.randint(0, d - pd + 1)
        else:
            sh = max(0, (h - ph) // 2)
            sw = max(0, (w - pw) // 2)
            sd = max(0, (d - pd) // 2)

        mri_patch = mri_data[:, sh:sh+ph, sw:sw+pw, sd:sd+pd]
        seg_patch = seg_data[sh:sh+ph, sw:sw+pw, sd:sd+pd]

        # Pad if volume smaller than patch size
        if mri_patch.shape[1:] != (ph, pw, pd):
            pad_h = ph - mri_patch.shape[1]
            pad_w = pw - mri_patch.shape[2]
            pad_d = pd - mri_patch.shape[3]
            mri_patch = np.pad(mri_patch, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)))
            seg_patch = np.pad(seg_patch, ((0, pad_h), (0, pad_w), (0, pad_d)))

        return {
            "image": torch.from_numpy(mri_patch).float(),
            "mask": torch.from_numpy(seg_patch).long(),
            "patient_id": mri_path.stem
        }


def partition_cases_by_patient(
    data_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[Tuple[Path, Path]], List[Tuple[Path, Path]]]:
    """
    Scans directory for NIfTI volume pairs and deterministically partitions
    by patient identifier to strictly prevent data leakage.
    """
    # Look for files ending with _t1ce.nii.gz / _seg.nii.gz or matching prefixes
    files = list(data_dir.glob("*_T1CE.nii*")) + list(data_dir.glob("*_t1ce.nii*"))
    pairs = []

    for mri_p in files:
        # Check corresponding seg file
        name = mri_p.name
        seg_name = name.replace("_T1CE", "_seg").replace("_t1ce", "_seg")
        seg_p = mri_p.parent / seg_name
        if seg_p.exists():
            pairs.append((mri_p, seg_p))

    if not pairs:
        # Fallback: check any .nii.gz and *seg.nii.gz
        all_nii = list(data_dir.glob("*.nii*"))
        segs = [p for p in all_nii if "seg" in p.name.lower()]
        for seg_p in segs:
            prefix = seg_p.name.lower().replace("_seg", "").replace(".seg", "").split(".")[0]
            for candidate in all_nii:
                if candidate != seg_p and prefix in candidate.name.lower():
                    pairs.append((candidate, seg_p))
                    break

    # Deterministic patient split
    rng = np.random.RandomState(seed)
    indices = np.arange(len(pairs))
    rng.shuffle(indices)

    n_val = max(1, int(len(pairs) * val_ratio)) if len(pairs) > 1 else 0
    val_idx = set(indices[:n_val])

    train_pairs = [pairs[i] for i in range(len(pairs)) if i not in val_idx]
    val_pairs = [pairs[i] for i in range(len(pairs)) if i in val_idx]

    return train_pairs, val_pairs


def train_segmentation_model(
    data_dir: str,
    output_checkpoint_path: str,
    epochs: int = 20,
    batch_size: int = 2,
    lr: float = 1e-4,
    patch_size: Tuple[int, int, int] = (64, 64, 64),
    device_name: str = "auto"
) -> Dict[str, Any]:
    """
    Executes supervised training of 3D U-Net using ground-truth masks.
    Saves trained checkpoint with weights and validation metadata.
    """
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "auto" else device_name
    )

    data_path = Path(data_dir)
    train_pairs, val_pairs = partition_cases_by_patient(data_path)

    if len(train_pairs) == 0:
        raise ValueError(f"No valid MRI/segmentation pairs found in {data_dir}")

    train_ds = NiftiSegmentationDataset(train_pairs, patch_size=patch_size, is_train=True)
    val_ds = NiftiSegmentationDataset(val_pairs or train_pairs, patch_size=patch_size, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    model = build_segmentation_model(in_channels=4, out_channels=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    if MONAI_LOSS_AVAILABLE:
        loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    else:
        loss_fn = NativeDiceCELoss(num_classes=4)

    best_val_dice = -1.0
    history = []

    print(f"[TRAIN] Starting 3D U-Net training on {len(train_pairs)} cases (Val: {len(val_pairs)}) on {device}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        start_t = time.perf_counter()

        for batch in train_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type if device.type in ["cuda", "cpu"] else "cpu", enabled=(device.type == "cuda")):
                logits = model(imgs)
                if MONAI_LOSS_AVAILABLE:
                    loss = loss_fn(logits, masks.unsqueeze(1))
                else:
                    loss = loss_fn(logits, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(train_loader))

        # Validation step
        model.eval()
        val_dices = []
        with torch.no_grad():
            for v_batch in val_loader:
                v_img = v_batch["image"].to(device)
                v_mask = v_batch["mask"].cpu().numpy()[0]
                with torch.amp.autocast(device_type=device.type if device.type in ["cuda", "cpu"] else "cpu", enabled=(device.type == "cuda")):
                    v_logits = model(v_img)
                v_pred = torch.argmax(v_logits, dim=1).cpu().numpy()[0]
                case_m = compute_case_subregion_metrics(v_pred, v_mask)
                val_dices.append(case_m["mean_dice"])

        val_mean_dice = float(np.mean(val_dices)) if val_dices else 0.0
        epoch_dur = round(time.perf_counter() - start_t, 2)
        history.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_mean_dice": round(val_mean_dice, 4),
            "duration_sec": epoch_dur
        })
        print(f"[Epoch {epoch:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | Val Dice: {val_mean_dice:.4f} ({epoch_dur}s)")

        # Save best model
        if val_mean_dice > best_val_dice or epoch == epochs:
            best_val_dice = val_mean_dice
            out_p = Path(output_checkpoint_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "metadata": {
                    "architecture": "3D U-Net (BraTS Benchmark)",
                    "epochs_trained": epoch,
                    "final_val_dice": round(best_val_dice, 4),
                    "patch_size": list(patch_size),
                    "in_channels": 4,
                    "out_channels": 4,
                    "date": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            }, str(out_p))

    return {
        "status": "success",
        "best_val_dice": round(best_val_dice, 4),
        "total_epochs": epochs,
        "checkpoint_path": str(output_checkpoint_path),
        "history": history
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NeuroScan 3D Brain Tumor Segmentation Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing NIfTI volume pairs")
    parser.add_argument("--output", type=str, default="models/NeuroScan_Seg3D_UNet.pth", help="Checkpoint output path")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Patch batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    train_segmentation_model(args.data_dir, args.output, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
