"""
Train baseline 3D U-Net checkpoint on benchmark NIfTI volumes with real ground-truth masks
"""
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from backend.segmentation.train import train_segmentation_model

if __name__ == "__main__":
    data_dir = root_dir / "backend" / "segmentation" / "data"
    output_ckpt = root_dir / "models" / "NeuroScan_Seg3D_UNet.pth"
    print(f"Training 3D U-Net on {data_dir} -> {output_ckpt}")
    result = train_segmentation_model(
        data_dir=str(data_dir),
        output_checkpoint_path=str(output_ckpt),
        epochs=4,
        batch_size=1,
        lr=5e-4,
        patch_size=(64, 64, 64)
    )
    print("Training finished with result:", result)
