"""
Generate research benchmark sample NIfTI datasets
"""
import sys
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from backend.segmentation.sample_data import create_benchmark_sample_suite

if __name__ == "__main__":
    target_dir = root_dir / "backend" / "segmentation" / "data"
    print(f"Generating benchmark samples in: {target_dir}")
    manifest = create_benchmark_sample_suite(target_dir)
    print(f"Successfully generated {len(manifest)} cases:")
    for k, v in manifest.items():
        print(f"  - [{k}] {v['title']} ({v['id']}): {v['mri_path']}")
