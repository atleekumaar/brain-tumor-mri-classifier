"""
NEUROSCAN AI — 3D Surface Mesh Generator for Actual Voxel Segmentations
Extracts true volumetric isosurfaces from segmentation masks using Marching Cubes
or boundary quad-triangulation, scaled by physical voxel spacing (dx, dy, dz).
"""

from typing import Dict, Any, List, Tuple, Optional
import numpy as np

try:
    from skimage.measure import marching_cubes
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def extract_surface_mesh_from_mask(
    mask_3d: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    max_triangles: int = 25000,
    smoothing_iterations: int = 1
) -> Optional[Dict[str, Any]]:
    """
    Extracts a true 3D polygonal surface mesh from a binary 3D segmentation mask
    using Marching Cubes. Scaled by physical voxel spacing (dx, dy, dz).
    Returns None if the mask is empty or contains insufficient foreground voxels (< 8).
    Never creates a point cloud or fabricated geometry.
    """
    binary_mask = (mask_3d > 0).astype(np.uint8)
    voxel_count = int(np.sum(binary_mask))
    if voxel_count < 8:
        return None

    dx, dy, dz = voxel_spacing

    if not SKIMAGE_AVAILABLE:
        return None

    try:
        # Determine appropriate step size to keep triangles within budget
        step = 1
        if voxel_count > 100000:
            step = 2

        # Pad volume by 1 voxel to ensure closed, watertight isosurface boundaries
        padded = np.pad(binary_mask, pad_width=1, mode="constant", constant_values=0)
        verts, faces, normals, _ = marching_cubes(
            padded,
            level=0.5,
            spacing=(dx, dy, dz),
            step_size=step
        )

        if len(verts) == 0 or len(faces) == 0:
            return None

        # Offset padding
        verts = np.asarray(verts, dtype=np.float32)
        verts[:, 0] -= dx
        verts[:, 1] -= dy
        verts[:, 2] -= dz

        # Center mesh around its geometric centroid for balanced 3D rotation
        centroid = np.mean(verts, axis=0)
        verts_centered = verts - centroid

        # Compute bounding box
        bbox = [
            round(float(np.max(verts[:, 0]) - np.min(verts[:, 0])), 2),
            round(float(np.max(verts[:, 1]) - np.min(verts[:, 1])), 2),
            round(float(np.max(verts[:, 2]) - np.min(verts[:, 2])), 2)
        ]

        # Convert to flat lists for Three.js BufferGeometry
        return {
            "vertices": np.round(verts_centered.flatten(), 2).tolist(),
            "faces": faces.flatten().tolist(),
            "normals": np.round(normals.flatten(), 3).tolist() if normals is not None else [],
            "vertex_count": len(verts),
            "face_count": len(faces),
            "centroid": np.round(centroid, 2).tolist(),
            "bbox_mm": bbox,
            "voxel_count": voxel_count,
            "algorithm": "marching_cubes"
        }
    except Exception as e:
        print(f"[WARNING] Marching cubes failed on mask: {e}")
        return None


def generate_subregion_meshes(
    mask_3d: np.ndarray,
    voxel_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)
) -> Dict[str, Any]:
    """
    Extracts separate true 3D surface meshes for Whole Tumor (WT),
    Tumor Core (TC), and Enhancing Tumor (ET) from the predicted volume.
    """
    clean_mask = np.where(mask_3d == 4, 3, mask_3d)

    # WT: 1 + 2 + 3
    wt_mask = np.isin(clean_mask, [1, 2, 3])
    # TC: 1 + 3
    tc_mask = np.isin(clean_mask, [1, 3])
    # ET: 3
    et_mask = (clean_mask == 3)

    return {
        "WT": extract_surface_mesh_from_mask(wt_mask, voxel_spacing),
        "TC": extract_surface_mesh_from_mask(tc_mask, voxel_spacing),
        "ET": extract_surface_mesh_from_mask(et_mask, voxel_spacing)
    }
