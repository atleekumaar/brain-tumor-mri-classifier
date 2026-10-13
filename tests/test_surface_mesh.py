"""
Unit tests for 3D surface mesh generation from actual voxel segmentations.
"""

import numpy as np
import pytest

from backend.segmentation.surface_mesh import (
    extract_surface_mesh_from_mask,
    generate_subregion_meshes
)


def test_empty_mask_mesh_is_none():
    empty_mask = np.zeros((30, 30, 30), dtype=np.uint8)
    mesh = extract_surface_mesh_from_mask(empty_mask)
    assert mesh is None, "Empty mask must return None without fabricating geometry"


def test_surface_mesh_generation_from_voxels():
    # Sphere-like solid mask
    x, y, z = np.ogrid[:30, :30, :30]
    mask = ((x - 15)**2 + (y - 15)**2 + (z - 15)**2) <= 8**2

    mesh = extract_surface_mesh_from_mask(mask, voxel_spacing=(1.5, 1.5, 2.0))
    assert mesh is not None
    assert "vertices" in mesh
    assert "vertex_count" in mesh
    assert mesh["vertex_count"] > 0
    assert "bbox_mm" in mesh
    assert len(mesh["bbox_mm"]) == 3
    # Bounding box should reflect voxel spacing (1.5, 1.5, 2.0)
    assert mesh["bbox_mm"][0] > 10.0
    assert mesh["voxel_count"] == int(np.sum(mask))


def test_generate_subregion_meshes():
    mask = np.zeros((30, 30, 30), dtype=np.uint8)
    # 1: NCR, 2: ED, 3: ET
    mask[10:14, 10:14, 10:14] = 1
    mask[8:18, 8:18, 8:18] = 2
    mask[10:14, 10:14, 10:14] = 1  # restore core
    mask[12:16, 12:16, 12:16] = 3

    sub_meshes = generate_subregion_meshes(mask, (1.0, 1.0, 1.0))
    assert "WT" in sub_meshes
    assert "TC" in sub_meshes
    assert "ET" in sub_meshes
    assert sub_meshes["WT"] is not None
    assert sub_meshes["TC"] is not None
    assert sub_meshes["ET"] is not None
