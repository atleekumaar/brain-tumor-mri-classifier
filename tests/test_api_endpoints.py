"""
Integration and regression tests for FastAPI backend:
Verifies existing classification + Grad-CAM regression and new 3D segmentation API endpoints.
"""

import io
from pathlib import Path
from PIL import Image
import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_existing_classification_status_endpoint():
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "classes" in data
    assert "glioma" in data["classes"]
    assert data.get("segmentation_module_enabled") is True


def test_existing_classification_prediction_regression():
    # Create test 224x224 RGB brain MRI JPEG image
    img = Image.new("RGB", (224, 224), color=(30, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test_mri.jpg", buf, "image/jpeg")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "gradcam_available" in data
    assert "inference_time_ms" in data


def test_segmentation_status_endpoint():
    response = client.get("/api/segmentation/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ["ready", "checkpoint_unavailable"]
    assert data["architecture"] == "3D U-Net (BraTS Multi-Parametric Benchmark)"
    assert ".nii.gz" in data["input_formats"]
    assert "labels" in data


def test_segmentation_metadata_endpoint():
    response = client.get("/api/segmentation/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "intended_use" in data
    assert "label_hierarchy" in data
    assert "preprocessing_pipeline" in data


def test_segmentation_samples_endpoint():
    response = client.get("/api/segmentation/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) >= 3
    first = data["samples"][0]
    assert "id" in first
    assert "title" in first


def test_segmentation_slice_endpoint():
    response = client.get("/api/segmentation/slice?case_key=case_001&plane=axial&slice_idx=48")
    assert response.status_code == 200
    data = response.json()
    assert data["plane"] == "axial"
    assert "mri_image" in data
    assert data["mri_image"].startswith("data:image/png;base64,")


def test_segmentation_predict_with_benchmark_case():
    response = client.post(
        "/api/segmentation/predict",
        data={"sample_key": "case_001"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "patient_study_id" in data
    assert "geometry" in data
    assert "measurements" in data
    assert "subregions" in data["measurements"]
    assert "WT" in data["measurements"]["subregions"]
    assert "mesh_3d" in data
    assert data["mesh_3d"] is not None
    # Ground truth validation metrics was computed
    assert "evaluation_metrics" in data
    assert data["evaluation_metrics"] is not None
    assert "mean_dice" in data["evaluation_metrics"]


def test_segmentation_predict_invalid_extension():
    fake_txt = io.BytesIO(b"Not a nifti file")
    response = client.post(
        "/api/segmentation/predict",
        files={"file": ("invalid.txt", fake_txt, "text/plain")}
    )
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]
