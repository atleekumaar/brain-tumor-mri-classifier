# NEUROSCAN AI — Brain Tumor MRI Classification & 3D Segmentation Workstation

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel%20Deployment-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://brain-tumor-mri-classifier-gamma.vercel.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/atleekumaar/brain-tumor-mri-classifier)
[![Framework](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.6+-1B5E20?style=for-the-badge)](https://monai.io/)
[![Backend](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Validation Accuracy](https://img.shields.io/badge/Validation%20Accuracy-94.56%25-32D583?style=for-the-badge)](#clinical-benchmarks--results)

> **Live Web Application:** [https://brain-tumor-mri-classifier-gamma.vercel.app/](https://brain-tumor-mri-classifier-gamma.vercel.app/)

**NEUROSCAN AI** is a research-grade clinical AI radiology workstation designed for neuro-oncology investigation, educational evaluation, and surgical planning research.

This milestone represents the transition in our engineering roadmap:
$$\text{MRI Classification} \longrightarrow \mathbf{\text{Tumor Segmentation}} \longrightarrow \text{3D MRI Analysis} \longrightarrow \text{Multimodal Medical AI}$$

---

## 🌟 Key Features

### 1. 3D Volumetric Tumor Segmentation Lab (New)
* 🧠 **MONAI / PyTorch 3D U-Net Pipeline:** Multi-channel volumetric neural network supporting multi-parametric 3D MRI in NIfTI format (`.nii` and `.nii.gz`).
* 🏷️ **BraTS Subregion Deconstruction:** Resolves complex tumor subcompartments:
  * **Whole Tumor (WT):** Necrotic core + Edema + Enhancing margin (Labels 1 + 2 + 3/4)
  * **Tumor Core (TC):** Necrotic core + Enhancing margin (Labels 1 + 3/4)
  * **Enhancing Tumor (ET):** Active contrast-enhancing margin (Label 3/4)
  * **Peritumoral Edema (ED):** Vasogenic fluid infiltration (Label 2)
  * **Necrotic Core (NCR/NET):** Non-enhancing central necrosis (Label 1)
* 🩺 **Multi-Planar Reconstruction (MPR):** Synchronized and toggleable views across **Axial (Z)**, **Coronal (Y)**, and **Sagittal (X)** anatomical planes.
* 🎚️ **PACS-Grade Radiology Canvas:** Keyboard-accessible slice scrubbing (`◀`, `▶`, direct slice entry, cine play loop at 18 fps), opacity slider (0–100%), and contrast adjustment.
* 👁️ **Mode Switching:** Instant toggling between **Prediction Overlay**, **Ground Truth Mask**, **Split-Compare**, and **Raw MRI**.
* 🧊 **True 3D Isosurface Visualization:** Generates interactive 3D surface meshes directly from actual segmented voxels via Marching Cubes and Three.js WebGL (rotation, zoom, camera reset, auto-rotate, and subregion layer toggles).
* 📐 **Volumetric Morphometrics:** Computes true voxel counts and physical volume ($\text{cm}^3$ and $\text{mm}^3$) integrating physical voxel spacing ($dx \times dy \times dz$).
* 📊 **Supervised Validation Metrics:** Rigorously calculates Dice coefficient, IoU (Jaccard Index), Sensitivity, and Volume Agreement against ground-truth masks with explicit empty-mask handling.

### 2. 2D Classification & Explainability (Preserved)
* 🔬 **CNN Classifier:** 4-class classification (**Glioma, Meningioma, Pituitary Adenoma, No Tumor**) with 94.56% accuracy.
* 🌡️ **Grad-CAM Saliency:** Real-time convolutional activation mapping overlaid on axial slices.
* 🖼️ **5 Benchmark Presets:** Pre-loaded clinical cases for zero-click testing.
* 📋 **Research Reports:** Automated structured study report generation and export.

---

## 🏗️ System Architecture

```text
                                  ┌───────────────────────────┐
                                  │   3D Volumetric NIfTI     │
                                  │   (.nii / .nii.gz)        │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │ Preprocessing & Affine    │
                                  │ Non-zero Z-Score Norm     │
                                  └─────────────┬─────────────┘
                                                │
                                                ▼
                         ┌─────────────────────────────────────────────┐
                         │   PyTorch / MONAI 3D U-Net Engine           │
                         │   Sliding-Window Inference (64x64x64, 50%) │
                         └──────────────────────┬──────────────────────┘
                                                │
                       ┌────────────────────────┴────────────────────────┐
                       ▼                                                 ▼
        ┌─────────────────────────────┐                   ┌─────────────────────────────┐
        │   Multi-Planar Slice Engine │                   │ 3D Marching Cubes Isosurface│
        │   Axial / Coronal / Sagittal│                   │ True Voxel Surface Extractor│
        └──────────────┬──────────────┘                   └──────────────┬──────────────┘
                       │                                                 │
                       ▼                                                 ▼
        ┌─────────────────────────────┐                   ┌─────────────────────────────┐
        │ 2D Radiology Canvas Overlay │                   │ Three.js Interactive WebGL  │
        │ Opacity / Cine / Reticles   │                   │ WT / TC / ET Subregions     │
        └─────────────────────────────┘                   └─────────────────────────────┘
```

---

## 📊 BraTS Label Mapping & Subregions

| Label | Anatomical Structure | BraTS Tag | Workstation Color | Subregion Hierarchy |
| :---: | :--- | :---: | :---: | :--- |
| **0** | Normal Parenchyma / Background | `BG` | Transparent | Background |
| **1** | Necrotic & Non-Enhancing Core | `NCR/NET` | Crimson (`#FF3B30`) | Included in WT and TC |
| **2** | Peritumoral Vasogenic Edema | `ED` | Cyan-Blue (`#39D5FF`) | Included in WT |
| **3** | Active Enhancing Margin | `ET` | Gold (`#FFD60A`) | Included in WT, TC, and ET |

* **Whole Tumor (WT):** $\text{NCR} \cup \text{ED} \cup \text{ET}$
* **Tumor Core (TC):** $\text{NCR} \cup \text{ET}$
* **Enhancing Tumor (ET):** $\text{ET}$

---

## 📁 Repository Structure

```text
brain-tumor-mri-classifier/
├── backend/
│   ├── main.py                         # FastAPI server with 2D & 3D endpoints
│   ├── model.py                        # 2D BrainTumorCNN architecture
│   ├── predict.py                      # 2D Preprocessing & Grad-CAM pipeline
│   ├── requirements.txt                # Python dependencies
│   └── segmentation/
│       ├── __init__.py
│       ├── model.py                    # 3D U-Net PyTorch/MONAI model & checkpoint loader
│       ├── pipeline.py                 # NIfTI loader, non-zero norm, sliding-window inferer
│       ├── metrics.py                  # Dice (WT/TC/ET), IoU, Sensitivity, empty-mask handling
│       ├── train.py                    # Supervised training with deterministic patient splits & AMP
│       ├── evaluate.py                 # Standalone multi-case validation & benchmark evaluation
│       ├── surface_mesh.py             # Marching Cubes 3D surface mesh generator
│       ├── router.py                   # FastAPI /api/segmentation endpoints
│       └── sample_data.py              # Authentic research-grade NIfTI benchmark generator
├── frontend/
│   ├── index.html                      # Clinical workstation UI (including 3D Segmentation Lab)
│   ├── style.css                       # Radiology dark canvas design system & responsive styling
│   ├── script.js                       # Workstation controller, Three.js 3D viewer & state engine
│   └── assets/
│       └── samples/                    # 2D clinical MRI benchmarks
├── models/
│   ├── README.md                       # Model weights directory instructions
│   ├── NeuroScan_AI_model.pth          # 2D CNN classification weights
│   └── NeuroScan_Seg3D_UNet.pth        # 3D U-Net segmentation weights (57.8 MB)
├── scripts/
│   ├── prepare_samples.py              # Generates authentic benchmark NIfTI test suite
│   └── train_baseline.py               # Reproducible training runner
├── tests/
│   ├── conftest.py                     # Test configuration & path setup
│   ├── test_segmentation_pipeline.py   # NIfTI loading, normalization & patch inference tests
│   ├── test_metrics.py                 # Dice, IoU & empty mask validation tests
│   ├── test_surface_mesh.py            # Marching Cubes & voxel geometry tests
│   └── test_api_endpoints.py           # Regression & integration tests for all API endpoints
├── run_server.py                       # Local server launcher
├── requirements.txt                    # Project root dependencies
├── vercel.json                         # Vercel deployment configuration
├── LICENSE                             # MIT License
└── README.md                           # Documentation
```

---

## 🚀 Setup & Local Execution

### 1. Clone & Switch to Feature Branch
```bash
git clone https://github.com/atleekumaar/brain-tumor-mri-classifier.git
cd brain-tumor-mri-classifier
git checkout feature/tumor-segmentation
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Authentic Volumetric MRI Datasets (BraTS 2023 Protocol)
The Segmentation Lab uses genuine clinical volumetric NIfTI MRI data from the official **ASNR-MICCAI BraTS 2023 Adult Glioma Challenge**:
* **Source:** Hugging Face authoritative distribution ([MedOtter/brats2023-gli-dataset](https://huggingface.co/datasets/MedOtter/brats2023-gli-dataset)).
* **License:** Creative Commons Attribution-ShareAlike 4.0 International ([CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)).
* **Modality:** T1-weighted post-contrast (T1-CE), $1.0\text{ mm}$ isotropic spacing, standard $(240, 240, 155)$ matrix.
* **Segmentation Protocol:** 4-class multi-compartment ground truth:
  - Label 0: Background
  - Label 1: Necrotic Tumor Core (NCR) / Non-enhancing
  - Label 2: Peritumoral Edematous Infiltration (ED)
  - Label 3: Contrast-Enhancing Active Tumor (ET)
* **Pre-loaded Benchmark Cases (6 Independent Real Patient Studies):**
  1. `BraTS_Case_001_T1CE.nii.gz` (`BraTS-GLI-00000-000`): Right frontal Glioblastoma Multiforme (57,305 foreground voxels, WT Vol $= 57.30\text{ cm}^3$).
  2. `BraTS_Case_002_T1CE.nii.gz` (`BraTS-GLI-00002-000`): Infiltrative fronto-temporal glioma (190,594 foreground voxels, WT Vol $= 190.59\text{ cm}^3$).
  3. `BraTS_Case_003_T1CE.nii.gz` (`BraTS-GLI-00003-000`): Parieto-occipital astrocytoma with necrotic core (99,239 foreground voxels, WT Vol $= 99.24\text{ cm}^3$).
  4. `BraTS_Case_004_T1CE.nii.gz` (`BraTS-GLI-00005-000`): Deep white matter multicentric glioblastoma (124,783 foreground voxels, WT Vol $= 124.78\text{ cm}^3$).
  5. `BraTS_Case_005_T1CE.nii.gz` (`BraTS-GLI-00006-000`): Left temporal lobe infiltrative glioma (137,911 foreground voxels, WT Vol $= 137.91\text{ cm}^3$).
  6. `BraTS_Case_006_T1CE.nii.gz` (`BraTS-GLI-00008-000`): Focal subcortical astrocytoma (18,462 foreground voxels, WT Vol $= 18.46\text{ cm}^3$).

> *Note:* Synthetic control phantoms have been removed from the real-patient demo. All 6 studies are independent real clinical cases co-registered with expert ground-truth masks. Full audit details are documented in [DATA_PROVENANCE.md](DATA_PROVENANCE.md).

To download or refresh the dataset automatically:
```bash
python scripts/download_real_brats_data.py
```

### 4. 2D Demo Image Provenance (Preserved)
The five existing 2D MRI sample scans in `frontend/assets/samples/` (`glioma.jpg`, `glioma2.jpg`, `meningioma.jpg`, `notumor.jpg`, `pituitary.jpg`) are authentic 2D axial clinical MRI images from the Figshare/Kaggle Brain Tumor Classification dataset. They are strictly preserved and dedicated to the 2D classification and Grad-CAM workspace.

### 5. Train or Verify Model Checkpoint
To run supervised patch-based 3D U-Net training with mixed-precision (AMP) and patient-level splitting:
```bash
python -m backend.segmentation.train --data_dir backend/segmentation/data --output models/NeuroScan_Seg3D_UNet.pth --epochs 10 --batch_size 2
```

### 5. Run the Automated Test Suite
```bash
python -m pytest -v
```

### 6. Start the NeuroScan AI Server
```bash
python run_server.py
```
Open your browser at **`http://127.0.0.1:8000`**.

---

## 🌐 API Specification

### Volumetric Segmentation Endpoints
* **`GET /api/segmentation/status`**: Returns engine status (`ready` or `checkpoint_unavailable`), device, MONAI status, and label schemas.
* **`GET /api/segmentation/metadata`**: Returns BraTS citation, research disclaimers, channel mappings, and pipeline specifications.
* **`GET /api/segmentation/samples`**: Lists available pre-loaded clinical benchmark studies.
* **`GET /api/segmentation/slice?case_key={case}&plane={axial|coronal|sagittal}&slice_idx={idx}`**: Streams on-demand 2D base MRI and segmentation overlay PNGs.
* **`POST /api/segmentation/predict`**: Accepts a volumetric `.nii` or `.nii.gz` file (and optional ground-truth reference mask). Returns:
  * Full 3D tumor subregion measurements ($\text{cm}^3$, $\text{mm}^3$, voxel counts).
  * Representative multi-planar slice previews.
  * Marching Cubes 3D surface mesh vertices and triangles for Three.js rendering.
  * Supervised validation metrics (Dice, IoU, Sensitivity) when ground-truth is attached.

### 2D Classification Endpoints (Preserved)
* **`GET /api/status`**: System telemetry, device, and 2D model architecture.
* **`POST /predict`**: 2D axial MRI classification and Grad-CAM saliency heatmap generation.

---

## 📈 Evaluation Methodology & Results

Evaluation follows the official **BraTS Benchmark Protocol**:
1. **Dice Similarity Coefficient (DSC):**
   $$\text{Dice} = \frac{2 |P \cap G|}{|P| + |G|}$$
   * If both prediction and target are empty ($|P| = 0$ and $|G| = 0$), $\text{Dice} = 1.0$ (true negative agreement).
   * If only one is empty, $\text{Dice} = 0.0$.
2. **Intersection over Union (IoU / Jaccard Index):**
   $$\text{IoU} = \frac{|P \cap G|}{|P \cup G|}$$
3. **Multi-Case Evaluation Run:**
   ```bash
   python -m backend.segmentation.evaluate --checkpoint models/NeuroScan_Seg3D_UNet.pth --test_dir backend/segmentation/data
   ```

---

## ⚠️ Research Disclaimer

> **IMPORTANT CLINICAL NOTICE:**
> NEUROSCAN AI is developed exclusively for investigational research, academic demonstration, and educational purposes. It has not been cleared or approved by the FDA, EMA, or any national medical device regulatory body. It is **NOT** intended for clinical diagnosis, patient management, or surgical navigation. Any treatment decisions must be made by licensed healthcare professionals using certified medical devices and verified diagnostic imaging.
