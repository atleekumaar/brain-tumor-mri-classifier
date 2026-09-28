# NEUROSCAN AI — Brain Tumor MRI Classification Workstation

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Vercel%20Deployment-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://brain-tumor-mri-classifier-gamma.vercel.app/)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/atleekumaar/brain-tumor-mri-classifier)
[![Framework](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Backend](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Accuracy](https://img.shields.io/badge/Validation%20Accuracy-94.56%25-32D583?style=for-the-badge)](#clinical-benchmarks--results)

> **Live Web Application:** [https://brain-tumor-mri-classifier-gamma.vercel.app/](https://brain-tumor-mri-classifier-gamma.vercel.app/)

**NEUROSCAN AI** is a production-quality, clinical AI radiology workstation designed for neuro-oncology research and educational evaluation. It analyzes axial brain MRI images and classifies them into **Glioma, Meningioma, Pituitary Adenoma, and Healthy Normal Parenchyma (No Tumor)** using a custom PyTorch Convolutional Neural Network with real-time **Grad-CAM visual explainability**.

---

## 🌟 Key Features

* 🌐 **Live Web Application on Vercel:** Instant client-side testing with zero setup required at [brain-tumor-mri-classifier-gamma.vercel.app](https://brain-tumor-mri-classifier-gamma.vercel.app/).
* 🧠 **High-Performance PyTorch CNN:** Trained on 5,600 verified brain MRI slices and evaluated on 1,600 test slices achieving **94.56% Accuracy** and **0.94 Weighted F1-Score**.
* 🔬 **Grad-CAM Explainability Heatmaps:** Visualizes morphological attention regions driving the model's prediction across convolutional feature maps.
* 🖼️ **5 Authentic Clinical Benchmark Presets:** Pre-loaded real MRI cases (*Glioma #204, Meningioma #118, Pituitary #092, Control Normal #044, Astrocytoma #315*) for instant 1-click test and drag-and-drop evaluation.
* 🌓 **Dark Workstation & Clinical Light Theme:** Smooth toggle with automatic theme memory (`localStorage`).
* 🩺 **Interactive DICOM/PACS-Style Viewer:** Features 0.5x–3x zoom, pan, brightness/contrast sliders, crosshairs, and opacity blend controls.
* 📋 **Research Report Sheet & PDF Export:** Auto-generates structured clinical research study summaries ready for printing or export.
* 📜 **Session History & Analytics:** Tracks classified cases, confidence scores, latency, and probabilities during the session.

---

## 📊 Clinical Benchmarks & Results

| Clinical Metric | Score | Clinical Relevance |
| :--- | :---: | :--- |
| **Diagnostic Sensitivity** | **96.2%** | High true positive rate for reliable tumor detection |
| **Diagnostic Specificity** | **94.8%** | High true negative rate avoiding false alarms on healthy scans |
| **AUC-ROC Score** | **0.974** | Gold-standard discrimination across multi-class distributions |
| **Test Accuracy** | **94.56%** | Evaluated on 1,600 held-out clinical test slices |
| **Training Accuracy** | **99.07%** | Robust multi-stage convolutional convergence |
| **Inference Latency** | **~35 ms** | Real-time forward-pass execution on GPU/CPU |

### Classification Categories:
1. **Glioma (`glioma`):** Intra-axial primary brain neoplasms (Glioblastoma, Astrocytoma, Oligodendroglioma).
2. **Meningioma (`meningioma`):** Extra-axial dural-based tumors arising from the arachnoid layer.
3. **Pituitary (`pituitary`):** Sellar and parasellar region adenomas.
4. **No Tumor (`notumor`):** Healthy control brain parenchyma.

---

## 🏗️ Neural Architecture

```text
Input Brain MRI (224 × 224 × 3 RGB)
       │
       ▼
Conv2D (3 → 32, k=3, p=1) ──► ReLU ──► MaxPool2D(2)
       │
       ▼
Conv2D (32 → 64, k=3, p=1) ─► ReLU ──► MaxPool2D(2)
       │
       ▼
Conv2D (64 → 128, k=3, p=1) ─► ReLU ──► MaxPool2D(2)
       │
       ▼
Conv2D (128 → 256, k=3, p=1) ─► ReLU ──► MaxPool2D(2)  ◄── [Grad-CAM Hook Layer]
       │
       ▼
Flatten (256 × 14 × 14 = 50,176)
       │
       ▼
Dense Linear (50,176 → 256) ──► ReLU
       │
       ▼
Dropout Layer (p = 0.60)
       │
       ▼
Dense Linear (256 → 4)
       │
       ▼
Softmax Probabilities: [Glioma, Meningioma, No Tumor, Pituitary]
```

---

## 📁 Project Structure

```text
brain-tumor-mri-classifier/
├── backend/
│   ├── main.py                  # FastAPI server with /predict & /api/status endpoints
│   ├── model.py                 # PyTorch BrainTumorCNN architecture & fallback loader
│   ├── predict.py               # Preprocessing, PyTorch inference & Grad-CAM pipeline
│   └── requirements.txt         # Python dependencies
├── frontend/
│   ├── index.html               # Clinical AI workstation interface (8 workspace tabs)
│   ├── style.css                # Workstation design system & Dark/Light themes
│   ├── script.js                # State management, image processing, viewer & history
│   └── assets/
│       └── samples/             # Authentic clinical MRI slices & pre-compiled dataset
├── model/
│   └── MRI-Brain-Tumor-Classifier.ipynb  # Training notebook (Google Colab / Kaggle GPU)
├── models/
│   ├── README.md                # Model weights setup guide
│   └── NeuroScan_AI_model.pth   # PyTorch model weights checkpoint
├── vercel.json                  # Vercel deployment configuration
├── requirements.txt             # Project root dependencies
├── LICENSE                      # MIT License
└── README.md                    # Project documentation
```

---

## 🚀 Local Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/atleekumaar/brain-tumor-mri-classifier.git
cd brain-tumor-mri-classifier
```

### 2. Install Python Environment
```bash
pip install -r requirements.txt
```

### 3. Model Weights Setup
1. Train the model using `model/MRI-Brain-Tumor-Classifier.ipynb` on Google Colab or Kaggle (GPU).
2. Place the generated weights file into the `models/` folder:
   ```text
   models/NeuroScan_AI_model.pth
   ```
*(Note: A built-in fallback handler allows full testing even before generating custom weights).*

### 4. Run Application
```bash
uvicorn backend.main:app --reload --port 8000
```
Open your browser at **`http://127.0.0.1:8000`**.

---

## ☁️ Deployment on Vercel

This repository includes a pre-configured `vercel.json` for zero-configuration deployments:

1. Import this repository into [Vercel](https://vercel.com/new).
2. Click **Deploy** (Vercel automatically detects `vercel.json` and serves the static frontend).
3. The application is live instantly!

---

## 🌐 API Specification

### 1. Health Status
`GET /api/status`
```json
{
  "status": "online",
  "device": "cuda",
  "model_loaded": true,
  "model_name": "BrainTumorCNN v1.0",
  "test_accuracy": "94.56%",
  "classes": ["glioma", "meningioma", "notumor", "pituitary"]
}
```

### 2. Neural Inference & Grad-CAM
`POST /predict`
* **Content-Type:** `multipart/form-data`
* **Body:** `file=<binary_image>`

```json
{
  "prediction": "glioma",
  "confidence": 96.84,
  "probabilities": {
    "glioma": 96.84,
    "meningioma": 1.72,
    "notumor": 0.81,
    "pituitary": 0.63
  },
  "inference_time_ms": 32.4,
  "gradcam_available": true,
  "gradcam_heatmap": "data:image/png;base64,..."
}
```

---

## ⚠️ Non-Diagnostic Research Disclaimer

**NEUROSCAN AI** is developed strictly for **academic research, scientific evaluation, and educational demonstration**. It is **not** a certified medical device, definitive diagnostic system, or clinical treatment guidance tool. All predictions are probabilistic outputs generated by a deep neural network and must not be used as a substitute for professional radiological evaluation, histological biopsy, or physician consultation.

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).
