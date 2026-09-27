# NeuroScan AI — Brain Tumor MRI Classifier

[![GitHub Repo](https://img.shields.io/badge/GitHub-atleekumaar%2Fbrain--tumor--mri--classifier-blue?logo=github)](https://github.com/atleekumaar/brain-tumor-mri-classifier)
[![Framework](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Backend](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Accuracy](https://img.shields.io/badge/Test%20Accuracy-94.56%25-brightgreen)](#results)

A deep learning and web-based diagnostic assistant for classifying brain MRI images into **Glioma, Meningioma, No Tumor, and Pituitary Tumor** using a custom CNN built with **PyTorch** and an interactive **FastAPI + Modern Web UI**.

---

## 📌 Overview

* **Task:** Multi-class Image Classification
* **Framework:** PyTorch
* **Backend:** FastAPI (Uvicorn)
* **Frontend:** Modern Responsive Web Interface (HTML5, CSS3, JavaScript)
* **Input Size:** 224 × 224 RGB
* **Training Images:** 5,600
* **Testing Images:** 1,600
* **Classes:** 4 (`glioma`, `meningioma`, `notumor`, `pituitary`)
* **Test Accuracy:** 94.56%
* **F1-Score:** 0.94

---

## 🏗️ Model Architecture

```text
Input Image (224 × 224 × 3)
      │
      ▼
Conv2D (3 → 32, k=3, p=1) + ReLU + MaxPool(2)
      │
      ▼
Conv2D (32 → 64, k=3, p=1) + ReLU + MaxPool(2)
      │
      ▼
Conv2D (64 → 128, k=3, p=1) + ReLU + MaxPool(2)
      │
      ▼
Conv2D (128 → 256, k=3, p=1) + ReLU + MaxPool(2)
      │
      ▼
Flatten (256 × 14 × 14 = 50,176)
      │
      ▼
Linear (50,176 → 256) + ReLU
      │
      ▼
Dropout (p=0.6)
      │
      ▼
Linear (256 → 4)
      │
      ▼
Softmax Probabilities & Prediction
```

---

## 📊 Dataset & Results

### Classes
* `glioma` — Glioma tumor
* `meningioma` — Meningioma tumor
* `notumor` — Healthy / No tumor detected
* `pituitary` — Pituitary gland tumor

### Metrics

| Metric | Score |
| :--- | :---: |
| **Training Accuracy** | 99.07% |
| **Test Accuracy** | **94.56%** |
| **Weighted F1-Score** | **0.94** |

---

## 📁 Project Structure

```text
brain-tumor-mri-classifier/
├── backend/
│   ├── main.py              # FastAPI server & static file serving
│   ├── model.py             # PyTorch CNN architecture & loader
│   ├── predict.py           # Preprocessing & inference pipeline
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── index.html           # Medical console web interface
│   ├── style.css            # Modern dark-mode UI styling
│   └── script.js            # Image upload, scan effect & API calls
├── model/
│   └── MRI-Brain-Tumor-Classifier.ipynb  # Training & evaluation notebook
├── models/
│   ├── README.md            # Weights setup guide
│   └── NeuroScan_AI_model.pth # Trained model weights (placed here)
├── .gitignore
├── LICENSE
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Clone the Repository

```bash
git clone https://github.com/atleekumaar/brain-tumor-mri-classifier.git
cd brain-tumor-mri-classifier
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Model Weights

1. Train the model using `model/MRI-Brain-Tumor-Classifier.ipynb` (e.g. on Google Colab or Kaggle with GPU).
2. Save or place the exported `NeuroScan_AI_model.pth` inside the `models/` directory:
   ```text
   models/NeuroScan_AI_model.pth
   ```
*(Note: If weights are not yet present, the backend will start in demo/fallback mode to allow UI and API testing).*

### 4. Run the Application

Start the FastAPI backend (which also serves the frontend):

```bash
uvicorn backend.main:app --reload --port 8000
```

Open your browser and navigate to:
```text
http://127.0.0.1:8000
```

---

## 🌐 API Endpoints

- `GET /` — Web interface
- `GET /api/status` — Backend & model health check
- `POST /predict` — Upload an MRI image file and receive classification probabilities

**Example Response:**
```json
{
  "prediction": "glioma",
  "confidence": 98.74,
  "probabilities": {
    "glioma": 98.74,
    "meningioma": 0.82,
    "notumor": 0.31,
    "pituitary": 0.13
  }
}
```

---

## ⚠️ Disclaimer

This project is developed for **educational and research purposes only**. It is not a certified medical device and should not be used as a substitute for professional medical advice, diagnosis, or treatment.
