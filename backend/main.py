from pathlib import Path
import os, shutil, uuid
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Support running directly or as module
try:
    from model import load_model
    from predict import predict_image
except ImportError:
    from backend.model import load_model
    from backend.predict import predict_image

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "NeuroScan_AI_model.pth"
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="NeuroScan AI - Brain Tumor Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, classes = load_model(str(MODEL_PATH), device)
print(f"[INFO] Running on device: {device} with classes: {classes}")

@app.get("/api/status")
async def get_status():
    return {
        "status": "online",
        "device": str(device),
        "model_loaded": MODEL_PATH.exists(),
        "model_name": "BrainTumorCNN v1.0",
        "architecture": "4x Conv2D (3->32->64->128->256) + MaxPool + Linear(256) + Dropout(0.6) + Linear(4)",
        "framework": "PyTorch",
        "input_size": "224 x 224 x 3",
        "test_accuracy": "94.56%",
        "train_accuracy": "99.07%",
        "f1_score": "0.94",
        "classes": classes
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    temp_path = UPLOAD_DIR / f"{uuid.uuid4()}.{ext}"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        result = predict_image(str(temp_path), model, classes, device)
    finally:
        if temp_path.exists():
            os.remove(temp_path)

    return result

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")