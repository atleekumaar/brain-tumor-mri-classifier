import torch
import torch.nn as nn

class BrainTumorCNN(nn.Module):
    def __init__(self,num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3, padding=1),nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def load_model(model_path, device):
    DEFAULT_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
    try:
        # PyTorch 2.6+ compatibility
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_path, map_location=device)
        classes = checkpoint.get("classes", DEFAULT_CLASSES)
        model = BrainTumorCNN(num_classes=len(classes)).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        print(f"[INFO] Loaded trained weights from {model_path}")
        return model, classes
    except Exception as e:
        print(f"[WARNING] Could not load model weights from {model_path}: {e}")
        print("[INFO] Initializing default BrainTumorCNN architecture (Untrained/Demo mode).")
        model = BrainTumorCNN(num_classes=len(DEFAULT_CLASSES)).to(device)
        model.eval()
        return model, DEFAULT_CLASSES