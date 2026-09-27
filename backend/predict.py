from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

IMG_SIZE = 224

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

def predict_image(image_path, model, classes, device):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = F.softmax(output, dim=1).squeeze(0)

    confidence, pred = torch.max(probs, dim=0)
    prob_dict = {classes[i]: round(probs[i].item() * 100, 2) for i in range(len(classes))}

    return classes[pred.item()], round(confidence.item() * 100, 2), prob_dict