from flask import Flask, request, jsonify
import torch
from torchvision import models, transforms
from PIL import Image
import io
import time

app = Flask(__name__)

# === Config ===
MODEL_PATH = "civix_model.pth"
THRESHOLD = 0.75
CLASS_NAMES = ['broken_streetlights', 'garbage', 'graffiti', 'manhole', 'pothole']

# === Load Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval().to(device)

# === Image Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.route("/predict", methods=["POST"])
def predict():
    upload_start_time = time.perf_counter()
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    upload_end_time = time.perf_counter()

    processing_start_time = time.perf_counter()
    img = Image.open(image).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)
        confidence = confidence.item()
        label = CLASS_NAMES[predicted_class.item()] if confidence >= THRESHOLD else "unknown"

    processing_end_time = time.perf_counter()
    return jsonify({
        "prediction": label,
        "confidence": round(confidence, 4),
        "upload_time": round(upload_end_time - upload_start_time, 4),
        "processing_time": round(processing_end_time - processing_start_time, 4),
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
