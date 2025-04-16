from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Load your model
model = torch.load("model.pth", map_location=torch.device("cpu"))
model.eval()

# Flask setup
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    image = Image.open(file.stream)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    input_tensor = transform(image).unsqueeze(0)
    output = model(input_tensor)
    predicted = torch.argmax(output, dim=1).item()

    return jsonify({"prediction": predicted})
