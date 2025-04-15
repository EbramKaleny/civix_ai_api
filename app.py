from flask import Flask, request, jsonify
from model_def import CivixModel
from PIL import Image
import torch
import torchvision.transforms as transforms
import io

app = Flask(__name__)

# Load model
model = CivixModel()
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# Preprocessing function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    image = Image.open(io.BytesIO(file.read())).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return jsonify({
        'class': predicted.item(),
        'confidence': round(confidence.item(), 3)
    })
