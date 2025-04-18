import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from flask import Flask, request, jsonify
from PIL import Image
import io

# --------------------------
# 1. Setup Model
# --------------------------
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 6)  # 6 classes

model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))
model.eval()

# --------------------------
# 2. Class Labels
# --------------------------
class_names = ['broken_streetlights', 'flooding', 'garbage', 'graffiti', 'manhole', 'pothole']

# --------------------------
# 3. Image Preprocessing
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --------------------------
# 4. Flask App
# --------------------------
app = Flask(__name__)

@app.route('/')
def home():
    return "Civix AI model is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    try:
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
    except Exception:
        return jsonify({'error': 'Invalid image format'}), 400

    image = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, dim=0)
        confidence = confidence.item()

        if confidence < 0.6:
            return jsonify({'prediction': 'Unknown'})
        
        predicted_label = class_names[predicted_idx.item()]
        return jsonify({'prediction': predicted_label})


# --------------------------
# 5. Run Server
# --------------------------
if __name__ == '__main__':
    app.run(debug=True)
