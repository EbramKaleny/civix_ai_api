import torch
import torchvision.models as models
import torch.nn as nn
from flask import Flask, request, jsonify

# Load a pretrained ResNet18
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 6)  # 6 classes

# Load the trained weights
model.load_state_dict(torch.load("model.pth", map_location=torch.device("cpu")))

# Put the model in evaluation mode
model.eval()

# Setup Flask
app = Flask(__name__)

@app.route('/')
def home():
    return "Civix AI model is running!"

if __name__ == '__main__':
    app.run(debug=True)
