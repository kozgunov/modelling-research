from flask import Flask, request, jsonify
import torch
from PIL import Image
import numpy as np


app = Flask(__name__)
model = torch.hub.load('ultralytics/yolov8', 'yolov8n', pretrained=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "Please upload an image file", 400
    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    img_tensor = torch.tensor(np.array(image) / 255.0).permute(2, 0, 1).unsqueeze(0).float().to(device)

    with torch.no_grad():
        results = model(img_tensor)

    detections = []
    for xmin, ymin, xmax, ymax, conf, cls in results.xyxy[0].cpu().numpy(): # extract predictions
        detections.append({
            "class": model.names[int(cls)],
            "confidence": float(conf),
            "box": [float(xmin), float(ymin), float(xmax), float(ymax)]
        })

    return jsonify(detections)


if __name__ == '__main__':
    app.run()
