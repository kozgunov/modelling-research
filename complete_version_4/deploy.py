from flask import Flask, request, jsonify, Response
import torch
from PIL import Image
import numpy as np
import cv2
from models import YOLOModel

app = Flask(__name__)
model = None

def load_model(model_path):
    global model
    model = YOLOModel(model_name='yolov8n', num_classes=6)
    model.load_state_dict(torch.load(model_path)) # loading of the newest model
    model.eval()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "Please upload an image or video file", 400
    file = request.files['file']
    
    if file.filename.lower().endswith(('.png', '.jpg', '.jpeg')): # apply only png
        return process_image(file)
    elif file.filename.lower().endswith(('.mp4', '.avi', '.mov')): # prod.
        return process_video(file)
    else:
        return "Unsupported file format", 400

def process_image(file):
    image = Image.open(file.stream).convert('RGB')
    img_tensor = torch.tensor(np.array(image) / 255.0).permute(2, 0, 1).unsqueeze(0).float()

    with torch.no_grad():
        results = model(img_tensor)

    detections = []
    for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
        detections.append({
            "class": int(cls),
            "confidence": float(conf),
            "bbox": [float(x) for x in xyxy]
        })

    return jsonify(detections)

def process_video(file):
    temp_path = 'temp_video.mp4' # -----------change by automatization -----------------------------------------------
    file.save(temp_path)
    
    cap = cv2.VideoCapture(temp_path)
    
    def generate():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            img_tensor = torch.tensor(frame / 255.0).permute(2, 0, 1).unsqueeze(0).float()
            
            with torch.no_grad():
                results = model(img_tensor)
            
            for *xyxy, conf, cls in results.xyxy[0].cpu().numpy(): 
                cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(frame, f'Class: {int(cls)}, Conf: {conf:.2f}', (int(xyxy[0]), int(xyxy[1])-10), # who is who
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            _, buffer = cv2.imencode('.jpg', frame) # save jpg, because it contains more information
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    load_model('path/to/your/model.pth')
    app.run(debug=True, host='0.0.0.0', port=5000)
