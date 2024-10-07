import torch

class YOLOModel:
    def __init__(self, model_name='yolov8n', device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = torch.hub.load('ultralytics/yolov8', model_name, pretrained=True, trust_repo=True)
        self.model.eval()
        self.model.to(self.device)

    def get_model(self):
        return self.model

    def get_device(self):
        return self.device
