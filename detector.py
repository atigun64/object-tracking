from ultralytics import YOLO
from models import Detection

class Detector:
    def __init__(self, model_path = "yolo26n.pt"):
        self.model = YOLO(model_path)

    def feed_frame(self, img):
        results = self.model(img)

        new_elements = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].tolist()
                cls_id = int(box.cls[0])
                name = self.model.names[cls_id]
                conf = float(box.conf[0])
                new_elements.append(Detection(box=xyxy, name=name, conf=conf, name_id=cls_id))

        return new_elements

