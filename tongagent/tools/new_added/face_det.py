from transformers import AutoProcessor, Tool
import torch
from PIL import Image
import numpy as np
import requests
import face_detection

class FaceDetTool(Tool):
    name = "facedetection"
    description = "A tool that can detect human faces in given images, outputing the bounding boxes of the human faces."
    inputs = {
        "image_path": {
            "description": "The path to the image on which to localize objects. This should be a local path to downloaded image.",
            "type": "string",
        },
    }
    output_type = "any"


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = face_detection.build_detector("DSFDDetector", confidence_threshold=.5, nms_iou_threshold=.3)


    def forward(self,image_path:str)-> list:
        img = Image.open(image_path)
        img = img.convert('RGB')
        with torch.no_grad():
            faces = self.model.detect(np.array(img))
        
        W,H = img.size
        objs = []
        for i,box in enumerate(faces):
            x1,y1,x2,y2,c = [int(v) for v in box.tolist()]
            x1,y1,x2,y2 = self.enlarge_face([x1,y1,x2,y2],W,H)
            mask = np.zeros([H,W]).astype(float)
            mask[y1:y2,x1:x2] = 1.0
            objs.append([x1,y1,x2,y2])
        return objs


    def enlarge_face(self,box,W,H,f=1.5):
        x1,y1,x2,y2 = box
        w = int((f-1)*(x2-x1)/2)
        h = int((f-1)*(y2-y1)/2)
        x1 = max(0,x1-w)
        y1 = max(0,y1-h)
        x2 = min(W,x2+w)
        y2 = min(H,y2+h)
        return [x1,y1,x2,y2]



# m=FaceDetTool()