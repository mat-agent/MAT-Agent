import os

from transformers import Tool
from PIL import Image
from paddleocr import PaddleOCR, draw_ocr
from tongagent.utils import CACHE_FOLDER, gen_random_id

class OCRTool(Tool):
    name = "ocr"
    description = "A tool that can extract texts from the image."
    inputs = {
        "image_path": {
            "description": "The path of image that the tool can read.",
            "type": "string",
        },
    }
    output_type = "any"
    
    ocr = PaddleOCR(use_angle_cls=True, lang='en')
    
    def forward(self, image_path: str, debug: bool = False) -> list:
        image = Image.open(image_path).convert("RGB")
        
        result = self.ocr.ocr(image_path, cls=True)
        texts = []
        for idx in range(len(result)):
            res = result[idx]
            for line in res:
                if debug: print(line[-1])
                texts.append(line[-1][0])
        if debug:
            result = result[0]
            boxes = [line[0] for line in result]
            txts = [line[1][0] for line in result]
            scores = [line[1][1] for line in result]
            im_show = draw_ocr(image, boxes, txts, scores, font_path='data/fonts/simfang.ttf')
            im_show = Image.fromarray(im_show)
            filename = os.path.join(CACHE_FOLDER, f"{gen_random_id()}.jpg")
            print("save to", filename)
            im_show.save(filename)
        return texts