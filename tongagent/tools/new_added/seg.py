import os
import string
import shortuuid
import pickle
from typing import List

import torch
from transformers import Tool
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class SegTool(Tool):
    name = "segmentation"
    description = "A tool that can do instance segmentation on the given image."
    inputs = {
        "image_path": {
            "description": "The path of image that the tool can read.",
            "type": "string",
        },
        "prompt": {
            "description": "The bounding box that you want this model to segment. The bounding boxes could be from user input or tool `objectlocation`. You can set it as None or empty list to enable 'Segment Anything' mode.",
            "type": "any",
        },
    }
    output_type = "string"
    
    
    cache_folder = ".cache"
    os.makedirs(cache_folder, exist_ok=True)
    alphabet = string.ascii_lowercase + string.digits
    su = shortuuid.ShortUUID(alphabet=alphabet)
    def __init__(self):
        super().__init__()
        sam2_model = build_sam2(
         "sam2_hiera_l.yaml",
         "model_checkpoints/sam2_checkpoints/sam2_hiera_large.pt"
    )
        predictor = SAM2ImagePredictor(
            sam2_model
        )
        
        mask_generator = SAM2AutomaticMaskGenerator(
            sam2_model,
            points_per_side=8,
            points_per_batch=32,
            pred_iou_thresh=0.7,
            stability_score_thresh=0.92,
            stability_score_offset=0.7,
            crop_n_layers=1,
            box_nms_thresh=0.7,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=25.0,
            use_m2m=True,
        )
        self.predictor = predictor
        self.mask_generator = mask_generator
        
    def forward(self, image_path: str, prompt: List[float] = []) -> str:
        image_raw = Image.open(image_path).convert("RGB")
        image = np.array(image_raw)
        image_uuid = self.su.random(length=8)
        output_image_path = os.path.join(self.cache_folder, f"{image_uuid}.jpg")
        if prompt is None or len(prompt) == 0:
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                masks = self.mask_generator.generate(image)
            plt.figure(figsize=(20, 20))
            plt.imshow(image)
            show_anns(masks)
            plt.axis('off')
            plt.savefig(output_image_path)
            plt.close()
        else:
            self.predictor.set_image(image)
            bboxs = np.array(prompt)
            # x1, x2 = bboxs[:, 0], bboxs[:, 2]
            # y1, y2 = bboxs[:, 1], bboxs[:, 3]
            # x = (x1 + x2) / 2
            # y = (y1 + y2) / 2
            # points = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
            # n_points = points.shape[0]
            # input_labels = np.array([1 for i in range(n_points)])
            masks, _, _ = self.predictor.predict(
                point_coords = None,
                point_labels = None,
                box = bboxs,
                multimask_output=False,
            )
            print(masks.shape)
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            for mask in masks:
                show_mask(mask.squeeze(0), plt.gca(), random_color=True)
            for box in bboxs:
                show_box(box, plt.gca())
            plt.axis('off')
            plt.savefig(output_image_path)
            plt.close()
        # print(len(masks), masks)
        
        
        uuid = self.su.random(length=8)
        fname = f"{uuid}.pkl"
        with open(os.path.join(self.cache_folder, fname), "wb") as f:
            pickle.dump(
                {
                    "image": output_image_path,
                    "masks": masks
                }, 
                f)
        return os.path.join(self.cache_folder, fname)
    
def show_anns(anns, borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            import cv2
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)
    
def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

    