import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO

class HandMaskGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_path": ("STRING", {"default": "hand_yolov8x.pt"})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("mask",)

    FUNCTION = "run"

    CATEGORY = "custom/hand"

    def run(self, image, model_path):
        # Convert to OpenCV
        img = np.array(image.convert("RGB"))[..., ::-1]

        # Load YOLOv8 model
        model = YOLO(model_path)
        results = model(img)

        # Create mask
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for r in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, r)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        # Convert to RGBA Image
        mask_img = Image.fromarray(mask).convert("RGB")
        return (mask_img,)
