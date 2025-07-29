import numpy as np
from PIL import Image
import cv2
from ultralytics import YOLO
import os

class HandMaskGenerator:
    _model_cache = {}

    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if model_path not in self._model_cache:
            self._model_cache[model_path] = YOLO(model_path)
        self.model = self._model_cache[model_path]

    def run(self, image):
        # image: PIL Image RGB
        pil_img = image.convert("RGB")
        img = np.array(pil_img)[..., ::-1]  # 转成 BGR for opencv

        results = self.model(img)

        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for r in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, r)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        mask_img = Image.fromarray(mask)
        return mask_img


if __name__ == "__main__":
    # 测试代码
    model_path = "E:\\AI\\yolov8\\hand_yolov8s.pt"  # 替换为你的模型路径
    input_image_path = "E:\\Desktop\\ComfyUI_temp_dhood_00001_.png"
    output_mask_path = "E:\\Desktop\\hand_mask.png"

    generator = HandMaskGenerator(model_path)

    input_img = Image.open(input_image_path)
    mask_img = generator.run(input_img)

    mask_img.save(output_mask_path)
    print(f"手部mask已保存到: {output_mask_path}")
