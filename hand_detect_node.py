import numpy as np
from PIL import Image
import cv2
import torch
from ultralytics import YOLO
import os

import logging
logging.basicConfig(level=logging.DEBUG)

class HandMaskGenerator:
    _model_cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model_path": ("STRING", {"default": "E:/AI/yolov8/hand_yolov8s.pt"})
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "run"
    CATEGORY = "custom/hand"

    def tensor_to_pil(self, image):
        # 添加类型和形状调试信息（输出到控制台）
        print(f"[HandMaskGenerator] 输入类型: {type(image)}")
        if hasattr(image, 'shape'):
            print(f"[HandMaskGenerator] 输入形状: {image.shape}")
        
        # 处理 torch.Tensor 输入
        if isinstance(image, torch.Tensor):
            # 记录张量详情
            print(f"[HandMaskGenerator] 张量类型: {image.dtype}, 值范围: [{image.min().item():.3f}, {image.max().item():.3f}]")
            
            # 移除批处理维度 [1, H, W, C] -> [H, W, C]
            if image.dim() == 4 and image.shape[0] == 1:
                image = image.squeeze(0)
            
            # 处理单通道图像
            if image.shape[0] == 1:  # CHW格式
                image = image.repeat(3, 1, 1)  # 复制为3通道
                print(f"[HandMaskGenerator] 单通道图像已转换为3通道")
            
            # 转换为HWC格式
            if image.shape[0] <= 4:  # C在第一个维度
                image = image.permute(1, 2, 0)
            
            # 转换为numpy并归一化
            arr = image.detach().cpu().numpy()
            
            # 自动检测值范围并归一化
            if arr.dtype == np.float32 or arr.dtype == np.float16:
                if arr.min() >= -1 and arr.max() <= 1:  # [-1,1]范围
                    arr = (arr + 1) * 127.5
                elif arr.min() >= 0 and arr.max() <= 1:  # [0,1]范围
                    arr = arr * 255
                arr = arr.clip(0, 255).astype(np.uint8)
                print(f"[HandMaskGenerator] 浮点张量已转换为uint8")
            elif arr.dtype == np.uint8:
                print(f"[HandMaskGenerator] 已是uint8类型")
            else:
                arr = arr.astype(np.uint8)
        
        # 处理numpy数组输入
        elif isinstance(image, np.ndarray):
            arr = image
            # 添加通道维度如果是灰度图
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=-1)
                arr = np.repeat(arr, 3, axis=-1)
                print(f"[HandMaskGenerator] 灰度图已转换为RGB")
        
        # 处理PIL图像
        elif isinstance(image, Image.Image):
            return image.convert("RGB")
        
        else:
            raise TypeError(f"不支持的输入类型: {type(image)}")

        # 验证最终形状
        if arr.ndim != 3 or arr.shape[2] not in (1, 3, 4):
            error_msg = f"无效的图像形状: {arr.shape}。应为(H, W, C)，C为1,3或4"
            print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)
        
        # 保存调试图像（在ComfyUI目录）
        debug_img = Image.fromarray(arr)
        # debug_path = os.path.join(os.getcwd(), "comfy_debug_handmask_input.png")
        # debug_img.save(debug_path)
        # print(f"[HandMaskGenerator] 已保存调试图像到: {debug_path}")
        
        return debug_img

    def run(self, image, model_path):
        # 打印输入摘要
        print(f"\n{'='*50}")
        print(f"[HandMaskGenerator] 开始处理")
        print(f"{'='*50}")
        
        # 检查模型路径
        if not os.path.isfile(model_path):
            error_msg = f"模型文件不存在: {model_path}"
            print(f"[ERROR] {error_msg}")
            raise FileNotFoundError(error_msg)
        
        # 模型缓存处理
        if model_path not in self._model_cache:
            print(f"[HandMaskGenerator] 加载模型: {model_path}")
            self._model_cache[model_path] = YOLO(model_path)
        model = self._model_cache[model_path]
        
        # 转换图像
        pil_img = self.tensor_to_pil(image)
        
        # 处理图像
        img = np.array(pil_img)[..., ::-1]  # RGB->BGR for OpenCV
        results = model(img)
        
        # 创建掩码
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
        for r in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, r)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # 将掩码转换为张量格式 [batch, height, width]
        mask_tensor = torch.from_numpy(mask).float() / 255.0  # 归一化到 0-1 范围
        mask_tensor = mask_tensor.unsqueeze(0)  # 添加批次维度 [1, H, W]
        
        # 调试：打印张量信息
        print(f"[HandMaskGenerator] 输出掩码形状: {mask_tensor.shape}")
        print(f"[HandMaskGenerator] 输出掩码范围: {mask_tensor.min().item():.3f}-{mask_tensor.max().item():.3f}")
        
        # 保存调试图像（可选）
        # debug_path = os.path.join(os.getcwd(), "comfy_debug_handmask_result.png")
        # Image.fromarray(mask).save(debug_path)
        # print(f"[HandMaskGenerator] 已保存结果掩码到: {debug_path}")
        
        return (mask_tensor,)