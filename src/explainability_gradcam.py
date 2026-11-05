"""
Script giai thich mo hinh bang Grad-CAM (Gradient-weighted Class Activation Mapping).
Phan tich truc quan hien tuong:
- Shortcut Learning (Hoc tat tren background phong lab)
- Background Bias (Mo hinh bi phan tam boi dat, troi, anh sang, ngon tay cam la tren anh thuc dia)

Dau ra:
Cac anh so sanh truc quan (Original Image vs Heatmap vs Overlay) luu trong thu muc `reports/gradcam_analysis/`
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F

from models import get_model
from dataset import get_test_transforms, IMAGENET_MEAN, IMAGENET_STD

class GradCAM:
    """
    Grad-CAM trien khai truc tiep nhe nhang, ho tro ResNet, MobileNet, ConvNeXt
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradient = None
        self.activation = None
        
        # Hook forward va backward
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activation = output

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        loss = output[0, target_class]
        loss.backward()
        
        # Global Average Pooling cua gradient tren target layer
        gradients = self.gradient[0]  # shape: [C, H, W]
        activations = self.activation[0]  # shape: [C, H, W]
        weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
        
        # Ket hop tuyen tinh giua weights va activations
        cam = torch.sum(weights * activations, dim=0).detach().cpu().numpy()
        cam = np.maximum(cam, 0) # ReLU
        
        # Chuan hoa ve [0, 1]
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
        else:
            cam = np.zeros_like(cam)
            
        return cam, target_class, output

def find_target_layer(model, model_name: str):
    """Tu dong tim layer convolution cuoi cung cua tung kien truc"""
    name = model_name.lower()
    if "resnet" in name:
        return model.layer4[-1].conv3 if hasattr(model.layer4[-1], "conv3") else model.layer4[-1].conv2
    elif "mobilenet" in name:
        return model.features[-1]
    elif "convnext" in name:
        # timm convnext stages[-1].blocks[-1]
        if hasattr(model, "stages"):
            return model.stages[-1].blocks[-1]
        elif hasattr(model, "features"):
            return model.features[-1]
    # Default fallback
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Conv2d):
            return module
    raise ValueError("Khong tim thay lop Conv phu hop de tao Grad-CAM.")

def overlay_heatmap(img_pil: Image.Image, cam: np.ndarray, alpha: float = 0.5):
    """Chong lop ban do nhiet Grad-CAM len anh goc"""
    img_cv = np.array(img_pil.convert("RGB"))
    h, w, _ = img_cv.shape
    cam_resized = cv2.resize(cam, (w, h))
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    overlay = np.float32(heatmap) * alpha + np.float32(img_cv) * (1 - alpha)
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return heatmap, overlay

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM Visualization de phan tich Shortcut Learning & Domain Shift")
    parser.add_argument("--checkpoint", type=str, required=True, help="File model checkpoint (.pth)")
    parser.add_argument("--image_path", type=str, required=True, help="Anh can phan tich (PlantVillage hoac PlantDoc)")
    parser.add_argument("--model", type=str, default="resnet50")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--output_file", type=str, default="reports/gradcam_analysis.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Classes
    classes_json = Path(args.processed_dir) / "classes.json"
    with open(classes_json, "r", encoding="utf-8") as f:
        classes_data = json.load(f)
    classes = classes_data["classes"]

    # 2. Load Model
    model = get_model(args.model, num_classes=len(classes), pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)

    # 3. Chuan bi Target Layer & Grad-CAM
    target_layer = find_target_layer(model, args.model)
    grad_cam = GradCAM(model, target_layer)

    # 4. Doc anh va Inference
    raw_img = Image.open(args.image_path).convert("RGB")
    transform = get_test_transforms(img_size=224)
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    cam, pred_idx, logits = grad_cam.generate_heatmap(input_tensor)
    pred_class_name = classes[pred_idx]
    confidence = F.softmax(logits, dim=1)[0, pred_idx].item() * 100

    heatmap, overlay = overlay_heatmap(raw_img, cam)

    # 5. Ve va luu bieu do
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(raw_img)
    axes[0].set_title(f"Ảnh Gốc\nFile: {Path(args.image_path).name}", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(heatmap)
    axes[1].set_title("Grad-CAM Attention Heatmap", fontsize=11)
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Dự Đoán: {pred_class_name}\nĐộ Tự Tin: {confidence:.2f}%", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"\n[OK] Đã hoàn thành Grad-CAM!")
    print(f" -> Lớp dự đoán: {pred_class_name} ({confidence:.2f}%)")
    print(f" -> Đã lưu hình ảnh trực quan hóa tại: {out_path.resolve()}")
    print("\n[PHÂN TÍCH HỌC THUẬT]:")
    print(" - Nếu vùng màu đỏ/vàng tập trung vào viền lá hoặc nền xám/trắng -> Mô hình bị 'Shortcut Learning / Background Bias'.")
    print(" - Nếu vùng màu đỏ/vàng tập trung đúng vào các đốm nấm/bệnh trên phiến lá -> Mô hình học đúng đặc trưng bệnh học.")

if __name__ == "__main__":
    main()
