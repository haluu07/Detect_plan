"""
Script dong goi va xuat mo hinh sang dinh dang ONNX:
- Dinh dang ONNX cho phep nhung mo hinh vao truc tiep:
  * Mobile App: Flutter (onnxruntime_flutter), React Native, iOS, Android (Java/Kotlin)
  * Web App: ONNX Runtime Web
  * Embedded: Raspberry Pi, Jetson Nano
"""

import os
import json
import argparse
from pathlib import Path
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from models import get_model

def export_to_onnx(checkpoint_path: str, output_onnx_path: str, model_name="convnext_tiny", num_classes=38):
    print(f"\n[*] Đang nạp mô hình {model_name} từ: {checkpoint_path}")
    model = get_model(model_name, num_classes=num_classes, pretrained=False)
    
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    # Tao input gia lap 1 anh 224x224
    dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

    print(f"[*] Bắt đầu xuất sang định dạng ONNX...")
    Path(output_onnx_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )
    
    file_size_mb = os.path.getsize(output_onnx_path) / (1024 * 1024)
    print("\n" + "="*60)
    print(f"[THÀNH CÔNG] ĐÃ XUẤT MÔ HÌNH SANG ĐỊNH DẠNG ONNX!")
    print(f" * File đầu ra:     {Path(output_onnx_path).resolve()}")
    print(f" * Dung lượng file: {file_size_mb:.2f} MB")
    print(f" * Sẵn sàng nhúng vào App di động (Flutter / React Native / Android)!")
    print("="*60 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/convnext_domain_adaptation/best_adaptation_model.pth")
    parser.add_argument("--output", default="deploy/plant_disease_model.onnx")
    parser.add_argument("--model", default="convnext_tiny")
    parser.add_argument("--classes_json", default="data/processed/classes.json")
    args = parser.parse_args()

    num_classes = 38
    if os.path.exists(args.classes_json):
        with open(args.classes_json, "r") as f:
            num_classes = json.load(f)["num_classes"]

    export_to_onnx(args.checkpoint, args.output, args.model, num_classes)
