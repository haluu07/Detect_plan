"""
Script test du doan (Inference Demo) tren 1 anh la bat ky:
- Nhap 1 anh bat ky (tu PlantDoc hoac tai tu mang ve)
- Model xuat ra ten benh, do tu tin (%) va Top 3 du doan cao nhat
- Tu dong ve va luu anh ket qua chan doan (Visual Prediction Card)
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from models import get_model
from dataset import get_test_transforms

def predict_single_image(image_path: str, model, classes: list, device, topk: int = 3):
    img_pil = Image.open(image_path).convert("RGB")
    transform = get_test_transforms(img_size=224)
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)[0]
        
    top_probs, top_indices = torch.topk(probs, k=topk)
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append({
            "class_name": classes[idx.item()],
            "confidence": prob.item() * 100
        })
        
    return img_pil, results

def main():
    parser = argparse.ArgumentParser(description="Demo du doan benh tren 1 anh la cay bat ky")
    parser.add_argument("--image", type=str, required=True, help="Duong dan file anh la can test")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/convnext_tiny_strongaug/best_indomain_model.pth")
    parser.add_argument("--model", type=str, default="convnext_tiny")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="reports/demo_prediction.png")
    parser.add_argument("--gpu_id", type=int, default=1)
    args = parser.parse_args()

    # Thiet bi
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    # Load Classes
    classes_json = Path(args.processed_dir) / "classes.json"
    with open(classes_json, "r", encoding="utf-8") as f:
        classes = json.load(f)["classes"]

    # Load Model
    model = get_model(args.model, num_classes=len(classes), pretrained=False)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model = model.to(device)

    # Du doan
    img_pil, results = predict_single_image(args.image, model, classes, device, topk=3)

    top1 = results[0]
    print("\n" + "="*60)
    print(" 🌿 KẾT QUẢ CHẨN ĐOÁN BỆNH CÂY TRỒNG BỞI AI")
    print("="*60)
    print(f" * File ảnh kiểm tra:  {args.image}")
    print(f" * Bệnh được chẩn đoán: {top1['class_name']}")
    print(f" * Độ tự tin (Conf):   {top1['confidence']:.2f}%")
    print("\n Top 3 chẩn đoán có khả năng cao nhất:")
    for i, r in enumerate(results, 1):
        print(f"   {i}. {r['class_name']} ({r['confidence']:.2f}%)")
    print("="*60)

    # Ve anh ket qua
    plt.figure(figsize=(8, 7))
    plt.imshow(img_pil)
    plt.axis("off")
    title_text = f"DỰ ĐOÁN: {top1['class_name']}\nĐỘ TỰ TIN: {top1['confidence']:.2f}%"
    plt.title(title_text, fontsize=13, fontweight="bold", color="darkgreen" if "healthy" in top1['class_name'] else "darkred", pad=12)
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"\n[OK] Đã lưu ảnh chẩn đoán trực quan tại: {out_path.resolve()}\n")

if __name__ == "__main__":
    main()
