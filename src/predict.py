"""
Script test chan doan benh tren la cay (ho tro ca 1 anh va NHIEU ANH dang luoi Grid):
- Neu truyen --image: Chan doan 1 anh cu the.
- Neu truyen --num_images 6 (hoac 9): Tu dong lay 6 (hoac 9) loai benh khac nhau tu tap thuc dia PlantDoc,
  chan doan va ghep thanh mot bang anh luoi (Grid Gallery) cuc dep de chen vao bao cao/slide!
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

def collect_diverse_images(test_dir: Path, max_images: int = 6):
    """Tu dong chon cac anh thuoc cac loai benh/cay khac nhau"""
    selected = []
    if not test_dir.exists():
        return selected

    subdirs = sorted([d for d in test_dir.iterdir() if d.is_dir()])
    for d in subdirs:
        imgs = list(d.glob("*.jpg")) + list(d.glob("*.png")) + list(d.glob("*.JPG"))
        if imgs:
            selected.append((d.name, str(imgs[0])))
        if len(selected) >= max_images:
            break
    return selected

def main():
    parser = argparse.ArgumentParser(description="Demo chan doan AI tren 1 hoac nhieu anh la cay")
    parser.add_argument("--image", type=str, default=None, help="Duong dan anh cu the (neu muon test 1 anh)")
    parser.add_argument("--num_images", type=int, default=6, help="So luong anh muon test dang bang luoi (mac dinh: 6)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/convnext_domain_adaptation/best_adaptation_model.pth")
    parser.add_argument("--fallback_checkpoint", type=str, default="checkpoints/convnext_tiny_strongaug/best_indomain_model.pth")
    parser.add_argument("--model", type=str, default="convnext_tiny")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--output", type=str, default="reports/demo_batch_predictions.png")
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

    # Load Model (Uu tien dung model da Domain Adaptation neu co, hoac dung ConvNeXt goc)
    model = get_model(args.model, num_classes=len(classes), pretrained=False)
    ckpt_path = args.checkpoint if os.path.exists(args.checkpoint) else args.fallback_checkpoint
    print(f"[*] Su dung Checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model = model.to(device)

    # TRƯỜNG HỢP 1: TEST 1 ẢNH CỤ THỂ
    if args.image:
        img_pil, results = predict_single_image(args.image, model, classes, device, topk=3)
        top1 = results[0]
        print("\n" + "="*60)
        print(" 🌿 KẾT QUẢ CHẨN ĐOÁN BỆNH CÂY TRỒNG (1 ẢNH)")
        print("="*60)
        print(f" * File ảnh:          {args.image}")
        print(f" * Chẩn đoán bởi AI: {top1['class_name']}")
        print(f" * Độ tự tin:         {top1['confidence']:.2f}%")
        print("\n Top 3 chẩn đoán:")
        for i, r in enumerate(results, 1):
            print(f"   {i}. {r['class_name']} ({r['confidence']:.2f}%)")
        print("="*60)

        plt.figure(figsize=(7, 7))
        plt.imshow(img_pil)
        plt.axis("off")
        plt.title(f"DỰ ĐOÁN: {top1['class_name']}\nĐỘ TỰ TIN: {top1['confidence']:.2f}%",
                  fontsize=12, fontweight="bold", color="darkgreen" if "healthy" in top1['class_name'] else "darkred", pad=10)
        out_path = Path("reports/demo_prediction.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[OK] Da luu anh don tai: {out_path.resolve()}\n")

    # TRƯỜNG HỢP 2: TEST NHIỀU ẢNH (BẢNG LƯỚI GRID GALLERY)
    else:
        test_dir = Path("data/plantdoc/test")
        sample_list = collect_diverse_images(test_dir, max_images=args.num_images)
        if not sample_list:
            print(f"[-] Khong tim thay anh trong {test_dir}")
            return

        n_samples = len(sample_list)
        cols = 3
        rows = (n_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten() if n_samples > 1 else [axes]

        print("\n" + "="*70)
        print(f" 🌿 BẢNG KẾT QUẢ CHẨN ĐOÁN TRÊN {n_samples} LOÀI CÂY & BỆNH THỰC TẾ KHÁC NHAU")
        print("="*70)

        for i, (folder_name, img_path) in enumerate(sample_list):
            img_pil, results = predict_single_image(img_path, model, classes, device, topk=1)
            top1 = results[0]
            pred_name = top1["class_name"]
            conf = top1["confidence"]

            print(f"[{i+1}/{n_samples}] Thực tế: {folder_name:<30} -> AI Chẩn đoán: {pred_name:<35} ({conf:.1f}%)")

            ax = axes[i]
            ax.imshow(img_pil)
            ax.axis("off")

            # Mau sac: Xanh neu la cay khoe manh, Do/Cam neu la cay bi benh
            title_color = "darkgreen" if "healthy" in pred_name.lower() else "darkred"
            ax.set_title(
                f"Ảnh thực tế: {folder_name}\nAI đoán: {pred_name}\nĐộ tự tin: {conf:.1f}%",
                fontsize=10, fontweight="bold", color=title_color, pad=8
            )

        # An cac o trong neu co
        for j in range(n_samples, len(axes)):
            axes[j].axis("off")

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print("="*70)
        print(f"\n[OK] ĐÃ XUẤT THÀNH CÔNG BẢNG LƯỚI {n_samples} ẢNH TẠI: {out_path.resolve()}\n")

if __name__ == "__main__":
    main()
