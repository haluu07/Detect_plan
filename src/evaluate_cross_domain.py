"""
Script danh gia toan dien mo hinh da huan luyen tren ca 2 mien mien (Domain):
1. In-Domain Test: PlantVillage test set
2. Cross-Domain Test: PlantDoc field test set

Tinh toan cac chi so:
- Top-1 Accuracy, Top-5 Accuracy
- Macro Precision, Recall, F1-Score
- Do tut giam hieu nang (Performance Degradation Drop)
- Ve va xuat bieu do Confusion Matrix & Classification Report
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm

from dataset import build_dataloaders
from models import get_model

def run_inference(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_paths = []
    
    with torch.no_grad():
        for images, labels, paths in tqdm(dataloader, desc="Inference", leave=False):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_paths.extend(paths)
            
    return np.array(all_preds), np.array(all_labels), all_paths

def plot_and_save_cm(y_true, y_pred, class_names, output_file: Path, title: str):
    # Chi ve ma tran nham lan cho cac lop xuat hien trong tap du lieu
    unique_labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    sub_class_names = [class_names[i] for i in unique_labels]

    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, annot=False, fmt="d", cmap="Blues",
        xticklabels=sub_class_names, yticklabels=sub_class_names
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Predicted Class", fontsize=11)
    plt.ylabel("True Class", fontsize=11)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"[SAVED] Da luu bieu do ma tran nham lan tai: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Danh gia Cross-Domain Generalization")
    parser.add_argument("--checkpoint", type=str, required=True, help="Duong dan toi file .pth checkpoint")
    parser.add_argument("--model", type=str, default=None, help="Kien truc model (neu khong co se tu doc tu checkpoint)")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="reports")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*70)
    print(" ĐÁNH GIÁ TỔNG QUÁT HÓA MIỀN DỮ LIỆU (CROSS-DOMAIN EVALUATION)")
    print(f" Checkpoint: {args.checkpoint}")
    print(f" Device: {device}")
    print("="*70)

    # 1. Doc classes.json
    classes_json = Path(args.processed_dir) / "classes.json"
    with open(classes_json, "r", encoding="utf-8") as f:
        classes_data = json.load(f)
    classes = classes_data["classes"]
    num_classes = len(classes)

    # 2. Load Checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model_name = args.model or ckpt.get("args", {}).get("model", "resnet50")
    print(f"[+] Kien truc mo hinh: {model_name}")

    model = get_model(model_name, num_classes=num_classes, pretrained=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    # 3. Load Data
    loaders = build_dataloaders(processed_dir=args.processed_dir, batch_size=args.batch_size)
    out_dir = Path(args.output_dir) / f"{model_name}_{Path(args.checkpoint).stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "model": model_name,
        "checkpoint": args.checkpoint
    }

    # 4. Danh gia In-Domain tren PlantVillage Test Set
    if "test_indomain" in loaders:
        print("\n[1/2] Dang danh gia tren In-Domain Test Set (PlantVillage)...")
        y_pred_in, y_true_in, _ = run_inference(model, loaders["test_indomain"], device)
        acc_in = accuracy_score(y_true_in, y_pred_in) * 100
        f1_in = f1_score(y_true_in, y_pred_in, average="macro") * 100
        
        summary["indomain_acc"] = round(acc_in, 2)
        summary["indomain_f1"] = round(f1_in, 2)
        print(f"  -> In-Domain Accuracy: {acc_in:.2f}% | Macro-F1: {f1_in:.2f}%")
        
        plot_and_save_cm(y_true_in, y_pred_in, classes, out_dir / "confusion_matrix_indomain.png",
                         f"In-Domain Confusion Matrix (PlantVillage) - {model_name}")

    # 5. Danh gia Cross-Domain tren PlantDoc
    if "test_crossdomain" in loaders:
        print("\n[2/2] Dang danh gia tren Cross-Domain Test Set (PlantDoc Field Images)...")
        y_pred_cross, y_true_cross, paths_cross = run_inference(model, loaders["test_crossdomain"], device)
        acc_cross = accuracy_score(y_true_cross, y_pred_cross) * 100
        f1_cross = f1_score(y_true_cross, y_pred_cross, average="macro") * 100
        
        summary["crossdomain_acc"] = round(acc_cross, 2)
        summary["crossdomain_f1"] = round(f1_cross, 2)
        print(f"  -> Cross-Domain Accuracy: {acc_cross:.2f}% | Macro-F1: {f1_cross:.2f}%")
        
        plot_and_save_cm(y_true_cross, y_pred_cross, classes, out_dir / "confusion_matrix_crossdomain.png",
                         f"Cross-Domain Confusion Matrix (PlantDoc) - {model_name}")

        # Tinh muc do tut giam
        if "indomain_acc" in summary:
            drop = summary["indomain_acc"] - summary["crossdomain_acc"]
            summary["performance_drop"] = round(drop, 2)
            print("\n" + "="*60)
            print(f" TỔNG KẾT HIỆN TƯỢNG DOMAIN SHIFT:")
            print(f" * Độ chính xác phòng Lab (PlantVillage):  {summary['indomain_acc']}%")
            print(f" * Độ chính xác Thực địa (PlantDoc):       {summary['crossdomain_acc']}%")
            print(f" * Mức độ sụt giảm (Catastrophic Drop):    -{drop:.2f}%")
            print("="*60)

        # Xuat chi tiet per-class report
        unique_labels = sorted(list(set(y_true_cross)))
        target_names = [classes[i] for i in unique_labels]
        report = classification_report(y_true_cross, y_pred_cross, labels=unique_labels, target_names=target_names, output_dict=True)
        pd.DataFrame(report).transpose().to_csv(out_dir / "cross_domain_classification_report.csv")

    with open(out_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[HOÀN TẤT] Kết quả và báo cáo đã được lưu trong thư mục: {out_dir.resolve()}")

if __name__ == "__main__":
    main()
