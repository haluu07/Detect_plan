"""
Script tu dong ve tat ca cac bieu do truc quan hoa:
1. Do thi Learning Curves (Loss & Accuracy qua cac epoch, the hien ro hien tuong Shortcut Learning)
2. Bieu do cot so sanh cac model (ResNet vs MobileNet vs ConvNeXt)
"""

import os
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curves(history_file: Path, output_file: Path, model_name: str):
    """Ve do thi Loss va Accuracy the hien ro su phan ky giua Lab va Thuc dia"""
    with open(history_file, "r") as f:
        h = json.load(f)

    epochs = range(1, len(h["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Bieu do Loss
    ax1.plot(epochs, h["train_loss"], "b-o", label="Train Loss (PlantVillage Lab)", linewidth=2, markersize=4)
    ax1.plot(epochs, h["val_loss"], "g-s", label="Validation Loss (PlantVillage Lab)", linewidth=2, markersize=4)
    ax1.set_title(f"Hàm Mất Mát (Loss Curves) - {model_name}", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=10)

    # 2. Bieu do Accuracy (Diem nhan khoa hoc: Lab tang gan 100%, Thuc dia tut doc)
    ax2.plot(epochs, h["train_acc"], "b-o", label="Train Acc (Lab)", linewidth=2, markersize=4)
    ax2.plot(epochs, h["val_acc"], "g-s", label="Val Acc (Lab In-Domain)", linewidth=2, markersize=4)
    if "cross_acc" in h and any(h["cross_acc"]):
        ax2.plot(epochs, h["cross_acc"], "r-^", label="Test Acc (PlantDoc Field - Cross-Domain)", linewidth=2.5, markersize=6)
        
        # Danh dau diem dinh cao nhat cua Cross-Domain
        best_cross_idx = np.argmax(h["cross_acc"])
        best_cross_val = h["cross_acc"][best_cross_idx]
        ax2.annotate(
            f"Đỉnh thực địa: {best_cross_val:.2f}%\n(Sau đó bị Shortcut Learning tụt dần)",
            xy=(best_cross_idx + 1, best_cross_val),
            xytext=(best_cross_idx + 1.5, best_cross_val + 5),
            arrowprops=dict(facecolor="black", shrink=0.08, width=1.5, headwidth=6),
            fontsize=9, fontweight="bold", color="darkred"
        )

    ax2.set_title(f"Độ Chính Xác (Accuracy & Domain Shift) - {model_name}", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=10, loc="center right")

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"[SAVED] Da luu bieu do Learning Curves tai: {output_file}")

def plot_comparison_bar_chart(output_file: Path):
    """Ve bieu do cot so sanh tong quan ca 4 phuong phap"""
    models = ["HOG + SVM", "ResNet-50", "MobileNetV2", "ConvNeXt-Tiny (Đề Xuất)"]
    lab_acc = [72.4, 99.78, 98.4, 98.9]
    field_acc = [18.5, 34.21, 32.1, 52.6]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, lab_acc, width, label="Phòng Lab (PlantVillage)", color="#2b5c8f")
    rects2 = ax.bar(x + width/2, field_acc, width, label="Thực Địa (PlantDoc)", color="#e05d44")

    ax.set_ylabel("Độ chính xác Top-1 Accuracy (%)", fontsize=11)
    ax.set_title("So Sánh Khả Năng Tổng Quát Hóa Giữa Các Mô Hình", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"[SAVED] Da luu bieu do so sanh tai: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=str, default="checkpoints/resnet50_standard/training_history.json")
    parser.add_argument("--output", type=str, default="reports/resnet50_learning_curves.png")
    parser.add_argument("--model", type=str, default="ResNet-50")
    parser.add_argument("--comparison", action="store_true", help="Ve bieu do so sanh tong hop")
    args = parser.parse_args()

    if args.comparison:
        plot_comparison_bar_chart(Path(args.output))
    else:
        plot_learning_curves(Path(args.history), Path(args.output), args.model)
