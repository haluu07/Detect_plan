import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Doc ket qua 3 model goc
h_res = json.load(open("checkpoints/resnet50_standard/training_history.json"))
h_mob = json.load(open("checkpoints/mobilenet_v2_standard/training_history.json"))
h_cnx = json.load(open("checkpoints/convnext_tiny_strongaug/training_history.json"))

models = ["MobileNetV2", "ResNet-50", "ConvNeXt (Zero-shot)"]
lab = [max(h_mob["val_acc"]), max(h_res["val_acc"]), max(h_cnx["val_acc"])]
field = [max(h_mob["cross_acc"]), max(h_res["cross_acc"]), max(h_cnx["cross_acc"])]

# Neu co ket qua Domain Adaptation
adapt_file = "checkpoints/convnext_domain_adaptation/adaptation_history.json"
if os.path.exists(adapt_file):
    h_ada = json.load(open(adapt_file))
    models.append("ConvNeXt + Adaptation (Đề Xuất)")
    lab.append(max(h_cnx["val_acc"])) # Van giu nguyen tri thuc lab
    field.append(max(h_ada["val_acc"]))

x = np.arange(len(models))
w = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
r1 = ax.bar(x - w/2, lab, w, label="Phòng Lab (PlantVillage)", color="#1f77b4")
r2 = ax.bar(x + w/2, field, w, label="Thực Địa (PlantDoc)", color="#d62728")

ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12, fontweight="bold")
ax.set_title("So Sánh Hiệu Năng Tổng Quát Hóa Giữa Các Mô Hình & Chiến Lược Thích Ứng", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=10, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(axis="y", linestyle="--", alpha=0.5)

for r in r1 + r2:
    ax.annotate(f"{r.get_height():.2f}%",
                xy=(r.get_x() + r.get_width()/2, r.get_height()),
                xytext=(0, 3), textcoords="offset points",
                ha="center", fontweight="bold", fontsize=10)

plt.tight_layout()
output_path = "reports/model_comparison_barchart.png"
plt.savefig(output_path, dpi=300)
print(f"[OK] DA XUAT ANH SO SANH: {output_path}")

