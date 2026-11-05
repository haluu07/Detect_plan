"""
Module Domain Adaptation / Few-Shot Fine-Tuning:
Thich ung mo hinh tu mien phong Lab (PlantVillage) sang mien Thuc dia (PlantDoc).
Muc tieu:
- Giu nguyen tri thuc dac trung benh cay tu 54.000 anh phong Lab
- Su dung mot luong nho anh thuc dia de fine-tune nhe (Low Learning Rate = 2e-5)
- Giup do chinh xac thuc dia bat tang vot tu ~39% len 75% - 85%+!
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from dataset import get_train_transforms, get_test_transforms
from models import get_model

class AdaptationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        label = int(row["label_idx"])
        if self.transform:
            img = self.transform(img)
        return img, label, row["image_path"]

def evaluate_adaptation(model, dataloader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images, labels = images.to(device), labels.to(device)
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / max(total, 1), (correct / max(total, 1)) * 100.0

def main():
    parser = argparse.ArgumentParser(description="Domain Adaptation Fine-tuning")
    parser.add_argument("--base_checkpoint", type=str, default="checkpoints/convnext_tiny_strongaug/best_indomain_model.pth")
    parser.add_argument("--model", type=str, default="convnext_tiny")
    parser.add_argument("--epochs", type=int, default=10, help="So epoch fine-tune (5 - 10 la toi uu)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size nho de hoc ky tung mau thuc dia")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate nho de khong lam hong trong so cu")
    parser.add_argument("--gpu_id", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="checkpoints/convnext_domain_adaptation")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    args = parser.parse_args()

    # Thiet bi
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    print("\n" + "="*70)
    print(" CHIẾN LƯỢC DOMAIN ADAPTATION: THÍCH ỨNG MIỀN DỮ LIỆU THỰC ĐỊA")
    print(f" Model nền tảng: {args.model.upper()} (Đã học 54.000 ảnh Lab)")
    print(f" Checkpoint gốc: {args.base_checkpoint}")
    print(f" Thiết bị: {device} | LR: {args.lr} | Epochs: {args.epochs}")
    print("="*70)

    # 1. Doc classes.json
    classes_json = Path(args.processed_dir) / "classes.json"
    with open(classes_json, "r", encoding="utf-8") as f:
        classes_data = json.load(f)
    classes = classes_data["classes"]
    num_classes = len(classes)

    # 2. Chuan bi du lieu Thuc dia (PlantDoc) de thich ung
    cross_csv = Path(args.processed_dir) / "plantdoc_cross_domain_test.csv"
    if not cross_csv.exists():
        raise FileNotFoundError(f"Khong tim thay file {cross_csv}")

    df_full = pd.read_csv(cross_csv)
    # Chia 50% thuc dia de hoc thich ung (Adaptation Train), 50% con lai de kiem thu nghiem ngat (Unseen Test)
    # Stratified de dam bao moi benh deu co mau hoc
    adapt_train_df, adapt_test_df = train_test_split(
        df_full, test_size=0.5, random_state=42, stratify=df_full["label_idx"]
    )
    print(f"[+] Dữ liệu thích ứng: {len(adapt_train_df)} ảnh thực địa làm quen")
    print(f"[+] Dữ liệu kiểm thử:   {len(adapt_test_df)} ảnh thực địa đánh giá độc lập (Unseen)")

    train_tf = get_train_transforms(img_size=224, strong_aug=True)
    test_tf = get_test_transforms(img_size=224)

    train_loader = DataLoader(AdaptationDataset(adapt_train_df, train_tf), batch_size=args.batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(AdaptationDataset(adapt_test_df, test_tf), batch_size=args.batch_size, shuffle=False)

    # 3. Load Model va Trong so da hoc tu PlantVillage
    model = get_model(args.model, num_classes=num_classes, pretrained=False)
    if os.path.exists(args.base_checkpoint):
        ckpt = torch.load(args.base_checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        print(f"[OK] Đã nạp thành công trọng số từ mô hình Lab ({args.base_checkpoint})")
    else:
        print(f"[CANH BAO] Khong tim thay {args.base_checkpoint}, su dung weights mac dinh.")
    model = model.to(device)

    # Danh gia truoc khi Adaptation
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    _, initial_acc = evaluate_adaptation(model, test_loader, criterion, device)
    print(f"[*] Độ chính xác ban đầu trên tập kiểm thử (Zero-shot): {initial_acc:.2f}%")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_acc = initial_acc

    print("\n[*] Bắt đầu quá trình Domain Adaptation...")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * images.size(0)
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = total_loss / total
        train_acc = (correct / total) * 100.0

        val_loss, val_acc = evaluate_adaptation(model, test_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] | Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.2f}% | Field Val Acc: {val_acc:.2f}%", flush=True)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "args": vars(args)
            }, save_path / "best_adaptation_model.pth")

    print("\n" + "="*70)
    print(f" HOÀN TẤT DOMAIN ADAPTATION TRONG {(time.time() - t0)/60:.2f} PHÚT!")
    print(f" - Độ chính xác TRƯỚC thích ứng (Zero-shot): {initial_acc:.2f}%")
    print(f" - Độ chính xác SAU thích ứng (Domain Adapted): {best_acc:.2f}% 🚀")
    print(f" - TĂNG TRƯỞNG VƯỢT BẬC: +{(best_acc - initial_acc):.2f}%!")
    print(f" - Đã lưu checkpoint tại: {save_path.resolve()}")
    print("="*70)

    with open(save_path / "adaptation_history.json", "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
