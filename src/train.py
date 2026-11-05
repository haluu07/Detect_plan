"""
Pipeline Huan luyen Deep Learning ho tro GPU/CUDA, Mixed Precision (AMP), Cosine Annealing,
Early Stopping va Danh gia song song ca In-Domain (PlantVillage) va Cross-Domain (PlantDoc).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from dataset import build_dataloaders
from models import get_model

class EarlyStopping:
    """
    Early stopping giup dung huan luyen som khi metric theo doi (val_acc hoac val_loss)
    khong cai thien sau `patience` epoch lien tiep, giup tranh overfitting va tiet kiem thoi gian GPU.
    """
    def __init__(self, patience=5, min_delta=1e-4, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, current_val):
        score = current_val

        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        elif self.mode == "min":
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0

        return self.early_stop


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)
                
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
    avg_loss = total_loss / max(total, 1)
    acc = (correct / max(total, 1)) * 100.0
    return avg_loss, acc

def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, labels, _ in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=torch.cuda.is_available()):
            outputs = model(images)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{(correct/total)*100:.2f}%"})
        
    return total_loss / total, (correct / total) * 100.0

def main():
    parser = argparse.ArgumentParser(description="Deep Learning Training Pipeline")
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=["resnet50", "mobilenet_v2", "convnext_tiny", "swin_tiny", "efficientnet_b0"],
                        help="Kien truc mo hinh")
    parser.add_argument("--epochs", type=int, default=15, 
                        help="So epoch huan luyen toi da (khuyen nghi: 15 cho Baseline, 20 cho ConvNeXt)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate khoi dau")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    parser.add_argument("--strong_aug", action="store_true", 
                        help="Bat Strong Augmentation (RandAugment, ColorJitter) de tang kha nang Cross-Domain")
    parser.add_argument("--num_workers", type=int, default=4, help="So workers dataloader")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--patience", type=int, default=4, 
                        help="So epoch cho doi Early Stopping truoc khi dung (mac dinh: 4, dat 0 de tat)")
    parser.add_argument("--early_stop_metric", type=str, default="val_loss", choices=["val_loss", "val_acc"],
                        help="Metric theo doi cho Early Stopping: val_loss (khuyen dung) hoac val_acc")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "="*70)
    print(f" HUẤN LUYỆN MÔ HÌNH: {args.model.upper()}")
    print(f" Thiết bị: {device} | Batch Size: {args.batch_size} | Epochs: {args.epochs}")
    print(f" Kỹ thuật Augmentation: {'STRONG (Domain Generalization)' if args.strong_aug else 'STANDARD BASELINE'}")
    print("="*70)

    # 1. Khoi tao DataLoaders
    loaders = build_dataloaders(
        processed_dir=args.processed_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        strong_aug=args.strong_aug
    )
    
    if "train" not in loaders or "val" not in loaders:
        print("[-] Khong tim thay tap train/val trong data/processed. Hay chay scripts/prepare_datasets.py truoc!")
        return

    # Doc so lop
    classes_json = Path(args.processed_dir) / "classes.json"
    with open(classes_json, "r", encoding="utf-8") as f:
        classes_data = json.load(f)
    num_classes = classes_data["num_classes"]
    print(f"[+] So luong lop phan loai: {num_classes}")

    # 2. Khoi tao Model
    model = get_model(args.model, num_classes=num_classes, pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # Label smoothing giup chong overconfidence
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=torch.cuda.is_available())

    save_dir = Path(args.save_dir) / f"{args.model}_{'strongaug' if args.strong_aug else 'standard'}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Khoi tao Early Stopping
    early_stopper = None
    if args.patience > 0:
        early_mode = "min" if args.early_stop_metric == "val_loss" else "max"
        early_stopper = EarlyStopping(patience=args.patience, min_delta=1e-4, mode=early_mode)
        print(f"[+] Bat Early Stopping: Theo doi '{args.early_stop_metric}' | Patience = {args.patience} epochs")
    else:
        print("[+] Early Stopping: TAT (chay du so epochs)")

    best_val_acc = 0.0
    best_cross_acc = 0.0
    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "cross_acc": []
    }

    print("\n[*] Bat dau huan luyen...")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        train_loss, train_acc = train_epoch(model, loaders["train"], criterion, optimizer, scaler, device)
        val_loss, val_acc = evaluate(model, loaders["val"], criterion, device)
        scheduler.step()
        
        # Danh gia ngay tren Cross-Domain PlantDoc neu co
        cross_acc = 0.0
        if "test_crossdomain" in loaders:
            _, cross_acc = evaluate(model, loaders["test_crossdomain"], criterion, device)
            
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["cross_acc"].append(cross_acc)
        
        time_epoch = time.time() - epoch_start
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({time_epoch:.1f}s) | "
              f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.2f}% | "
              f"Cross-Domain (PlantDoc): {cross_acc:.2f}%")

        # Luu best checkpoint theo in-domain validation
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "cross_acc": cross_acc,
                "args": vars(args)
            }, save_dir / "best_indomain_model.pth")
            
        # Luu best checkpoint theo cross-domain generalization
        if cross_acc > best_cross_acc:
            best_cross_acc = cross_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_acc": val_acc,
                "cross_acc": cross_acc,
                "args": vars(args)
            }, save_dir / "best_crossdomain_model.pth")

        # Kiem tra Early Stopping de ngat training som neu khong con tien bo
        if early_stopper:
            metric_val = val_loss if args.early_stop_metric == "val_loss" else val_acc
            if early_stopper(metric_val):
                print("\n" + "!"*70)
                print(f" [EARLY STOPPING KÍCH HOẠT TẠI EPOCH {epoch:02d}]")
                print(f" -> '{args.early_stop_metric}' không cải thiện trong {args.patience} epochs liên tiếp.")
                print(f" -> Tự động dừng huấn luyện để chống Overfitting và tiết kiệm GPU.")
                print("!"*70)
                break

    total_time = (time.time() - start_time) / 60.0
    print("\n" + "="*70)
    print(f" HUẤN LUYỆN HOÀN TẤT ({total_time:.2f} phút)!")
    print(f" - Best In-Domain Validation Acc (PlantVillage): {best_val_acc:.2f}%")
    print(f" - Best Cross-Domain Test Acc (PlantDoc):        {best_cross_acc:.2f}%")
    print(f" - Luu checkpoints tai: {save_dir.resolve()}")
    print("="*70)

    # Ghi log lich su huan luyen
    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

if __name__ == "__main__":
    main()
