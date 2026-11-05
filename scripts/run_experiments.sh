#!/usr/bin/env bash
# ==============================================================================
# SCRIPT CHẠY TOÀN BỘ EXPERIMENT TRÊN LINUX GPU SERVER (SLURM / DEDICATED SERVER)
# ==============================================================================

set -e # Dung ngay neu co loi xay ra

echo "======================================================================"
echo " PLANT DISEASE RECOGNITION AND CROSS-DOMAIN GENERALIZATION EXPERIMENT "
echo "======================================================================"

# Kiem tra GPU
nvidia-smi || echo "[Canh bao] Khong phat hien thay GPU Nvidia, se chay bang CPU."

# 1. Cai dat thu vien neu can
echo -e "\n[Buoc 1] Kiem tra va cai dat thu vien..."
pip install -r requirements.txt

# 2. Download du lieu
echo -e "\n[Buoc 2] Tai du lieu PlantVillage va PlantDoc..."
python scripts/download_datasets.py

# 3. Tien xu ly du lieu va tao Class Mapping
echo -e "\n[Buoc 3] Tien xu ly du lieu & tao cross-domain label alignment..."
python scripts/prepare_datasets.py

# 4. Chay Baseline 1: Traditional ML (HOG + SVM)
echo -e "\n[Buoc 4] Chay Baseline 1: HOG + SVM..."
python src/baseline_hog_svm.py --max_train_samples 15000

# 5. Chay Baseline 2: ResNet-50 (Standard Augmentations)
echo -e "\n[Buoc 5] Train Baseline Deep Learning: ResNet-50 (Standard Aug)..."
python src/train.py --model resnet50 --epochs 15 --batch_size 64 --lr 1e-4 --patience 4

# 6. Chay Baseline 3: MobileNetV2 (Nhe, toi uu edge)
echo -e "\n[Buoc 6] Train MobileNetV2 (Standard Aug)..."
python src/train.py --model mobilenet_v2 --epochs 15 --batch_size 64 --lr 1e-4 --patience 4

# 7. Chay Mo hinh de xuat: ConvNeXt-Tiny + Strong Domain Generalization Augmentations
echo -e "\n[Buoc 7] Train Proposed Model: ConvNeXt-Tiny (Strong Augmentations)..."
python src/train.py --model convnext_tiny --epochs 20 --batch_size 64 --lr 1e-4 --strong_aug --patience 5

# 8. Danh gia toan dien Cross-Domain tren PlantDoc
echo -e "\n[Buoc 8] Danh gia Cross-Domain tren PlantDoc..."
python src/evaluate_cross_domain.py --checkpoint checkpoints/resnet50_standard/best_indomain_model.pth --model resnet50
python src/evaluate_cross_domain.py --checkpoint checkpoints/convnext_tiny_strongaug/best_indomain_model.pth --model convnext_tiny

echo "======================================================================"
echo " TOAN BO THUC NGHIEM HOAN TAT! KET QUA DUOC LUU TAI /reports"
echo "======================================================================"
