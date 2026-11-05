#!/usr/bin/env bash
# ==============================================================================
# SCRIPT CHẠY TOÀN BỘ EXPERIMENT TRÊN LINUX GPU SERVER (SLURM / DEDICATED SERVER)
# ==============================================================================

set -e # Dung ngay neu co loi xay ra

echo "======================================================================"
echo " PLANT DISEASE RECOGNITION AND CROSS-DOMAIN GENERALIZATION EXPERIMENT "
echo "======================================================================"

# CAU HINH GPU VA BATCH SIZE (Phu hop voi server hien tai)
# GPU 0 dang full 98C va het VRAM, nen chon GPU 1 (RTX A4000)
GPU_ID=1
BATCH_SIZE=32

# Kiem tra GPU NVIDIA
echo -e "\n[*] Kiem tra trang thai GPU:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo -e "\n[LỖI] Không tìm thấy lệnh nvidia-smi hoặc máy chủ không có GPU NVIDIA!"
    echo "Hệ thống yêu cầu GPU để huấn luyện. Nếu vẫn muốn chạy CPU, hãy thêm cờ --no_force_gpu vào các lệnh train."
    exit 1
fi

# 1. Cai dat thu vien neu can
echo -e "\n[Buoc 1] Kiem tra va cai dat thu vien..."
pip install -r requirements.txt

# 2. Download va tien xu ly (neu chua co)
if [ ! -f "data/processed/plantvillage_train.csv" ]; then
    echo -e "\n[Buoc 2] Tai du lieu PlantVillage va PlantDoc..."
    python scripts/download_datasets.py

    echo -e "\n[Buoc 3] Tien xu ly du lieu & tao cross-domain label alignment..."
    python scripts/prepare_datasets.py
else
    echo -e "\n[+] Du lieu da duoc tien xu ly san trong data/processed/, bo qua buoc download va prepare."
fi

# 4. Chay Baseline 1: Traditional ML (HOG + SVM)
echo -e "\n[Buoc 4] Chay Baseline 1: HOG + SVM..."
python src/baseline_hog_svm.py --max_train_samples 15000

# 5. Chay Baseline 2: ResNet-50 (Standard Augmentations)
echo -e "\n[Buoc 5] Train Baseline Deep Learning: ResNet-50 tren GPU $GPU_ID..."
python src/train.py --model resnet50 --epochs 15 --batch_size $BATCH_SIZE --lr 1e-4 --patience 4 --gpu_id $GPU_ID

# 6. Chay Baseline 3: MobileNetV2 (Nhe, toi uu edge)
echo -e "\n[Buoc 6] Train MobileNetV2 tren GPU $GPU_ID..."
python src/train.py --model mobilenet_v2 --epochs 15 --batch_size $BATCH_SIZE --lr 1e-4 --patience 4 --gpu_id $GPU_ID

# 7. Chay Mo hinh de xuat: ConvNeXt-Tiny + Strong Domain Generalization Augmentations
echo -e "\n[Buoc 7] Train Proposed Model: ConvNeXt-Tiny tren GPU $GPU_ID..."
python src/train.py --model convnext_tiny --epochs 20 --batch_size $BATCH_SIZE --lr 1e-4 --strong_aug --patience 5 --gpu_id $GPU_ID

# 8. Danh gia toan dien Cross-Domain tren PlantDoc
echo -e "\n[Buoc 8] Danh gia Cross-Domain tren PlantDoc..."
python src/evaluate_cross_domain.py --checkpoint checkpoints/resnet50_standard/best_indomain_model.pth --model resnet50 --gpu_id $GPU_ID
python src/evaluate_cross_domain.py --checkpoint checkpoints/convnext_tiny_strongaug/best_indomain_model.pth --model convnext_tiny --gpu_id $GPU_ID

echo "======================================================================"
echo " TOAN BO THUC NGHIEM HOAN TAT! KET QUA DUOC LUU TAI /reports"
echo "======================================================================"
