@echo off
REM ==============================================================================
REM SCRIPT CHAY TOAN BO EXPERIMENT TREN WINDOWS POWERSHELL / CMD
REM ==============================================================================

echo ======================================================================
echo  PLANT DISEASE RECOGNITION AND CROSS-DOMAIN GENERALIZATION EXPERIMENT
echo ======================================================================

REM 1. Cai dat thu vien
echo [Buoc 1] Cai dat dependencies...
pip install -r requirements.txt

REM 2. Download du lieu
echo [Buoc 2] Tai du lieu PlantVillage va PlantDoc...
python scripts/download_datasets.py

REM 3. Tien xu ly du lieu va mapping
echo [Buoc 3] Tien xu ly du lieu va tao cross-domain mapping...
python scripts/prepare_datasets.py

REM 4. Chay Baseline 1 (HOG + SVM)
echo [Buoc 4] Chay Baseline 1: HOG + SVM...
python src/baseline_hog_svm.py --max_train_samples 15000

REM 5. Train ResNet-50
echo [Buoc 5] Train ResNet-50 Baseline...
python src/train.py --model resnet50 --epochs 15 --batch_size 32 --lr 1e-4 --patience 4

REM 6. Train ConvNeXt-Tiny de xuat
echo [Buoc 6] Train ConvNeXt-Tiny (Strong Augmentations)...
python src/train.py --model convnext_tiny --epochs 20 --batch_size 32 --lr 1e-4 --strong_aug --patience 5

REM 7. Danh gia Cross-Domain
echo [Buoc 7] Danh gia Cross-Domain tren PlantDoc...
python src/evaluate_cross_domain.py --checkpoint checkpoints\resnet50_standard\best_indomain_model.pth --model resnet50
python src/evaluate_cross_domain.py --checkpoint checkpoints\convnext_tiny_strongaug\best_indomain_model.pth --model convnext_tiny

echo ======================================================================
echo  HOAN TAT THUC NGHIEM! KET QUA DUOC LUU TAI reports\
echo ======================================================================
pause
