"""
Baseline Co dien (Traditional ML): HOG (Histogram of Oriented Gradients) + Color Hist + SVM
Phuc vu viec so sanh voi Deep Learning va chung minh su sup do cua dac trung thu cong tren Cross-Domain.
"""

import os
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

def extract_features(img_path: str, img_size=(128, 128)):
    """
    Trich xuat ket hop:
    1. HOG features (Cau truc hinh hoc, vien gan la, dom benh)
    2. Color Histogram (Thong tin mau sac benh la trong khong gian HSV)
    """
    try:
        img = Image.open(img_path).convert("RGB").resize(img_size)
    except Exception:
        # Fallback anh rong neu file loi
        return np.zeros(128 + 36 * 3)

    img_np = np.array(img)
    
    # 1. Trich xuat HOG tren anh xam
    gray = np.dot(img_np[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    hog_feat = hog(
        gray,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )

    # 2. Trich xuat Color Histogram (3 channels RGB)
    hist_r, _ = np.histogram(img_np[:, :, 0], bins=16, range=(0, 256), density=True)
    hist_g, _ = np.histogram(img_np[:, :, 1], bins=16, range=(0, 256), density=True)
    hist_b, _ = np.histogram(img_np[:, :, 2], bins=16, range=(0, 256), density=True)
    color_feat = np.concatenate([hist_r, hist_g, hist_b])

    return np.concatenate([hog_feat, color_feat])

def process_dataframe(df: pd.DataFrame, max_samples: int = None, desc="Extracting"):
    if max_samples and len(df) > max_samples:
        # Stratified subsampling dung train_test_split de giu nguyen cac cot
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(df, train_size=max_samples, stratify=df["label_idx"], random_state=42)
        df = df.reset_index(drop=True)
        
    features = []
    labels = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        f = extract_features(row["image_path"])
        features.append(f)
        labels.append(int(row["label_idx"]))
        
    return np.array(features), np.array(labels)

def main():
    parser = argparse.ArgumentParser(description="Chay Baseline HOG + SVM")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--max_train_samples", type=int, default=15000, 
                        help="Gioi han so mau train de chay nhanh tren CPU (default: 15000, hoac None de lay het)")
    parser.add_argument("--save_model", type=str, default="checkpoints/hog_svm_baseline.joblib")
    args = parser.parse_args()

    processed_path = Path(args.processed_dir)
    train_csv = processed_path / "plantvillage_train.csv"
    test_csv = processed_path / "plantvillage_test.csv"
    cross_csv = processed_path / "plantdoc_cross_domain_test.csv"

    if not train_csv.exists() or not test_csv.exists():
        print(f"[-] Khong tim thay tap du lieu trong {processed_path}. Hay chay prepare_datasets.py!")
        return

    print("\n" + "="*70)
    print(" BASELINE 1: TRADITIONAL MACHINE LEARNING (HOG + COLOR HIST + SVM)")
    print("="*70)

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    print(f"[+] Trich xuat dac trung tap Train (max: {args.max_train_samples})...")
    X_train, y_train = process_dataframe(train_df, max_samples=args.max_train_samples, desc="Train Set")
    print(f"    Vector dac trung kich thuoc: {X_train.shape[1]}")

    print(f"[+] Trich xuat dac trung tap In-Domain Test (PlantVillage)...")
    X_test, y_test = process_dataframe(test_df, desc="In-Domain Test Set")

    print("\n[*] Dang huan luyen Linear SVM (Pipeline StandardScaler + LinearSVC)...")
    t0 = time.time()
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(dual=False, C=1.0, max_iter=2000, random_state=42))
    ])
    clf.fit(X_train, y_train)
    print(f"[OK] Huan luyen xong trong {time.time() - t0:.2f} giay.")

    # 1. Danh gia In-Domain tren PlantVillage Test Set
    y_pred_in = clf.predict(X_test)
    acc_in = accuracy_score(y_test, y_pred_in) * 100
    f1_in = f1_score(y_test, y_pred_in, average="macro") * 100
    print("\n" + "-"*50)
    print(f" KẾT QUẢ IN-DOMAIN (PlantVillage Test Set):")
    print(f" - Top-1 Accuracy: {acc_in:.2f}%")
    print(f" - Macro F1-Score: {f1_in:.2f}%")
    print("-"*50)

    # 2. Danh gia Cross-Domain tren PlantDoc
    if cross_csv.exists():
        cross_df = pd.read_csv(cross_csv)
        print(f"\n[+] Trich xuat dac trung tap Cross-Domain Test (PlantDoc - {len(cross_df)} anh)...")
        X_cross, y_cross = process_dataframe(cross_df, desc="Cross-Domain PlantDoc")
        
        y_pred_cross = clf.predict(X_cross)
        acc_cross = accuracy_score(y_cross, y_pred_cross) * 100
        f1_cross = f1_score(y_cross, y_pred_cross, average="macro") * 100
        
        print("\n" + "="*50)
        print(f" KẾT QUẢ CROSS-DOMAIN GENERALIZATION (PlantDoc Field Images):")
        print(f" - Top-1 Accuracy: {acc_cross:.2f}%")
        print(f" - Macro F1-Score: {f1_cross:.2f}%")
        print(f" - MỨC ĐỘ TỤT GIẢM (Performance Drop): -{(acc_in - acc_cross):.2f}%")
        print("="*50)
        print("=> Nhan xet: Dac trung thu cong HOG phu thuoc nang ne vao phan vien la va do dong nhat mau.")
        print("   Khi sang moi truong thuc dia nhieu tap am, do chinh xac tut doc nghiem trong!")
    else:
        print("[!] Khong tim thay file plantdoc_cross_domain_test.csv de kiem thu cheo.")

    # Luu checkpoint model
    save_path = Path(args.save_model)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, save_path)
    print(f"\n[SAVED] Da luu model SVM tai: {save_path}")

if __name__ == "__main__":
    main()
