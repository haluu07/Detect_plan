"""
Script download dữ liệu PlantVillage & PlantDoc phục vụ nghiên cứu Cross-Domain Generalization.
Hỗ trợ:
1. Tự động tải qua Kaggle API (khuyên dùng khi chạy trên Server/Colab)
2. Hướng dẫn tải trực tiếp qua link nếu không dùng Kaggle API
3. Giải nén và tự động chuẩn hóa cấu trúc thư mục data/
"""

import os
import sys
import zipfile
import shutil
import argparse
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def download_kaggle_dataset(dataset_slug: str, output_dir: Path, zip_name: str):
    """Tải và giải nén Kaggle dataset bằng lệnh kaggle CLI hoặc python kaggle API"""
    ensure_dir(output_dir)
    print(f"\n[+] Dang tai dataset '{dataset_slug}' tu Kaggle vao {output_dir}...")
    
    # Kiểm tra xem kaggle CLI hoặc API có sẵn không
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(dataset_slug, path=str(output_dir), unzip=True)
        print(f"[OK] Da tai va giai nen thanh cong {dataset_slug}")
        return True
    except Exception as e:
        print(f"[-] Kaggle Python API gap loi: {e}")
        print("    Thu goi kaggle CLI qua he thong...")
        ret = os.system(f"kaggle datasets download -d {dataset_slug} -p \"{output_dir}\" --unzip")
        if ret == 0:
            print(f"[OK] Da tai va giai nen thanh cong bang Kaggle CLI.")
            return True
        else:
            print("[-] Khong the tai tu dong qua Kaggle API/CLI.")
            print("    Luu y: Can co file ~/.kaggle/kaggle.json (Linux) hoac C:\\Users\\<User>\\.kaggle\\kaggle.json (Windows)")
            return False

def download_plantvillage(target_dir: Path):
    """
    PlantVillage: ~54.000 anh, 38 lop
    """
    pv_dir = target_dir / "plantvillage"
    if pv_dir.exists() and len(list(pv_dir.glob("*/*"))) > 1000:
        print(f"[INFO] PlantVillage da ton tai tai: {pv_dir}")
        return

    print("\n" + "="*60)
    print(" BẮT ĐẦU TẢI PLANTVILLAGE DATASET (~54.000 ảnh, 38 lớp)")
    print("="*60)
    
    # 1. Thu Kaggle API neu co token
    success = download_kaggle_dataset("abdallahalidev/plantvillage-dataset", pv_dir, "plantvillage.zip")
    
    # 2. Neu Kaggle khong duoc, clone tu repo GitHub chinh chu cua spMohanty (depth=1)
    if not success:
        print("[+] Thu clone repository PlantVillage tu GitHub (spMohanty/PlantVillage-Dataset)...")
        git_target = target_dir / "PlantVillage_git"
        ret = os.system(f"git clone --depth 1 https://github.com/spMohanty/PlantVillage-Dataset.git \"{git_target}\"")
        if ret == 0:
            color_dir = git_target / "raw" / "color"
            if color_dir.exists():
                ensure_dir(pv_dir)
                for class_folder in color_dir.iterdir():
                    if class_folder.is_dir():
                        shutil.move(str(class_folder), str(pv_dir / class_folder.name))
                print(f"[OK] Da chuyen anh tu repo GitHub vao {pv_dir}")
                return

        print("\n" + "!"*60)
        print("HƯỚNG DẪN TẢI THỦ CÔNG PLANTVILLAGE:")
        print("CÁCH 1 (Nhanh nhat tren Linux qua Git):")
        print(f"  git clone --depth 1 https://github.com/spMohanty/PlantVillage-Dataset.git")
        print(f"  mv PlantVillage-Dataset/raw/color/* \"{pv_dir.resolve()}/\"")
        print("CÁCH 2: Tải zip từ Kaggle:")
        print("  https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        print("!"*60 + "\n")

def download_plantdoc(target_dir: Path):
    """
    PlantDoc: Anh thuc dia (in-the-wild), dung danh gia Cross-Domain Generalization
    """
    pd_dir = target_dir / "plantdoc"
    if pd_dir.exists() and len(list(pd_dir.glob("*/*"))) > 200:
        print(f"[INFO] PlantDoc da ton tai tai: {pd_dir}")
        return

    print("\n" + "="*60)
    print(" BẮT ĐẦU TẢI PLANTDOC DATASET (Ảnh thực địa)")
    print("="*60)
    
    # Cach 1: Clone truc tiep tu GitHub tac gia (pratikkayal/PlantDoc-Dataset) - KHONG CAN KAGGLE TOKEN!
    print("[+] Dang clone PlantDoc tu GitHub chinh chu (pratikkayal/PlantDoc-Dataset)...")
    github_url = "https://github.com/pratikkayal/PlantDoc-Dataset.git"
    clone_target = target_dir / "PlantDoc_git"
    
    if clone_target.exists():
        shutil.rmtree(clone_target, ignore_errors=True)
        
    ret = os.system(f"git clone --depth 1 {github_url} \"{clone_target}\"")
    if ret == 0:
        print(f"[OK] Da clone thanh cong repo PlantDoc vao {clone_target}")
        ensure_dir(pd_dir)
        for sub in ["train", "test"]:
            src_sub = clone_target / sub
            if src_sub.exists():
                shutil.copytree(src_sub, pd_dir / sub, dirs_exist_ok=True)
        print(f"[OK] Da sao chep du lieu PlantDoc vao {pd_dir}")
        return
        
    # Cach 2: Kaggle dataset neu git gap loi
    success = download_kaggle_dataset("pratik2911/plantdoc-dataset", pd_dir, "plantdoc.zip")
    if not success:
        print("\n" + "!"*60)
        print("HƯỚNG DẪN TẢI THỦ CÔNG PLANTDOC:")
        print(f"  git clone --depth 1 https://github.com/pratikkayal/PlantDoc-Dataset.git")
        print(f"  cp -r PlantDoc-Dataset/train PlantDoc-Dataset/test \"{pd_dir.resolve()}/\"")
        print("!"*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Tool tai du lieu PlantVillage va PlantDoc")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR), help="Thu muc goc chua du lieu")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    ensure_dir(data_path)
    
    print(f"[INFO] Thu muc du lieu: {data_path.resolve()}")
    download_plantvillage(data_path)
    download_plantdoc(data_path)
    print("\n[HOAN TAT] Kiem tra du lieu san sang trong thu muc data/!")

if __name__ == "__main__":
    main()
