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
    Kaggle slug pho bien nhat: 'abdallahalidev/plantvillage-dataset' hoac 'emmarex/plantdisease'
    """
    pv_dir = target_dir / "plantvillage"
    if pv_dir.exists() and len(list(pv_dir.glob("*/*"))) > 1000:
        print(f"[INFO] PlantVillage da ton tai tai: {pv_dir}")
        return

    print("\n" + "="*60)
    print(" BẮT ĐẦU TẢI PLANTVILLAGE DATASET (~54.000 ảnh, 38 lớp)")
    print("="*60)
    success = download_kaggle_dataset("abdallahalidev/plantvillage-dataset", pv_dir, "plantvillage.zip")
    
    if not success:
        print("\n" + "!"*60)
        print("HƯỚNG DẪN TẢI THỦ CÔNG PLANTVILLAGE:")
        print("1. Truy cap: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        print("   hoac: https://data.mendeley.com/datasets/tywbtsjrjv/1")
        print(f"2. Tai file zip ve va giai nen vao thu muc: {pv_dir.resolve()}")
        print("   Cau truc thu muc can dat duoc:")
        print(f"   {pv_dir.resolve()}/")
        print("       ├── Apple___Apple_scab/")
        print("       ├── Apple___Black_rot/")
        print("       └── ... (38 thu muc con ung voi 38 lop)")
        print("!"*60 + "\n")

def download_plantdoc(target_dir: Path):
    """
    PlantDoc: Anh thuc dia (in-the-wild), dung danh gia Cross-Domain Generalization
    Kaggle slug: 'pratik2911/plantdoc-dataset' hoac clone tu repo GitHub
    """
    pd_dir = target_dir / "plantdoc"
    if pd_dir.exists() and len(list(pd_dir.glob("*/*"))) > 200:
        print(f"[INFO] PlantDoc da ton tai tai: {pd_dir}")
        return

    print("\n" + "="*60)
    print(" BẮT ĐẦU TẢI PLANTDOC DATASET (Ảnh thực địa)")
    print("="*60)
    
    # Cach 1: Kaggle dataset
    success = download_kaggle_dataset("pratik2911/plantdoc-dataset", pd_dir, "plantdoc.zip")
    
    # Cach 2: Neu Kaggle khong duoc, clone truc tiep tu GitHub repo chinh chu cua tac gia (Singhal et al.)
    if not success:
        print("[+] Thu clone repository PlantDoc tu GitHub tac gia (pratik2911/PlantDoc-Dataset)...")
        github_url = "https://github.com/pratik2911/PlantDoc-Dataset.git"
        clone_target = target_dir / "PlantDoc_git"
        ret = os.system(f"git clone --depth 1 {github_url} \"{clone_target}\"")
        if ret == 0:
            print(f"[OK] Da clone thanh cong repo PlantDoc vao {clone_target}")
            # Tim thu muc anh train/test cua PlantDoc
            ensure_dir(pd_dir)
            for sub in ["train", "test"]:
                src_sub = clone_target / sub
                if src_sub.exists():
                    shutil.copytree(src_sub, pd_dir / sub, dirs_exist_ok=True)
            print(f"[OK] Da sao chep du lieu PlantDoc vao {pd_dir}")
            return
            
        print("\n" + "!"*60)
        print("HƯỚNG DẪN TẢI THỦ CÔNG PLANTDOC:")
        print("1. Truy cap: https://www.kaggle.com/datasets/pratik2911/plantdoc-dataset")
        print("   hoac GitHub: https://github.com/pratik2911/PlantDoc-Dataset")
        print(f"2. Giai nen toan bo vao: {pd_dir.resolve()}")
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
