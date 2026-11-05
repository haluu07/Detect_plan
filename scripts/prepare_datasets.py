"""
Script chuan bi, chuan hoa va tao bo chia tap Train/Val/Test cho:
1. PlantVillage (In-Domain): 70% train, 15% val, 15% test
2. PlantDoc (Cross-Domain): Align nhan sang class ID cua PlantVillage va xuat file metadata CSV

Dau ra:
- data/processed/plantvillage_train.csv
- data/processed/plantvillage_val.csv
- data/processed/plantvillage_test.csv
- data/processed/plantdoc_cross_domain_test.csv
- data/processed/classes.json
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "class_mapping.json"
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}

def find_plantvillage_root(raw_dir: Path) -> Path:
    """Tim thu muc thuc su chua 38 folder lop cua PlantVillage"""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc: {raw_dir}")
        
    candidates = [raw_dir] + list(raw_dir.glob("**/"))
    for cand in candidates:
        subdirs = [d for d in cand.iterdir() if d.is_dir() and not d.name.startswith(".")]
        # Kiem tra xem co phai thu muc chua cac lop benh cay khong
        has_apple = any("Apple" in d.name for d in subdirs)
        has_tomato = any("Tomato" in d.name for d in subdirs)
        if (has_apple or has_tomato) and len(subdirs) >= 15:
            # Neu co subfolder 'color', uu tien dung 'color' (khong dung grayscale/segmented neu muon test tong quat)
            color_sub = cand / "color"
            if color_sub.exists() and color_sub.is_dir():
                return color_sub
            return cand
    return raw_dir

def find_plantdoc_root(raw_dir: Path) -> Path:
    """Tim thu muc chua cac anh cua PlantDoc"""
    if not raw_dir.exists():
        raise FileNotFoundError(f"Khong tim thay thu muc: {raw_dir}")
        
    candidates = [raw_dir] + list(raw_dir.glob("**/"))
    for cand in candidates:
        subdirs = [d for d in cand.iterdir() if d.is_dir() and not d.name.startswith(".")]
        has_leaf = any("leaf" in d.name.lower() or "blight" in d.name.lower() for d in subdirs)
        if has_leaf and len(subdirs) >= 10:
            return cand
    return raw_dir

def prepare_plantvillage(pv_root: Path, output_dir: Path, val_ratio=0.15, test_ratio=0.15, seed=42):
    print(f"\n[*] Dang quet PlantVillage tai: {pv_root}")
    class_dirs = sorted([d for d in pv_root.iterdir() if d.is_dir() and not d.name.startswith(".")])
    
    classes = [d.name for d in class_dirs]
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    records = []
    for cls_name in classes:
        cls_dir = pv_root / cls_name
        cls_id = class_to_idx[cls_name]
        for img_path in cls_dir.iterdir():
            if img_path.is_file() and img_path.suffix in VALID_IMAGE_EXTENSIONS:
                records.append({
                    "image_path": str(img_path.resolve()),
                    "label_name": cls_name,
                    "label_idx": cls_id,
                    "domain": "PlantVillage"
                })
                
    df = pd.DataFrame(records)
    print(f"[OK] Tim thay tong cong {len(df)} anh PlantVillage tren {len(classes)} lop.")
    
    # Train / Val / Test split stratified
    train_df, temp_df = train_test_split(
        df, test_size=(val_ratio + test_ratio), random_state=seed, stratify=df["label_idx"]
    )
    val_weight = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1.0 - val_weight), random_state=seed, stratify=temp_df["label_idx"]
    )
    
    print(f"    - Train set: {len(train_df)} anh")
    print(f"    - Val set:   {len(val_df)} anh")
    print(f"    - Test set:  {len(test_df)} anh (In-Domain Evaluation)")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "plantvillage_train.csv", index=False)
    val_df.to_csv(output_dir / "plantvillage_val.csv", index=False)
    test_df.to_csv(output_dir / "plantvillage_test.csv", index=False)
    
    # Luu class mapping goc
    with open(output_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump({
            "classes": classes,
            "class_to_idx": class_to_idx,
            "num_classes": len(classes)
        }, f, indent=2, ensure_ascii=False)
        
    return class_to_idx

def prepare_plantdoc(pd_root: Path, output_dir: Path, class_to_idx: dict, mapping_config: dict):
    print(f"\n[*] Dang quet PlantDoc tai: {pd_root}")
    pd_map = mapping_config.get("plantdoc_to_plantvillage", {})
    
    # Chuẩn hóa key mapping (chữ thường, bỏ dấu gạch/khoảng trắng) để match linh hoạt
    normalized_map = {}
    for k, v in pd_map.items():
        clean_k = k.lower().replace("_", " ").strip()
        normalized_map[clean_k] = v
        
    records = []
    unmapped_folders = set()
    mapped_count = 0
    
    # Duyet tat ca cac thu muc con trong PlantDoc
    for folder in pd_root.glob("**/"):
        if not folder.is_dir():
            continue
        folder_name = folder.name
        clean_folder = folder_name.lower().replace("_", " ").strip()
        
        target_pv_class = None
        if folder_name in pd_map:
            target_pv_class = pd_map[folder_name]
        elif clean_folder in normalized_map:
            target_pv_class = normalized_map[clean_folder]
        else:
            # Thu tim partial match
            for k, v in normalized_map.items():
                if k in clean_folder or clean_folder in k:
                    target_pv_class = v
                    break
                    
        if target_pv_class and target_pv_class in class_to_idx:
            pv_idx = class_to_idx[target_pv_class]
            for img_path in folder.iterdir():
                if img_path.is_file() and img_path.suffix in VALID_IMAGE_EXTENSIONS:
                    records.append({
                        "image_path": str(img_path.resolve()),
                        "plantdoc_class": folder_name,
                        "label_name": target_pv_class,
                        "label_idx": pv_idx,
                        "domain": "PlantDoc"
                    })
                    mapped_count += 1
        else:
            if any(folder.glob("*.jpg")) or any(folder.glob("*.png")):
                unmapped_folders.add(folder_name)
                
    if records:
        df_pd = pd.DataFrame(records)
        output_csv = output_dir / "plantdoc_cross_domain_test.csv"
        df_pd.to_csv(output_csv, index=False)
        print(f"[OK] Da map thanh cong {len(df_pd)} anh PlantDoc vao {len(df_pd['label_idx'].unique())} lop giao nhau voi PlantVillage.")
        print(f"    File ket qua: {output_csv}")
    else:
        print("[!] Canh bao: Khong map duoc anh PlantDoc nao. Kiem tra lai thu muc hoac configs/class_mapping.json")
        
    if unmapped_folders:
        print(f"[INFO] Cac thu muc PlantDoc chua duoc map (hoac la thu muc goc): {list(unmapped_folders)[:5]}")

def main():
    parser = argparse.ArgumentParser(description="Tien xu ly va mapping du lieu")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR))
    parser.add_argument("--output_dir", type=str, default=str(PROCESSED_DIR))
    args = parser.parse_args()
    
    data_path = Path(args.data_dir)
    output_path = Path(args.output_dir)
    
    pv_raw = data_path / "plantvillage"
    pd_raw = data_path / "plantdoc"
    
    if not pv_raw.exists():
        print(f"[-] Chua co du lieu PlantVillage tai {pv_raw}. Hay chay `python scripts/download_datasets.py` truoc!")
        return
        
    pv_root = find_plantvillage_root(pv_raw)
    class_to_idx = prepare_plantvillage(pv_root, output_path)
    
    if pd_raw.exists():
        pd_root = find_plantdoc_root(pd_raw)
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            mapping_config = json.load(f)
        prepare_plantdoc(pd_root, output_path, class_to_idx, mapping_config)
    else:
        print(f"[!] Chua thay thu muc PlantDoc tai {pd_raw}. Bo qua buoc tao tap Cross-domain PlantDoc.")

if __name__ == "__main__":
    main()
