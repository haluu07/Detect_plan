"""
Module dinh nghia PyTorch Dataset va Data Transforms cho PlantVillage & PlantDoc.
Ho tro ca Standard Augmentations va Strong Augmentations (Domain Generalization).
"""

import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# Thong so chuan hoa ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def get_train_transforms(img_size=224, strong_aug=False):
    """
    Data Augmentation cho tap huan luyen:
    - strong_aug=False: Baseline augmentation (RandomResizedCrop, RandomHorizontalFlip)
    - strong_aug=True: Strong Domain Generalization Augmentation (RandAugment, ColorJitter, RandomErasing)
      giup giam thieu Shortcut Learning tren nen phong lab.
    """
    if strong_aug:
        return T.Compose([
            T.Resize((int(img_size * 1.15), int(img_size * 1.15))),
            T.RandomCrop((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(degrees=30),
            # Pha vo su dong nhat ve anh sang trong phong lab
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            T.RandomErasing(p=0.2, scale=(0.02, 0.2), value="random")
        ])
    else:
        return T.Compose([
            T.Resize((int(img_size * 1.15), int(img_size * 1.15))),
            T.RandomCrop((img_size, img_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

def get_test_transforms(img_size=224):
    """Transform chuan cho Validation va Test (ca In-domain va Cross-domain)"""
    return T.Compose([
        T.Resize((int(img_size * 1.14), int(img_size * 1.14))),
        T.CenterCrop((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

class PlantCSVDataset(Dataset):
    """
    Dataset doc tu metadata CSV tao boi prepare_datasets.py.
    """
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        
        # Loc cac file anh khong ton tai hoac hong neu co
        valid_indices = []
        for idx, row in self.df.iterrows():
            if os.path.exists(row["image_path"]):
                valid_indices.append(idx)
        if len(valid_indices) < len(self.df):
            print(f"[CANH BAO] Loc bo {len(self.df) - len(valid_indices)} anh khong ton tai trong {csv_file}")
            self.df = self.df.iloc[valid_indices].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]
        label = int(row["label_idx"])
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            # Fallback neu anh loi
            print(f"[LOI ANH] Khong doc duoc {img_path}: {e}")
            image = Image.new("RGB", (224, 224), (0, 0, 0))
            
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label, img_path

def build_dataloaders(
    processed_dir="data/processed",
    batch_size=64,
    num_workers=4,
    img_size=224,
    strong_aug=False
):
    processed_path = Path(processed_dir)
    
    train_csv = processed_path / "plantvillage_train.csv"
    val_csv = processed_path / "plantvillage_val.csv"
    test_csv = processed_path / "plantvillage_test.csv"
    cross_csv = processed_path / "plantdoc_cross_domain_test.csv"
    
    train_tf = get_train_transforms(img_size=img_size, strong_aug=strong_aug)
    eval_tf = get_test_transforms(img_size=img_size)
    
    loaders = {}
    
    if train_csv.exists():
        train_ds = PlantCSVDataset(train_csv, transform=train_tf)
        loaders["train"] = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
        
    if val_csv.exists():
        val_ds = PlantCSVDataset(val_csv, transform=eval_tf)
        loaders["val"] = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
    if test_csv.exists():
        test_ds = PlantCSVDataset(test_csv, transform=eval_tf)
        loaders["test_indomain"] = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
    if cross_csv.exists():
        cross_ds = PlantCSVDataset(cross_csv, transform=eval_tf)
        loaders["test_crossdomain"] = DataLoader(
            cross_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
    return loaders
