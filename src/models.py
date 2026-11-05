"""
Module dinh nghia va khoi tao Model Factory cho cac kien truc:
- Baselines: ResNet-50, MobileNetV2
- Mo hinh hien dai (High Generalization): ConvNeXt-Tiny, Swin Transformer, EfficientNetV2
Su dung timm (PyTorch Image Models) va torchvision.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models
import timm

def get_model(model_name: str, num_classes: int = 38, pretrained: bool = True):
    """
    Khoi tao mo hinh phan loai.
    Cac mo hinh ho tro:
    - 'resnet50': Baseline pho bien
    - 'mobilenet_v2': Baseline sieu nhe, toi uu edge devices
    - 'convnext_tiny': Modern CNN (de xuat ngon hon, chong domain shift cuc tot)
    - 'swin_tiny': Vision Transformer voi Hierarchical Shifted Windows
    - 'efficientnet_b0' / 'efficientnet_v2_s': Can bang giua toc do va do chinh xac
    """
    model_name_lower = model_name.lower().replace("-", "_")
    
    if model_name_lower == "resnet50":
        weights = tv_models.ResNet50_Weights.DEFAULT if pretrained else None
        model = tv_models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    elif model_name_lower == "mobilenet_v2":
        weights = tv_models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = tv_models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    # Su dung timm cho cac kien truc Vision Transformers va Modern ConvNets
    elif "convnext" in model_name_lower:
        timm_name = "convnext_tiny" if model_name_lower == "convnext_tiny" else model_name
        model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
        return model

    elif "swin" in model_name_lower:
        timm_name = "swin_tiny_patch4_window7_224" if model_name_lower == "swin_tiny" else model_name
        model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
        return model

    elif "efficientnet" in model_name_lower:
        timm_name = "tf_efficientnetv2_s.in21k_ft_in1k" if "v2" in model_name_lower else "efficientnet_b0"
        model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
        return model

    else:
        # Fallback qua timm generic
        try:
            model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
            return model
        except Exception as e:
            raise ValueError(f"Khong ho tro kien truc '{model_name}'. Loi: {e}")

if __name__ == "__main__":
    # Test khoi tao
    for name in ["resnet50", "mobilenet_v2", "convnext_tiny", "swin_tiny"]:
        m = get_model(name, num_classes=38, pretrained=False)
        x = torch.randn(2, 3, 224, 224)
        out = m(x)
        print(f"Model {name} Output Shape: {out.shape}")
