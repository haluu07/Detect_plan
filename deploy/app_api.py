"""
API Backend su dung FastAPI de phuc vu cho Mobile App / Web App:
- Nhan anh tu Camera dien thoai (Upload file multipart/form-data)
- Model AI (ConvNeXt-Tiny Domain Adapted) chan doan benh
- Tra ve ket qua JSON sach se gom: Ten cay, Ten benh tieng Viet & Anh, Do tu tin, va Huong dan dieu tri!
"""

import io
import json
import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from models import get_model
from dataset import get_test_transforms

app = FastAPI(
    title="Plant Disease AI Diagnosis API",
    description="API Chẩn Đoán Bệnh Cây Trồng Phục Vụ Ứng Dụng Nông Nghiệp Thông Minh",
    version="1.0.0"
)

# Cho phep cac App Mobile / Web goi API khong bi chan CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tu dien tra cuu y nghia benh va huong dan phong tru
DISEASE_INFO = {
    "Apple___Apple_scab": {"vi_name": "Bệnh vảy táo", "advice": "Cắt tỉa lá bệnh, dọn sạch lá rụng và phun thuốc diệt nấm Captan hoặc Mancozeb."},
    "Apple___Black_rot": {"vi_name": "Bệnh thối đen táo", "advice": "Loại bỏ cành khô và quả bị bệnh, phun thuốc bảo vệ thực vật gốc đồng."},
    "Apple___Cedar_apple_rust": {"vi_name": "Bệnh rỉ sắt táo", "advice": "Cách ly khỏi cây bách xù xung quanh, phun thuốc phòng nấm Myclobutanil."},
    "Apple___healthy": {"vi_name": "Lá táo khỏe mạnh", "advice": "Cây phát triển tốt, duy trì chế độ tưới nước và dinh dưỡng hợp lý."},
    "Corn_(maize)___Common_rust_": {"vi_name": "Bệnh rỉ sắt ngô", "advice": "Phun thuốc trừ nấm chứa hoạt chất Azoxystrobin hoặc Pyraclostrobin."},
    "Corn_(maize)___Northern_Leaf_Blight": {"vi_name": "Bệnh cháy lá ngô phía Bắc", "advice": "Luân canh cây trồng, cày vùi tàn dư sau thu hoạch."},
    "Pepper,_bell___Bacterial_spot": {"vi_name": "Bệnh đốm vi khuẩn ớt chuông", "advice": "Hạn chế tưới phun mưa lên tán lá, phun thuốc gốc đồng phòng trừ."},
    "Pepper,_bell___healthy": {"vi_name": "Lá ớt khỏe mạnh", "advice": "Cây sinh trưởng tốt, bổ sung kali và canxi định kỳ."},
    "Potato___Early_blight": {"vi_name": "Bệnh đốm sớm khoai tây", "advice": "Duy trì độ ẩm đất hợp lý, phun thuốc trừ nấm chứa Chlorothalonil."},
    "Potato___Late_blight": {"vi_name": "Bệnh mốc sương / sương mai khoai tây", "advice": "Cực kỳ nguy hiểm! Cắt bỏ phần nhiễm bệnh và phun Metalaxyl ngay lập tức."},
    "Tomato___Early_blight": {"vi_name": "Bệnh đốm vòng cà chua", "advice": "Tỉa bỏ các lá già sát mặt đất, phun thuốc trừ nấm sinh học hoặc Mancozeb."},
    "Tomato___Late_blight": {"vi_name": "Bệnh sương mai cà chua", "advice": "Cách ly luống bệnh, phun thuốc phòng trừ có hoạt chất Dimethomorph."},
    "Tomato___healthy": {"vi_name": "Lá cà chua khỏe mạnh", "advice": "Cây đang phát triển rất tốt, duy trì kiểm tra định kỳ."}
}

# Khoi tao model toan cuc
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = []
MODEL = None
TRANSFORM = get_test_transforms(img_size=224)

@app.on_event("startup")
def load_ai_model():
    global MODEL, CLASSES
    base_dir = Path(__file__).resolve().parent.parent
    classes_path = base_dir / "data" / "processed" / "classes.json"
    
    with open(classes_path, "r", encoding="utf-8") as f:
        CLASSES = json.load(f)["classes"]

    # Uu tien nạp model da Domain Adaptation
    ckpt_path = base_dir / "checkpoints" / "convnext_domain_adaptation" / "best_adaptation_model.pth"
    if not ckpt_path.exists():
        ckpt_path = base_dir / "checkpoints" / "convnext_tiny_strongaug" / "best_indomain_model.pth"

    MODEL = get_model("convnext_tiny", num_classes=len(CLASSES), pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    MODEL.load_state_dict(ckpt.get("model_state_dict", ckpt))
    MODEL.to(DEVICE)
    MODEL.eval()
    print(f"[API] Đã nạp thành công mô hình AI ({ckpt_path.name}) trên thiết bị {DEVICE}!")

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "Plant Disease Diagnosis AI Engine",
        "device": str(DEVICE),
        "total_classes": len(CLASSES)
    }

@app.post("/predict")
async def predict_plant_disease(file: UploadFile = File(...)):
    """
    Endpoint nhan file anh tu Camera cua App va tra ve ket qua chan doan
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File tải lên bắt buộc phải là hình ảnh (jpg, png)!")

    # Doc anh
    content = await file.read()
    try:
        img_pil = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Không thể đọc file ảnh: {str(e)}")

    # Du doan AI
    input_tensor = TRANSFORM(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = MODEL(input_tensor)
        probs = F.softmax(logits, dim=1)[0]

    top_prob, top_idx = torch.topk(probs, k=3)
    
    top1_class = CLASSES[top_idx[0].item()]
    top1_conf = round(top_prob[0].item() * 100, 2)

    # Tra cuu thong tin benh
    info = DISEASE_INFO.get(top1_class, {
        "vi_name": top1_class.replace("___", " - ").replace("_", " "),
        "advice": "Hãy duy trì chăm sóc cây trồng và tham khảo ý kiến kỹ sư nông nghiệp địa phương nếu vết bệnh lan rộng."
    })

    # Danh sach Top 3
    top3 = []
    for p, idx in zip(top_prob, top_idx):
        cls_name = CLASSES[idx.item()]
        top3.append({
            "class_raw": cls_name,
            "disease_name_vi": DISEASE_INFO.get(cls_name, {}).get("vi_name", cls_name),
            "confidence_percent": round(p.item() * 100, 2)
        })

    is_healthy = "healthy" in top1_class.lower()

    return {
        "success": True,
        "diagnosis": {
            "disease_code": top1_class,
            "disease_name_vi": info["vi_name"],
            "confidence_percent": top1_conf,
            "is_healthy": is_healthy,
            "status": "Khỏe mạnh" if is_healthy else "Phát hiện sâu bệnh",
            "treatment_recommendation": info["advice"]
        },
        "top3_candidates": top3
    }

if __name__ == "__main__":
    uvicorn.run("app_api:app", host="0.0.0.0", port=8000, reload=False)
