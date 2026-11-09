"""
API Backend su dung FastAPI de phuc vu cho Mobile App / Web App:
- Nhan anh tu Camera dien thoai (Upload file multipart/form-data)
- Model AI (ConvNeXt-Tiny Domain Adapted) chan doan benh
- Tra ve ket qua JSON sach se gom: Ten cay, Ten benh tieng Viet & Anh, Do tu tin, va Huong dan dieu tri!
- Tu dong tao Public HTTPS Link (Cloudflare Tunnel) de Expo Go va Web truy cap tu moi noi!
- Co che tong hop xac suat benh hoc (Binary Disease Aggregation) tranh loi nhan dinh "Khoe manh" tren la co dom!
"""

import io
import json
import os
import re
import stat
import time
import urllib.request
import threading
import subprocess
import platform
import argparse
from pathlib import Path
from contextlib import asynccontextmanager
from PIL import Image
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from models import get_model
from dataset import get_test_transforms

# Thu muc chua template frontend
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

# Khoi tao bien toan cuc
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = []
MODEL = None
TRANSFORM = get_test_transforms(img_size=224)

# Tu dien tra cuu toan dien 38 lop benh cay trong (PlantVillage & PlantDoc)
DISEASE_INFO = {
    # Táo (Apple)
    "Apple___Apple_scab": {"vi_name": "Bệnh vảy táo (Apple Scab)", "advice": "Cắt tỉa cành thông thoáng, dọn sạch lá rụng dưới gốc và phun thuốc trừ nấm gốc Captan, Mancozeb hoặc Difenoconazole."},
    "Apple___Black_rot": {"vi_name": "Bệnh thối đen táo (Black Rot)", "advice": "Cắt bỏ cành khô và quả bị bệnh, phun thuốc bảo vệ thực vật gốc đồng (Copper Hydroxide) hoặc Thiophanate-methyl."},
    "Apple___Cedar_apple_rust": {"vi_name": "Bệnh rỉ sắt táo (Cedar Rust)", "advice": "Cách ly vườn táo khỏi cây bách xù xung quanh, phun thuốc phòng nấm Myclobutanil hoặc Tebuconazole."},
    "Apple___healthy": {"vi_name": "Lá táo khỏe mạnh", "advice": "Cây sinh trưởng tốt. Duy trì chế độ tưới tiêu và bón phân định kỳ."},

    # Việt quất (Blueberry)
    "Blueberry___healthy": {"vi_name": "Lá việt quất khỏe mạnh", "advice": "Cây phát triển bình thường. Giữ đất chua nhẹ (pH 4.5 - 5.5) và đủ độ ẩm."},

    # Anh đào (Cherry)
    "Cherry_(including_sour)___Powdery_mildew": {"vi_name": "Bệnh phấn trắng anh đào", "advice": "Tỉa thưa tán lá cho thông thoáng, phun thuốc trừ nấm chứa lưu huỳnh hữu cơ hoặc Myclobutanil."},
    "Cherry_(including_sour)___healthy": {"vi_name": "Lá anh đào khỏe mạnh", "advice": "Cây sinh trưởng tốt. Duy trì bón lót phân hữu cơ và thoát nước tốt."},

    # Ngô / Bắp (Corn)
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {"vi_name": "Bệnh đốm xám lá ngô (Gray Leaf Spot)", "advice": "Luân canh cây trồng, cày vùi tàn dư rơm rạ sau thu hoạch và phun thuốc trừ nấm nhóm Triazole hoặc Strobilurin."},
    "Corn_(maize)___Common_rust_": {"vi_name": "Bệnh rỉ sắt ngô (Common Rust)", "advice": "Phun thuốc trừ nấm Azoxystrobin hoặc Pyraclostrobin ngay khi phát hiện vết rỉ sắt đầu tiên."},
    "Corn_(maize)___Northern_Leaf_Blight": {"vi_name": "Bệnh cháy lá ngô phía Bắc", "advice": "Sử dụng giống kháng bệnh, thu dọn tàn dư vụ trước và phun Mancozeb khi bệnh chớm xuất hiện."},
    "Corn_(maize)___healthy": {"vi_name": "Lá ngô khỏe mạnh", "advice": "Cây phát triển xanh tốt, cung cấp đủ đạm và nước trong giai đoạn trổ cờ."},

    # Nho (Grape)
    "Grape___Black_rot": {"vi_name": "Bệnh thối đen nho (Black Rot)", "advice": "Cắt tỉa chùm quả và lá bệnh, phun phòng ngừa bằng Mancozeb hoặc Ziram từ đầu mùa xuân."},
    "Grape___Esca_(Black_Measles)": {"vi_name": "Bệnh sởi đen nho (Esca)", "advice": "Khử trùng dụng cụ cắt tỉa, quét vôi hoặc keo liền sẹo vào vết cắt để ngăn nấm xâm nhập vào thân gỗ."},
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {"vi_name": "Bệnh cháy lá nho (Isariopsis)", "advice": "Thu dọn lá rụng, phun thuốc bảo vệ thực vật gốc đồng hoặc Chlorothalonil định kỳ."},
    "Grape___healthy": {"vi_name": "Lá nho khỏe mạnh", "advice": "Cây sinh trưởng tốt, duy trì kiểm tra sâu bọ trĩ và nấm mốc định kỳ."},

    # Cam (Orange)
    "Orange___Haunglongbing_(Citrus_greening)": {"vi_name": "Bệnh vàng lá gân xanh (Citrus Greening / HLB)", "advice": "Cực kỳ nguy hiểm! Tiêu hủy rầy chổng cánh (vật trung gian), chặt bỏ cây bị bệnh nặng để tránh lây lan cả vườn."},

    # Đào (Peach)
    "Peach___Bacterial_spot": {"vi_name": "Bệnh đốm vi khuẩn trên đào", "advice": "Phun oxytetracycline hoặc thuốc diệt khuẩn gốc đồng liều lượng nhẹ trong mùa mưa ẩm."},
    "Peach___healthy": {"vi_name": "Lá đào khỏe mạnh", "advice": "Cây phát triển tốt, duy trì tỉa cành đón ánh sáng mặt trời."},

    # Ớt chuông (Pepper bell)
    "Pepper,_bell___Bacterial_spot": {"vi_name": "Bệnh đốm vi khuẩn ớt chuông", "advice": "Tránh tưới nước lên lá vào chiều tối, phun thuốc diệt khuẩn gốc đồng kết hợp Mancozeb."},
    "Pepper,_bell___healthy": {"vi_name": "Lá ớt chuông khỏe mạnh", "advice": "Cây sinh trưởng tốt, bổ sung phân kali và canxi định kỳ."},

    # Khoai tây (Potato)
    "Potato___Early_blight": {"vi_name": "Bệnh đốm sớm khoai tây", "advice": "Tránh tưới quá nhiều ẩm, luân canh với cây họ đậu và phun Chlorothalonil hoặc Mancozeb."},
    "Potato___Late_blight": {"vi_name": "Bệnh mốc sương khoai tây (Late Blight)", "advice": "Bệnh lây lan rất nhanh! Cắt bỏ lá bệnh và phun Metalaxyl, Dimethomorph hoặc Cymoxanil ngay."},
    "Potato___healthy": {"vi_name": "Lá khoai tây khỏe mạnh", "advice": "Cây phát triển tốt, vun gốc định kỳ để bảo vệ củ non."},

    # Mâm xôi (Raspberry)
    "Raspberry___healthy": {"vi_name": "Lá mâm xôi khỏe mạnh", "advice": "Cây sinh trưởng tốt, đảm bảo giàn đỡ và đất thoát nước tốt."},

    # Đậu nành (Soybean)
    "Soybean___healthy": {"vi_name": "Lá đậu nành khỏe mạnh", "advice": "Cây phát triển bình thường, giữ cỏ dại sạch sẽ để tránh cạnh tranh dinh dưỡng."},

    # Bí ngô (Squash)
    "Squash___Powdery_mildew": {"vi_name": "Bệnh phấn trắng bí ngô", "advice": "Tưới nước sát gốc (tránh ướt lá), phun dung dịch baking soda sinh học hoặc thuốc trừ nấm gốc lưu huỳnh."},

    # Dâu tây (Strawberry)
    "Strawberry___Leaf_scorch": {"vi_name": "Bệnh cháy xém lá dâu tây", "advice": "Tỉa bỏ lá già úa sát luống, giữ bề mặt luống phủ màng nilon khô ráo và phun Captan."},
    "Strawberry___healthy": {"vi_name": "Lá dâu tây khỏe mạnh", "advice": "Cây đang phát triển rất tốt, duy trì độ ẩm vừa phải cho gốc rễ."},

    # Cà chua (Tomato)
    "Tomato___Bacterial_spot": {"vi_name": "Bệnh đốm vi khuẩn cà chua", "advice": "Khử trùng hạt giống, cách ly cây bệnh và phun thuốc bảo vệ thực vật gốc đồng."},
    "Tomato___Early_blight": {"vi_name": "Bệnh đốm vòng cà chua (Early Blight)", "advice": "Tỉa bỏ lá già sát mặt đất, phun thuốc trừ nấm sinh học hoặc Mancozeb, Chlorothalonil."},
    "Tomato___Late_blight": {"vi_name": "Bệnh mốc sương cà chua (Late Blight)", "advice": "Cực kỳ nguy hiểm! Phun thuốc đặc trị có hoạt chất Metalaxyl hoặc Dimethomorph ngay lập tức."},
    "Tomato___Leaf_Mold": {"vi_name": "Bệnh nấm mốc lá cà chua", "advice": "Giảm độ ẩm trong nhà lưới, thông gió tối đa và phun thuốc Difenoconazole."},
    "Tomato___Septoria_leaf_spot": {"vi_name": "Bệnh đốm lá Septoria cà chua", "advice": "Dọn dẹp lá rụng, tưới nước vào buổi sáng và phun thuốc trừ nấm chứa Chlorothalonil."},
    "Tomato___Spider_mites Two-spotted_spider_mite": {"vi_name": "Bệnh nhện đỏ hai đốm cà chua", "advice": "Tăng cường độ ẩm không khí, phun dầu khoáng hoặc thuốc trừ sâu sinh học Abamectin."},
    "Tomato___Target_Spot": {"vi_name": "Bệnh đốm hồng tâm / đốm đích cà chua", "advice": "Tỉa bớt cành nhánh cho thông gió, phun thuốc diệt nấm Azoxystrobin."},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {"vi_name": "Bệnh xoăn vàng lá cà chua (TYLCV)", "advice": "Do bọ phấn trắng truyền bệnh. Cần bẫy dính màu vàng diệt bọ phấn và nhổ bỏ cây bệnh nặng để bảo vệ ruộng."},
    "Tomato___Tomato_mosaic_virus": {"vi_name": "Bệnh khảm lá cà chua (Mosaic Virus)", "advice": "Khử trùng tay và dụng cụ khi tỉa cây, nhổ bỏ cây bệnh để tránh lây lan qua nhựa cây."},
    "Tomato___healthy": {"vi_name": "Lá cà chua khỏe mạnh", "advice": "Cây đang phát triển rất tốt, duy trì kiểm tra định kỳ."}
}

def load_ai_model():
    global MODEL, CLASSES
    base_dir = Path(__file__).resolve().parent.parent
    classes_path = base_dir / "data" / "processed" / "classes.json"
    
    with open(classes_path, "r", encoding="utf-8") as f:
        CLASSES = json.load(f)["classes"]

    # Uu tien nap model da Domain Adaptation
    ckpt_path = base_dir / "checkpoints" / "convnext_domain_adaptation" / "best_adaptation_model.pth"
    if not ckpt_path.exists():
        ckpt_path = base_dir / "checkpoints" / "convnext_tiny_strongaug" / "best_indomain_model.pth"

    MODEL = get_model("convnext_tiny", num_classes=len(CLASSES), pretrained=False)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    MODEL.load_state_dict(ckpt.get("model_state_dict", ckpt))
    MODEL.to(DEVICE)
    MODEL.eval()
    print(f"[API] Đã nạp thành công mô hình AI ({ckpt_path.name}) trên thiết bị {DEVICE}!")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_ai_model()
    yield

app = FastAPI(
    title="Plant Disease AI Diagnosis API",
    description="API Chẩn Đoán Bệnh Cây Trồng Phục Vụ Ứng Dụng Nông Nghiệp Thông Minh",
    version="1.0.0",
    lifespan=lifespan
)

# Cho phep cac App Mobile / Web goi API khong bi chan CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = TEMPLATE_DIR / "index.html"
    if html_path.exists():
        with open(html_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse("<h2>AgriDoctor AI Pro - Frontend Template Not Found</h2>")

@app.get("/health")
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

    # Phan tich tong hop Benh vs Khoe Manh (Binary Disease Aggregation)
    healthy_indices = [i for i, c in enumerate(CLASSES) if "healthy" in c.lower()]
    disease_indices = [i for i, c in enumerate(CLASSES) if "healthy" not in c.lower()]
    
    prob_healthy_total = float(probs[healthy_indices].sum().item())
    prob_disease_total = float(probs[disease_indices].sum().item())

    # Tim benh co xac suat cao nhat trong tap cac lop benh
    disease_probs = probs[disease_indices]
    best_disease_sub_idx = torch.argmax(disease_probs).item()
    best_disease_orig_idx = disease_indices[best_disease_sub_idx]
    best_disease_class = CLASSES[best_disease_orig_idx]
    best_disease_conf = round(probs[best_disease_orig_idx].item() * 100, 2)

    top_prob, top_idx = torch.topk(probs, k=3)
    raw_top1_class = CLASSES[top_idx[0].item()]
    raw_top1_conf = round(top_prob[0].item() * 100, 2)

    # LOGIC CHẨN ĐOÁN THÔNG MINH (Tránh kết luận lá có đốm nấm là "Khỏe mạnh")
    # Neu tong xac suat mang mam benh >= 35%, TUYET DOI khong the ket luan la la khoe manh!
    is_diseased_overall = prob_disease_total >= 0.35

    if is_diseased_overall and ("healthy" in raw_top1_class.lower()):
        # Xay ra hien tuong: Xac suat benh bi phan tan cho nhieu lop benh (Thoi den, Dom la...)
        # nen 1 lop healthy don le chiem 45%, nhung tong cac benh lai chiem > 55%!
        chosen_class = best_disease_class
        chosen_conf = round(prob_disease_total * 100, 2)
        is_healthy = False
        is_ood = raw_top1_conf < 50.0
        
        disease_info_obj = DISEASE_INFO.get(chosen_class, {"vi_name": chosen_class, "advice": "Cắt tỉa lá bệnh và phun thuốc trừ nấm."})
        status_text = "PHÁT HIỆN DẤU HIỆU BỆNH (ĐỐM LÁ / NẤM HOẠI TỬ)"
        vi_name_display = f"{disease_info_obj['vi_name']}"
        treatment_advice = (
            f"⚠️ Mô hình phát hiện phiến lá có các vết đốm hoại tử / nấm bệnh rất rõ rệt (Tổng xác suất mầm bệnh: {chosen_conf}%). "
            f"Triệu chứng biểu hiện gần nhất với nhóm: {disease_info_obj['vi_name']}. "
            "Nếu đây là loại cây ngoài danh mục 14 cây của đề tài (ví dụ: Dâu tằm, Cây cảnh...), các vết đốm này thường là bệnh Đốm lá nấm (Cercospora/Septoria) hoặc Thán thư. "
            "Biện pháp: Cắt tỉa tiêu hủy lá bệnh, hạn chế đọng nước qua đêm và phun thuốc trừ nấm phổ rộng (Mancozeb, Ridomil Gold hoặc thuốc gốc đồng)."
        )
    else:
        chosen_class = raw_top1_class
        chosen_conf = raw_top1_conf
        is_ood = chosen_conf < 45.0
        is_healthy = ("healthy" in chosen_class.lower()) and (prob_healthy_total >= 0.65) and (not is_ood)

        info = DISEASE_INFO.get(chosen_class, {
            "vi_name": chosen_class.replace("___", " - ").replace("_", " "),
            "advice": "Hãy duy trì chăm sóc cây trồng và tham khảo ý kiến kỹ sư nông nghiệp địa phương nếu vết bệnh lan rộng."
        })

        if is_ood:
            status_text = "CẢNH BÁO: ĐỘ TIN CẬY THẤP (< 45%)"
            vi_name_display = f"{info['vi_name']} (Không chắc chắn)"
            treatment_advice = (
                f"⚠️ Độ tin cậy của AI chỉ đạt {chosen_conf}%. "
                "Khả năng cao loại lá cây này KHÔNG NẰM TRONG 14 LOẠI CÂY đề tài đã nghiên cứu (Táo, Cà chua, Khoai tây, Ngô, Ớt chuông, Nho, Dâu tây...). "
                "Vui lòng thử chụp một trong 14 loại cây trồng được đề tài hỗ trợ hoặc chụp cận cảnh rõ phiến lá hơn!"
            )
        else:
            status_text = "LÁ CÂY KHỎE MẠNH" if is_healthy else "PHÁT HIỆN SÂU BỆNH"
            vi_name_display = info["vi_name"]
            treatment_advice = info["advice"]

    # Danh sach Top 3
    top3 = []
    for p, idx in zip(top_prob, top_idx):
        cls_name = CLASSES[idx.item()]
        top3.append({
            "class_raw": cls_name,
            "disease_name_vi": DISEASE_INFO.get(cls_name, {}).get("vi_name", cls_name),
            "confidence_percent": round(p.item() * 100, 2)
        })

    return {
        "success": True,
        "diagnosis": {
            "disease_code": chosen_class,
            "disease_name_vi": vi_name_display,
            "confidence_percent": chosen_conf,
            "is_healthy": is_healthy,
            "is_ood": is_ood,
            "status": status_text,
            "treatment_recommendation": treatment_advice
        },
        "top3_candidates": top3
    }

def start_tunnel(port: int):
    """Tu dong tao public HTTPS link qua Cloudflare Tunnel de mobile app va web truy cap"""
    system = platform.system().lower()
    bin_dir = Path(__file__).resolve().parent
    if system == "linux":
        bin_path = bin_dir / "cloudflared"
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
    elif system == "windows":
        bin_path = bin_dir / "cloudflared.exe"
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe"
    elif system == "darwin":
        bin_path = bin_dir / "cloudflared"
        url = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-darwin-amd64"
    else:
        print("[Tunnel] Hệ điều hành không hỗ trợ tự tải cloudflared.")
        return None

    if not bin_path.exists():
        print(f"[Tunnel] Đang tải công cụ Cloudflare Tunnel (~30MB) để tạo link HTTPS công khai...")
        try:
            urllib.request.urlretrieve(url, str(bin_path))
            if system != "windows":
                bin_path.chmod(bin_path.stat().st_mode | stat.S_IEXEC)
            print("[Tunnel] Tải công cụ thành công!")
        except Exception as e:
            print(f"[Tunnel] Không thể tải cloudflared: {e}")
            return None

    # Su dung protocol http2 (TCP port 443) de tranh bi tuong lua/nha mang chan UDP/QUIC (gay ra Error 1033)
    cmd = [str(bin_path), "tunnel", "--protocol", "http2", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        tunnel_url = None
        start_time = time.time()
        while time.time() - start_time < 35:
            line = proc.stdout.readline()
            if not line:
                break
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", line)
            if match:
                tunnel_url = match.group(0)
                break

        if tunnel_url:
            print("\n" + "="*75)
            print("🌿 HỆ THỐNG AGRI-DOCTOR AI PRO ĐÃ SẴN SÀNG TRÊN INTERNET (4G / WIFI):")
            print(f"👉 GIAO DIỆN WEB DEMO VIP PRO:  {tunnel_url}/")
            print(f"👉 KẾT NỐI CHO APP EXPO GO:     {tunnel_url}/predict")
            print(f"👉 TÀI LIỆU SWAGGER API DOCS:   {tunnel_url}/docs")
            print("="*75)
            print("💡 Lưu ý: Khi vừa khởi tạo link, nếu thấy 'Error 1033', vui lòng đợi 5-10s rồi bấm F5!")
            print("="*75 + "\n", flush=True)

            # Doc tiep output trong background de theo doi log
            def drain_output():
                try:
                    for line in proc.stdout:
                        if "ERR" in line:
                            print(f"[Tunnel Error] {line.strip()}", flush=True)
                except Exception:
                    pass
            threading.Thread(target=drain_output, daemon=True).start()
            return proc
        else:
            print("[Tunnel] Chưa bắt được link sau 35s. Server nội bộ vẫn hoạt động bình thường.")
            return proc
    except Exception as e:
        print(f"[Tunnel] Lỗi khi khởi động tunnel: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plant Disease AI FastAPI Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8008, help="Port to listen on (default 8008)")
    parser.add_argument("--share", action="store_true", default=True, help="Tu dong tao link Public HTTPS (mac dinh: True)")
    parser.add_argument("--no-share", dest="share", action="store_false", help="Tat tinh nang tao link Public")
    args = parser.parse_args()

    tunnel_proc = None
    if args.share:
        # Chay tunnel trong background thread de khong block uvicorn
        def run_tunnel():
            global tunnel_proc
            # Cho Uvicorn khoi dong khoang 1-2 giay
            time.sleep(2)
            tunnel_proc = start_tunnel(args.port)
        threading.Thread(target=run_tunnel, daemon=True).start()

    try:
        uvicorn.run("app_api:app", host=args.host, port=args.port, reload=False)
    finally:
        if tunnel_proc:
            try:
                tunnel_proc.terminate()
            except Exception:
                pass
