# Plant Disease Recognition and Cross-Domain Generalization
**Nhận dạng bệnh cây trồng và đánh giá khả năng tổng quát hóa sang ảnh thực địa**

---

## 📌 1. Phân Tích Chuyên Sâu Đề Tài (Academic Analysis)

### 1.1. Bản chất bài toán và Hiện tượng Domain Shift (Dịch chuyển miền dữ liệu)
* **Tập nguồn (Source Domain - PlantVillage)**: Gồm ~54.000 ảnh thuộc 38 lớp (14 loài cây trồng kết hợp các bệnh và trạng thái khỏe mạnh). Tuy nhiên, ảnh PlantVillage được chụp trong **điều kiện phòng thí nghiệm lý tưởng**:
  - Mỗi ảnh là 1 phiến lá duy nhất đặt trên nền xám/đen/trắng phẳng đồng nhất.
  - Ánh sáng chuẩn, góc chụp vuông góc trực diện (top-down view).
  - Không có tạp âm ngoại cảnh (cỏ dại, đất cát, bóng râm, côn trùng khác, tay người cầm).
  - **Hệ quả**: Hầu hết các mô hình CNN (ResNet, MobileNet) huấn luyện trên PlantVillage đều dễ dàng đạt **98% - 99.5% Test Accuracy** sau vài epoch.
* **Tập đích (Target Domain - PlantDoc / In-the-wild)**: Bộ dữ liệu do Singhal et al. (2020) công bố, chụp trực tiếp ngoài vườn cây thực tế của nông dân:
  - Nhiều lá lồng vào nhau, cành cây, quả, thân cây.
  - Nền tự nhiên phức tạp: đất, cỏ dại, ánh nắng chói chang hoặc bóng râm gắt, nước đọng.
  - Người cầm lá chụp bằng điện thoại thông minh ở các góc nghiêng khác nhau.

### 1.2. Hiện tượng cốt lõi: "Shortcut Learning" & "Background Bias"
* Khi mô hình Deep Learning học trên PlantVillage, nó có xu hướng tìm **con đường tắt (shortcut)** để tối ưu hàm mất mát (loss function).
* Thay vì chỉ học hoa văn vi khuẩn, đốm nấm hay cấu trúc bệnh lý tế bào trên lá, mạng nơ-ron vô tình học luôn:
  1. **Độ tương phản viền lá với nền lab**: Nền xám/đen làm viền lá quá rõ, khiến mạng phụ thuộc vào hình dáng viền lá nhân tạo.
  2. **Ánh sáng studio đồng nhất**: Không có biến thiên độ phơi sáng tự nhiên.
* Khi đưa mô hình này sang kiểm thử trực tiếp trên **PlantDoc** (Cross-Domain Generalization):
  - Độ chính xác thường bị **sụt giảm thảm hại (Catastrophic Performance Drop)**: từ **~99% rơi xuống còn 20% - 40%**!
  - **Điểm nhấn nghiên cứu của đề tài**: Chứng minh định lượng sự sụt giảm này, giải thích bằng trực quan hóa Explainable AI (**Grad-CAM**), và đề xuất giải pháp thu hẹp khoảng cách miền (Domain Gap).

---

## 🚀 2. Đề Xuất Mô Hình & Kỹ Thuật "Ngon Hơn" (Proposed Improvements)

| Hạng mục | Baseline Ban Đầu | Đề Xuất Cải Tiến ("Ngon Hơn") | Ưu Điểm Đột Phá |
| :--- | :--- | :--- | :--- |
| **Traditional ML** | HOG + SVM | HOG + Color Histogram + LinearSVC Pipeline | Nắm bắt đồng thời cả hình học cạnh và phân bố màu sắc đặc trưng của nấm bệnh |
| **Deep Architecture** | ResNet-50 / MobileNetV2 | **ConvNeXt-Tiny** / **Swin Transformer** | ConvNeXt kết hợp inductive bias của CNN với thiết kế hiện đại của Vision Transformer, trích xuất đặc trưng đa quy mô cực tốt, bền vững hơn nhiều trước domain shift |
| **Data Augmentation** | Resize + RandomCrop + Flip | **Strong Domain Generalization Augmentation** (RandAugment, ColorJitter, RandomErasing) | Phá vỡ cấu trúc nền giả tạo của lab, mô phỏng ánh sáng chói lóa ngoài thực địa |
| **Regularization** | Standard CrossEntropy | **Label Smoothing (0.1) + Cosine Annealing LR** | Ngăn ngừa mô hình trở nên "overconfident" vào các đặc trưng cục bộ của phòng lab |
| **Giải Thích Học Thuật (XAI)** | Không có | **Grad-CAM Attention Heatmap Analysis** | Vẽ bản đồ nhiệt so sánh trực quan xem mô hình chú ý vào vết bệnh hay bị lừa bởi nền đất/bóng râm |

---

## 📂 3. Cấu Trúc Dự Án (Project Structure)

```
thay_HUY/
├── configs/
│   └── class_mapping.json          # Mapping chuẩn hóa nhãn giữa PlantDoc và PlantVillage (38 lớp)
├── data/
│   ├── plantvillage/               # Ảnh gốc PlantVillage (~54.000 ảnh)
│   ├── plantdoc/                   # Ảnh gốc PlantDoc (In-the-wild)
│   └── processed/                  # CSV chia tập Train/Val/Test và aligned Cross-Domain test set
├── scripts/
│   ├── download_datasets.py        # Tự động kéo dữ liệu từ Kaggle API / GitHub
│   ├── prepare_datasets.py         # Chia tập Stratified & Map nhãn giao nhau
│   └── run_experiments.sh          # Script chạy toàn bộ thực nghiệm từ A-Z trên Server Linux
├── src/
│   ├── dataset.py                  # PyTorch Dataset + Standard & Strong Augmentations
│   ├── models.py                   # Model Factory: ResNet, MobileNet, ConvNeXt, Swin
│   ├── baseline_hog_svm.py         # Baseline Traditional ML (HOG + Color Hist + SVM)
│   ├── train.py                    # Pipeline huấn luyện Deep Learning (AMP, Cosine LR, Logging)
│   ├── evaluate_cross_domain.py    # Kiểm thử chéo In-Domain vs Cross-Domain, xuất Confusion Matrix
│   └── explainability_gradcam.py   # Phân tích Grad-CAM giải thích hiện tượng Shortcut Learning
├── reports/                        # Nơi lưu đồ thị, bảng tổng kết số liệu và ảnh Grad-CAM
├── requirements.txt                # Danh sách thư viện Python
└── run_experiments.bat             # Script chạy tự động trên Windows
```

---

## 🛠️ 4. Hướng Dẫn Chạy Từ A-Z Trên Server Training (GPU)

### Bước 1: Cài Đặt Môi Trường
```bash
pip install -r requirements.txt
```

### Bước 2: Kéo Dữ Liệu Tự Động
```bash
python scripts/download_datasets.py
```
> *Lưu ý*: Script hỗ trợ tải tự động qua **Kaggle API**. Đảm bảo đã đặt file `kaggle.json` tại `~/.kaggle/` (Linux) hoặc `C:\Users\<Tên>\.kaggle\` (Windows). Nếu tải thủ công, chỉ cần giải nén dữ liệu vào thư mục `data/plantvillage/` và `data/plantdoc/`.

### Bước 3: Tiền Xử Lý Dữ Liệu & Tạo Mapping Nhãn
```bash
python scripts/prepare_datasets.py
```
Lệnh này sẽ quét dữ liệu, chia tập Train (70%), Val (15%), In-Domain Test (15%) cho PlantVillage và lọc ra tập kiểm thử chéo thực địa PlantDoc đã map đúng tên nhãn.

### Bước 4: Chạy Baseline 1 (HOG + SVM)
```bash
python src/baseline_hog_svm.py --max_train_samples 15000
```

### Bước 5: Huấn Luyện Deep Learning Baselines (Đã tích hợp Early Stopping)
* Mặc định script sẽ theo dõi `val_loss` và tự động dừng nếu sau `--patience 5` epoch không tiến bộ (hoặc truyền `--patience 0` để tắt).
* **Chạy ResNet-50 (Standard Augmentations)**:
  ```bash
  python src/train.py --model resnet50 --epochs 25 --batch_size 64 --lr 1e-4 --patience 5
  ```
* **Chạy MobileNetV2 (Mô hình nhẹ tối ưu cho thiết bị nhúng/di động)**:
  ```bash
  python src/train.py --model mobilenet_v2 --epochs 25 --batch_size 64 --lr 1e-4 --patience 5
  ```

### Bước 6: Huấn Luyện Mô Hình Đề Xuất (ConvNeXt-Tiny + Strong Augmentation)
```bash
python src/train.py --model convnext_tiny --epochs 30 --batch_size 64 --lr 1e-4 --strong_aug --patience 7
```

### Bước 7: Đánh Giá Tổng Quát Hóa Miền (Cross-Domain Evaluation)
Đánh giá và so sánh mức độ tụt giảm hiệu năng giữa môi trường phòng lab và thực địa:
```bash
# Đánh giá ResNet-50
python src/evaluate_cross_domain.py \
  --checkpoint checkpoints/resnet50_standard/best_indomain_model.pth \
  --model resnet50

# Đánh giá ConvNeXt-Tiny đề xuất
python src/evaluate_cross_domain.py \
  --checkpoint checkpoints/convnext_tiny_strongaug/best_indomain_model.pth \
  --model convnext_tiny
```

### Bước 8: Trực Quan Hóa Grad-CAM (Phân Tích Hiện Tượng "Shortcut Learning")
Tạo hình ảnh phân tích nhiệt xem mạng nơ-ron nhìn vào đâu:
```bash
python src/explainability_gradcam.py \
  --checkpoint checkpoints/resnet50_standard/best_indomain_model.pth \
  --image_path path/to/sample_leaf_image.jpg \
  --model resnet50 \
  --output_file reports/gradcam_analysis.png
```

---

## 📊 5. Mẫu Bảng Kết Quả Đánh Giá Để Đưa Vào Báo Cáo / Luận Văn

| Phương Pháp / Mô Hình | Kỹ Thuật Huấn Luyện | In-Domain Acc (PlantVillage Lab) | Cross-Domain Acc (PlantDoc Field) | Sụt Giảm ($\Delta$ Drop) | Đánh Giá Khả Năng Tổng Quát Hóa |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Baseline 1: HOG + SVM** | Feature Engineering | ~72.4% | ~18.5% | **-53.9%** | Rất kém; sụp đổ trước nhiễu ngoại cảnh |
| **Baseline 2: ResNet-50** | Standard Augmentation | **~99.2%** | ~34.8% | **-64.4%** | Accuracy lab cực cao nhưng bị "học tắt" trên nền |
| **Baseline 3: MobileNetV2**| Standard Augmentation | ~98.4% | ~32.1% | **-66.3%** | Nhẹ, chạy nhanh nhưng dễ bị ảnh hưởng bởi ánh sáng |
| **Đề Xuất: ConvNeXt-Tiny** | **Strong Aug (RandAug + ColorJitter)** | ~98.9% | **~52.6%** | **-46.3%** | **Tốt nhất**: Cải thiện vượt bậc độ chính xác thực tế |
"# Detect_plan" 
