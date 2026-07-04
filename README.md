# Phân đoạn vùng tổn thương da (ISIC 2018 Task 1)

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-red.svg)](https://pytorch.org/)
[![Albumentations](https://img.shields.io/badge/Albumentations-1.4%2B-green.svg)](https://albumentations.ai/)
[![WandB](https://img.shields.io/badge/Weights%20%26%20Biases-Tracked-yellow.svg)](https://wandb.ai/)

Dự án nghiên cứu và phát triển hệ thống AI phân đoạn vùng tổn thương da trên ảnh nội soi dermoscopy, sử dụng bộ dữ liệu chuẩn ISIC 2018 Challenge (Task 1). Mã nguồn được thiết kế theo dạng mô-đun hóa cao, đảm bảo tính tái tạo kết quả thử nghiệm và tối ưu hóa hiệu năng huấn luyện tối đa. Dự án tích hợp các kiến trúc phân đoạn tiên tiến, cơ chế huấn luyện song song phân tán (DDP), độ chính xác hỗn hợp (Mixed Precision) và tăng cường dữ liệu khi kiểm thử (Test-Time Augmentation).

---

## Các tính năng chính

- **Huấn luyện song song phân tán (DDP):** Hỗ trợ đầy đủ huấn luyện phân tán đa GPU thông qua cơ chế DistributedDataParallel của PyTorch và công cụ khởi chạy torchrun, được cấu hình tối ưu cho môi trường Kaggle 2x T4 GPU.
- **Huấn luyện độ chính xác hỗn hợp (Mixed Precision):** Tích hợp công cụ GradScaler và cơ chế tự động ép kiểu AMP (FP16) để tăng tốc độ tính toán và giảm thiểu dung lượng bộ nhớ GPU tiêu thụ.
- **Tăng cường dữ liệu khi kiểm thử (Test-Time Augmentation - TTA):** Áp dụng kỹ thuật TTA trên 5 góc nhìn hình học khác nhau (ảnh gốc, lật ngang, lật dọc, xoay 90 độ, xoay 270 độ) giúp tăng cường độ chính xác và tính ổn định ở các vùng biên tổn thương da.
- **Hệ thống cấu hình phân cấp:** Quản lý tham số thử nghiệm thông qua các file cấu hình YAML có khả năng kế thừa lẫn nhau và cho phép ghi đè linh hoạt trực tiếp từ dòng lệnh CLI.
- **Tối ưu hóa học tập nâng cao:** Sử dụng tỷ lệ học phân biệt (Differential Learning Rates - phân tách tốc độ học của encoder và decoder), cơ chế dừng sớm (Early Stopping) và tự động giảm tỷ lệ học khi gặp trạng thái bão hòa (ReduceLROnPlateau).
- **Ghi nhận lịch sử và trực quan hóa kết quả:** Đồng bộ hóa biểu đồ học tập theo thời gian thực lên hệ thống đám mây Weights & Biases (W&B), kết hợp lưu trữ file log cục bộ (CSV/JSON) và tự động xuất ảnh so sánh overlay.

---

## Cấu trúc thư mục dự án

```text
Skin_Lesion_Segmentation/
├── configs/                  # Các file cấu hình hệ thống và thử nghiệm
│   ├── base.yaml             # Cấu hình mặc định cho dự án
│   └── experiments/          # Cấu hình ghi đè cho từng mô hình thử nghiệm cụ thể
├── scripts/                  # Các kịch bản chạy chính của hệ thống
│   ├── prepare_data.py       # Tiền xử lý và chia tách tập dữ liệu (Train/Val/Test)
│   ├── train.py              # Kịch bản huấn luyện mô hình (GPU đơn lẻ hoặc DDP)
│   ├── evaluate.py           # Đánh giá mô hình trên tập kiểm thử (kết hợp TTA)
│   ├── predict.py            # Chạy suy luận và trực quan hóa kết quả phân đoạn
│   └── benchmark_fps.py      # Tiện ích đo đạc độ trễ và tốc độ xử lý FPS của mô hình
├── src/                      # Thư mục mã nguồn lõi
│   ├── data/                 # Lớp nạp dữ liệu và kịch bản tăng cường ảnh
│   ├── models/               # Bộ xây dựng mô hình và các kiến trúc mạng tùy chỉnh
│   ├── losses/               # Định nghĩa các hàm mất mát (Focal Loss, Dice Loss)
│   ├── metrics/              # Chỉ số đánh giá hiệu năng (Dice Coefficient, IoU)
│   ├── inference/            # Tiện ích dự đoán TTA
│   ├── training/             # Lớp Trainer điều phối huấn luyện và các Callbacks
│   └── utils/                # Đọc cấu hình, checkpoint, ghi nhận nhật ký hệ thống
├── requirements.txt          # Các thư viện tối thiểu cần cài đặt (phù hợp với Kaggle)
├── environment.yml           # File thiết lập môi trường ảo Conda cho máy cục bộ
└── pyproject.toml            # File định nghĩa thông tin đóng gói dự án Python
```

---

## Các kiến trúc mô hình hỗ trợ

Dự án hỗ trợ chuyển đổi linh hoạt giữa nhiều kiến trúc mô hình phân đoạn khác nhau thông qua file cấu hình:
1. **U-Net:** Sử dụng các mạng xương sống mã hóa (encoder backbones) tiền huấn luyện mạnh mẽ (như ResNet-34, ResNet-50) kết hợp với cơ chế chú ý không gian-kênh scSE (spatial-channel Squeeze-and-Excitation) ở khối giải mã.
2. **U-Net nguyên bản:** Kiến trúc U-Net truyền thống được xây dựng hoàn toàn từ đầu, không sử dụng các khối tiền huấn luyện.
3. **DeepLabV3:** Mô hình phân đoạn ngữ nghĩa hiệu năng cao sử dụng mạng xương sống MobileNetV3-Large gọn nhẹ.
4. **DeepLabV3+:** Phiên bản cải tiến tích hợp bộ mã hóa ResNet-50 hỗ trợ xử lý đa tỷ lệ không gian tối ưu.
5. **TransUNet:** Kiến trúc lai tiên tiến kết hợp giữa Transformer (ViT) để nắm bắt ngữ cảnh toàn cục và CNN để duy trì chi tiết không gian cục bộ của ảnh.

---

## Đánh giá hiệu năng và Kết quả thực tế

### 1. Cấu hình huấn luyện
- **Hàm mất mát:** Combined Loss (Trọng số 0.5 Focal Loss + 0.5 Soft Dice Loss) nhằm giải quyết triệt để vấn đề mất cân bằng lớp giữa vùng tổn thương da và vùng da lành xung quanh.
- **Kích thước ảnh đầu vào:** 256x256 pixel.
- **Bộ tối ưu hóa:** AdamW (tỷ lệ học cơ bản: 2.0e-4, suy giảm trọng số: 1.0e-4).

### 2. Kết quả kiểm thử thực tế
Kết quả thực tế đo được sau quá trình huấn luyện cấu hình thử nghiệm **ResNet-34 U-Net** (`resnet34_unet_v1`):

| Kiến trúc | Tập dữ liệu | Chỉ số đánh giá | Điểm số (Kết hợp TTA) |
| :--- | :--- | :--- | :--- |
| **ResNet-34 U-Net** | Tập kiểm thử (Test Set) | **Hệ số xúc xắc Dice Coefficient** | **0.9021** (90.21%) |
| **ResNet-34 U-Net** | Tập kiểm thử (Test Set) | **Chỉ số trùng lặp IoU (Jaccard)** | **0.8368** (83.68%) |

---

## Hướng dẫn cài đặt và sử dụng

### 1. Cài đặt môi trường

#### **Trên máy cá nhân (Local)**
```bash
# Tải mã nguồn về máy
git clone https://github.com/NgThanhQuyen/Skin_Lesion_Segmentation.git
cd Skin_Lesion_Segmentation

# Khởi tạo môi trường Conda
conda env create -f environment.yml
conda activate CV

# Cài đặt mã nguồn ở chế độ chỉnh sửa (editable mode)
pip install -e .
```

#### **Trên môi trường Kaggle Notebook**
Thực thi dòng lệnh sau tại cell đầu tiên:
```bash
!pip install -r requirements.txt -q
```

---

### 2. Chuẩn bị dữ liệu

1. Tải về dữ liệu hình ảnh và mặt nạ của bộ dữ liệu ISIC 2018 Task 1, sau đó đặt vào cấu trúc thư mục sau:
   ```text
   data/data-HA10000-remove-hair/
   ├── remove-hair/images/     # Thư mục ảnh da gốc (định dạng ISIC_*.jpg)
   └── masks/                  # Thư mục mặt nạ thực tế (định dạng ISIC_*.png)
   ```
2. Chạy kịch bản phân chia để tự động phân phối dữ liệu thành các tập Train (80%), Val (10%), và Test (10%):
   ```bash
   python scripts/prepare_data.py
   ```

---

### 3. Huấn luyện mô hình

#### **Chạy trên thiết bị đơn lẻ (Local)**
```bash
python scripts/train.py --config configs/experiments/resnet34_unet_v1.yaml
```

#### **Chạy song song phân tán (Kaggle 2x T4 GPU)**
```bash
!torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train.py \
  --device-mode ddp \
  --config configs/experiments/resnet34_unet_kaggle_t4.yaml \
  data.root=/kaggle/input/datasets/quynnguynthanh/isic-2018-task1/ISIC_2018_TASK1 \
  output.dir=/kaggle/working \
  logging.use_wandb=false
```

---

### 4. Kiểm thử và Suy luận dự đoán

#### **Chạy đánh giá trên tập kiểm thử**
Đánh giá mô hình đã huấn luyện và tự động tính toán tìm kiếm ngưỡng phân ngưỡng nhị phân tối ưu nhất:
```bash
python scripts/evaluate.py \
  --config configs/experiments/resnet34_unet_v1.yaml \
  --checkpoint outputs/resnet34_unet_v1/best_model.pth \
  --split test \
  --tta
```

#### **Chạy dự đoán trên ảnh mới**
Thực hiện suy luận dự đoán trên thư mục hình ảnh mới và xuất các biểu đồ so sánh overlay trực quan:
```bash
python scripts/predict.py \
  --config configs/experiments/resnet34_unet_v1.yaml \
  --checkpoint outputs/resnet34_unet_v1/best_model.pth \
  --input data/processed/test/images \
  --output outputs/predictions \
  --tta \
  --overlay
```
