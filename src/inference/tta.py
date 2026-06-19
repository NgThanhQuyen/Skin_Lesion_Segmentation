# =============================================================================
# Thư mục: src/inference/tta.py
# Chức năng: Tiện ích dự đoán sử dụng kỹ thuật tăng cường dữ liệu khi kiểm thử (Test-Time Augmentation - TTA).
# Hàm quan trọng:
#   - tta_predict: Thực hiện dự đoán trên 5 góc xoay/lật hình học khác nhau và lấy trung bình xác suất đầu ra.
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn


@torch.no_grad()
def tta_predict(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """
    Test-Time Augmentation: Tính trung bình xác suất sigmoid trên 5 góc nhìn hình học khác nhau.

    Các góc nhìn hình học bao gồm:
        1. Ảnh gốc.
        2. Lật ảnh theo chiều ngang.
        3. Lật ảnh theo chiều dọc.
        4. Xoay ảnh 90 độ (k=1).
        5. Xoay ảnh 270 độ (k=3).

    Ảnh da liễu có tính đối xứng xoay (không có hướng chuẩn cố định),
    vì vậy cả 5 góc nhìn này đều hợp lệ và giúp tăng độ chính xác phân đoạn đường viền.

    Các tham số đầu vào:
        model:  Mô hình phân đoạn trả về logits thô kích thước (B, C, H, W).
        images: Tensor ảnh đầu vào kích thước (B, C, H, W).

    Kết quả trả về:
        Xác suất trung bình nằm trong khoảng [0, 1], kích thước (B, C, H, W).
    """
    # 1. Ảnh gốc
    probs = torch.sigmoid(model(images))

    # 2. Lật ảnh theo chiều ngang (trục W) và lật ngược lại dự đoán tương ứng
    probs += torch.sigmoid(model(torch.flip(images, dims=[3]))).flip(dims=[3])

    # 3. Lật ảnh theo chiều dọc (trục H) và lật ngược lại dự đoán tương ứng
    probs += torch.sigmoid(model(torch.flip(images, dims=[2]))).flip(dims=[2])

    # 4. Xoay ảnh 90 độ - xoay ảnh đầu vào, xoay ngược lại dự đoán đầu ra (k=3)
    probs += torch.sigmoid(model(torch.rot90(images, k=1, dims=[2, 3]))).rot90(k=3, dims=[2, 3])

    # 5. Xoay ảnh 270 độ - xoay ảnh đầu vào, xoay ngược lại dự đoán đầu ra (k=1)
    probs += torch.sigmoid(model(torch.rot90(images, k=3, dims=[2, 3]))).rot90(k=1, dims=[2, 3])

    return probs / 5.0
