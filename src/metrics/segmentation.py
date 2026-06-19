# =============================================================================
# Thư mục: src/metrics/segmentation.py
# Chức năng: Định nghĩa các chỉ số đánh giá (Evaluation metrics) cho bài toán phân đoạn nhị phân.
# Hàm quan trọng:
#   - dice_coefficient: Tính hệ số xúc xắc Dice Coefficient (độ tương đồng ảnh).
#   - iou_score: Tính hệ số trùng lặp IoU / Jaccard Score.
# =============================================================================

from __future__ import annotations

import torch


@torch.no_grad()
def dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Tính hệ số xúc xắc Dice Coefficient (áp dụng macro-average: tính trên từng ảnh rồi lấy trung bình).

    Các tham số đầu vào:
        pred:      Các giá trị logit thô dự đoán từ mô hình (kích thước B, 1, H, W).
        target:    Mặt nạ nhị phân thực tế (kích thước B, 1, H, W).
        threshold: Ngưỡng để nhị phân hóa dự đoán từ mô hình.

    Kết quả trả về:
        Giá trị số thực nằm trong khoảng [0, 1].
    """
    probs = torch.sigmoid(pred)
    binary = (probs > threshold).float()

    flat_pred = binary.view(binary.size(0), -1)
    flat_target = target.view(target.size(0), -1).float()

    intersection = (flat_pred * flat_target).sum(1)
    union = flat_pred.sum(1) + flat_target.sum(1)
    dice = (2.0 * intersection + 1e-7) / (union + 1e-7)

    return dice.mean().item()


@torch.no_grad()
def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    """
    Tính chỉ số IoU / Jaccard Score (áp dụng macro-average: tính trên từng ảnh rồi lấy trung bình).

    Các tham số đầu vào:
        pred:      Các giá trị logit thô dự đoán từ mô hình (kích thước B, 1, H, W).
        target:    Mặt nạ nhị phân thực tế (kích thước B, 1, H, W).
        threshold: Ngưỡng để nhị phân hóa dự đoán từ mô hình.

    Kết quả trả về:
        Giá trị số thực nằm trong khoảng [0, 1].
    """
    probs = torch.sigmoid(pred)
    binary = (probs > threshold).float()

    flat_pred = binary.view(binary.size(0), -1)
    flat_target = target.view(target.size(0), -1).float()

    intersection = (flat_pred * flat_target).sum(1)
    union = flat_pred.sum(1) + flat_target.sum(1) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)

    return iou.mean().item()
