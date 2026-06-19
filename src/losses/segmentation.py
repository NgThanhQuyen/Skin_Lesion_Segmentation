# =============================================================================
# Thư mục: src/losses/segmentation.py
# Chức năng: Định nghĩa các hàm mất mát (Loss functions) dùng cho bài toán phân đoạn nhị phân (Binary Segmentation).
# Lớp quan trọng:
#   - FocalLoss: Hàm mất mát Focal Loss giúp tập trung vào các mẫu khó phân loại (như pixel đường biên).
#   - SoftDiceLoss: Hàm mất mát Soft Dice Loss giúp xử lý vấn đề mất cân bằng lớp (class imbalance).
#   - CombinedLoss: Hàm mất mát kết hợp giữa Focal Loss và Soft Dice Loss theo trọng số cấu hình.
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = (1 - p_t)^gamma * BCE(p_t)
    Hàm mất mát này tập trung tối ưu hóa các mẫu khó phân loại (ví dụ các pixel ở đường biên của vùng tổn thương).

    Các tham số đầu vào:
        gamma: Tham số điều chỉnh tiêu điểm (mặc định=2.0).
        alpha: Trọng số cho lớp tổn thương (lesion class). Nếu truyền None thì không áp dụng trọng số lớp.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: Các giá trị logit thô (kích thước B, 1, H, W)
        # targets: Mặt nạ nhị phân có giá trị thuộc khoảng {0, 1} (kích thước B, 1, H, W)
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = (1 - pt) ** self.gamma * bce

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal = alpha_t * focal

        return focal.mean()


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss - Hàm mất mát giúp giải quyết vấn đề mất cân bằng lớp hiệu quả.
    Giá trị được tính toán trên từng ảnh đơn lẻ rồi mới tính trung bình trên toàn bộ batch,
    nhờ đó các vùng tổn thương nhỏ không bị ảnh hưởng quá mức bởi các vùng lớn hơn.

    Các tham số đầu vào:
        eps: Hệ số làm mịn để tránh lỗi chia cho số 0.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(inputs)
        targets = targets.float()
        dims = (1, 2, 3)  # Tính toán trên các chiều H, W, C của từng ảnh

        intersection = (probs * targets).sum(dims)
        union = probs.sum(dims) + targets.sum(dims)
        dice_score = (2.0 * intersection + self.eps) / (union + self.eps)

        return 1.0 - dice_score.mean()


class CombinedLoss(nn.Module):
    """
    Hàm mất mát kết hợp: Combined Loss = focal_weight * FocalLoss + dice_weight * SoftDiceLoss

    Các tham số cấu hình từ hệ thống:
        config.training.loss.focal_weight  (mặc định: 0.5)
        config.training.loss.dice_weight   (mặc định: 0.5)
        config.training.loss.focal_gamma   (mặc định: 2.0)
        config.training.loss.focal_alpha   (mặc định: null)
    """

    def __init__(self, config):
        super().__init__()
        loss_cfg = config.training.loss
        self.focal_weight = loss_cfg.focal_weight
        self.dice_weight = loss_cfg.dice_weight
        self.focal = FocalLoss(
            gamma=loss_cfg.focal_gamma,
            alpha=loss_cfg.focal_alpha,
        )
        self.dice = SoftDiceLoss()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.focal_weight * self.focal(inputs, targets)
            + self.dice_weight * self.dice(inputs, targets)
        )
