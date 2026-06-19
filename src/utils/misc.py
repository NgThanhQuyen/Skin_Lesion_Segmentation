# =============================================================================
# Thư mục: src/utils/misc.py
# Chức năng: Định nghĩa các hàm tiện ích dùng chung (Miscellaneous utilities) cho toàn bộ project.
# Hàm quan trọng:
#   - set_seed: Cài đặt seed ngẫu nhiên cho tính tái tạo kết quả (reproducibility).
#   - get_device: Kiểm tra và trả về thiết bị phần cứng khả dụng (CUDA hoặc CPU).
#   - count_parameters: Đếm số lượng tham số của mô hình (total, trainable, frozen).
#   - plot_training_curves: Vẽ đồ thị diễn biến huấn luyện (Loss, Dice, IoU, LR).
#   - denormalize: Chuyển đổi ảnh chuẩn hóa ImageNet về khoảng [0, 1].
# =============================================================================

from __future__ import annotations

import random
import numpy as np
from pathlib import Path

import torch
import matplotlib.pyplot as plt


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Cài đặt seed ngẫu nhiên để đảm bảo tính tái tạo kết quả (reproducible).

    Các tham số đầu vào:
        seed:          Giá trị seed ngẫu nhiên cho tất cả các bộ sinh số ngẫu nhiên.
        deterministic: Nếu bằng True, bật chế độ cudnn deterministic và tắt chế độ benchmark
                       -> kết quả tái tạo hoàn toàn giống nhau nhưng tốc độ huấn luyện chậm hơn khoảng 10-30%.
                       Nếu bằng False, chỉ cài đặt seed và bật chế độ cudnn benchmark
                       -> tốc độ huấn luyện nhanh hơn, phù hợp khi chạy production thực tế.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def get_device() -> torch.device:
    """Kiểm tra và trả về thiết bị phần cứng khả dụng nhất (ưu tiên GPU CUDA hơn CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: torch.nn.Module) -> dict:
    """Đếm tổng số lượng tham số (parameters) của mô hình."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "size_mb": total * 4 / 1024**2,  # Giả định lưu trữ kiểu số thực float32 (4 bytes)
    }


def plot_training_curves(history: list[dict], save_path: Path) -> None:
    """
    Vẽ 4 đồ thị học tập bao gồm: Loss, Dice, IoU và Learning Rate sau khi kết thúc huấn luyện.

    Các tham số đầu vào:
        history:   Danh sách chứa các từ điển ghi nhận chỉ số đánh giá qua từng epoch.
        save_path: Đường dẫn lưu trữ hình ảnh biểu đồ đầu ra (định dạng PNG).
    """
    epochs = list(range(1, len(history) + 1))

    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_dice = [h["train_dice"] for h in history]
    val_dice = [h["val_dice"] for h in history]
    train_iou = [h["train_iou"] for h in history]
    val_iou = [h["val_iou"] for h in history]
    lr = [h["lr"] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Vẽ đồ thị biểu diễn hàm mất mát (Loss)
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, "b-", label="Hao hụt huấn luyện", linewidth=2)
    ax.plot(epochs, val_loss, "r-", label="Hao hụt kiểm định", linewidth=2)
    ax.set_title("Hàm mất mát kết hợp (Focal + Dice)", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hao hụt")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    # Vẽ đồ thị biểu diễn hệ số Dice Coefficient
    ax = axes[0, 1]
    ax.plot(epochs, train_dice, "b-", label="Dice huấn luyện", linewidth=2)
    ax.plot(epochs, val_dice, "r-", label="Dice kiểm định", linewidth=2)
    if val_dice:
        best_ep = int(np.argmax(val_dice))
        ax.plot(
            best_ep + 1,
            val_dice[best_ep],
            "g*",
            markersize=14,
            label=f"Tốt nhất: {val_dice[best_ep]:.4f}",
        )
    ax.set_title("Hệ số xúc xắc Dice Coefficient", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    # Vẽ đồ thị biểu diễn chỉ số IoU Score
    ax = axes[1, 0]
    ax.plot(epochs, train_iou, "b-", label="IoU huấn luyện", linewidth=2)
    ax.plot(epochs, val_iou, "r-", label="IoU kiểm định", linewidth=2)
    ax.set_title("Chỉ số trùng lặp IoU Score (Jaccard)", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    # Vẽ đồ thị biểu diễn tỷ lệ học (Learning Rate)
    ax = axes[1, 1]
    ax.plot(epochs, lr, "g-", label="Tỷ lệ học", linewidth=2)
    ax.set_title("Chiến lược cập nhật tỷ lệ học", fontsize=14)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def denormalize(tensor: "torch.Tensor") -> np.ndarray:
    """
    Hàm chuyển đổi ngược hình ảnh từ định dạng chuẩn hóa ImageNet về khoảng [0, 1] để hiển thị.

    Tham số đầu vào:
        tensor: Ảnh đầu vào dạng PyTorch Tensor kích thước (C, H, W).
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    return np.clip(img * std + mean, 0, 1)
