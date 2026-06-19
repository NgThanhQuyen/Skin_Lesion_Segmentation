# =============================================================================
# Thư mục: src/training/callbacks.py
# Chức năng: Định nghĩa các lớp Callback để giám sát quá trình huấn luyện mô hình.
# Lớp quan trọng:
#   - EarlyStopping: Dừng huấn luyện sớm nếu chỉ số đánh giá không cải thiện.
#   - ModelCheckpoint: Tự động lưu trữ mô hình tốt nhất dựa trên chỉ số giám sát.
# =============================================================================

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Dừng huấn luyện sớm khi chỉ số giám sát không được cải thiện sau một số lượng epoch (patience) nhất định.

    Các tham số đầu vào:
        patience:   Số lượng epoch chờ đợi sự cải thiện trước khi dừng huấn luyện.
        min_delta:  Ngưỡng thay đổi tối thiểu để được tính là một sự cải thiện.
        mode:       Chế độ đánh giá: "max" (đối với Dice, IoU) hoặc "min" (đối với Loss).
        monitor:    Tên của chỉ số dùng để giám sát quá trình (phục vụ ghi nhận log).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
        monitor: str = "val_dice",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.monitor = monitor
        self.counter = 0
        self.best = None
        self.triggered = False

    def step(self, value: float) -> bool:
        """
        Đánh giá giá trị của chỉ số tại epoch hiện tại.

        Kết quả trả về:
            True nếu điều kiện dừng sớm được kích hoạt, ngược lại trả về False.
        """
        if self.best is None:
            self.best = value
            return False

        improved = (
            value > self.best + self.min_delta
            if self.mode == "max"
            else value < self.best - self.min_delta
        )

        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            logger.debug(
                f"EarlyStopping: Chỉ số {self.monitor} không cải thiện "
                f"({self.counter}/{self.patience})"
            )
            if self.counter >= self.patience:
                self.triggered = True

        return self.triggered

    def reset(self) -> None:
        """Đặt lại toàn bộ trạng thái của bộ giám sát dừng sớm."""
        self.counter = 0
        self.best = None
        self.triggered = False


class ModelCheckpoint:
    """
    Lưu trữ trạng thái mô hình tốt nhất (weights) dựa trên việc đánh giá chỉ số giám sát.

    Các tham số đầu vào:
        save_path:  Đường dẫn lưu file checkpoint (.pth).
        mode:       Chế độ đánh giá: "max" hoặc "min".
        monitor:    Tên của chỉ số dùng để giám sát (ví dụ: val_dice).
        best:       Giá trị tốt nhất hiện tại (nếu có).
        best_epoch: Epoch đạt được kết quả tốt nhất hiện tại.
    """

    def __init__(
        self,
        save_path: Path | str,
        mode: str = "max",
        monitor: str = "val_dice",
        best: float | None = None,
        best_epoch: int | None = None,
    ):
        self.save_path = Path(save_path)
        self.mode = mode
        self.monitor = monitor
        self.best = best
        self.best_epoch = best_epoch

    def step(
        self,
        value: float,
        model: torch.nn.Module,
        epoch: int,
        extra: dict | None = None,
        model_config: dict[str, Any] | None = None,
    ) -> bool:
        """
        Đánh giá giá trị chỉ số hiện tại và thực hiện lưu đè mô hình nếu đạt kết quả tốt nhất.

        Kết quả trả về:
            True nếu mô hình hiện tại tốt hơn và đã thực hiện lưu file, ngược lại trả về False.
        """
        is_best = (
            self.best is None
            or (self.mode == "max" and value > self.best)
            or (self.mode == "min" and value < self.best)
        )

        if is_best:
            self.best = value
            self.best_epoch = epoch
            payload = {
                "epoch": epoch,
                "model_state_dict": (
                    model.module.state_dict()
                    if hasattr(model, "module")  # Hỗ trợ cấu hình song song DataParallel/DDP
                    else model.state_dict()
                ),
                self.monitor: value,
            }
            if model_config is not None:
                payload["model_config"] = model_config
            if extra:
                payload.update(extra)

            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, self.save_path)
            logger.info(
                f"Lưu checkpoint thành công -> {self.save_path} "
                f"({self.monitor}={value:.4f}, epoch={epoch + 1})"
            )

        return is_best
