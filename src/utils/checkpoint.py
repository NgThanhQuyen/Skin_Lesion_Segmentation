# =============================================================================
# Thư mục: src/utils/checkpoint.py
# Chức năng: Cung cấp các tiện ích tải trọng số mô hình (checkpoint loading)
#            hỗ trợ tương thích ngược cho các phiên bản mô hình cũ.
# Hàm quan trọng:
#   - load_state_dict_with_aux_compat: Tải state_dict và tự động loại bỏ các khóa lỗi thời.
# =============================================================================

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Tiền tố của khóa bổ trợ lỗi thời (legacy aux classifier)
_AUX_KEY_PREFIX = "aux_classifier."


def load_state_dict_with_aux_compat(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    context: str = "checkpoint",
) -> None:
    """
    Nạp trọng số state_dict vào mô hình và tự động bỏ qua các khóa của nhánh phân loại phụ cũ (legacy aux head).

    Hàm này hỗ trợ quá trình chuyển đổi từ các checkpoint DeepLab cũ (có sử dụng nhánh phụ aux head) sang
    mô hình mới đã tắt tính năng này (cấu hình aux_loss=False), nhưng vẫn duy trì cơ chế bắt lỗi nghiêm ngặt (strict)
    đối với tất cả các trường hợp không khớp trọng số khác nhằm tránh bỏ sót lỗi.

    Các tham số đầu vào:
        model:      Thực thể mô hình PyTorch (hỗ trợ cả DataParallel).
        state_dict: Bộ trọng số tải về từ tệp checkpoint.
        context:    Chuỗi ngữ cảnh hỗ trợ ghi nhật ký và gỡ lỗi (debug).

    Ngoại lệ:
        RuntimeError: Nếu phát hiện có lỗi thiếu khóa (missing) hoặc thừa khóa (unexpected) ngoài nhánh phụ aux head.
    """
    model_ref = model.module if hasattr(model, "module") else model
    incompatible = model_ref.load_state_dict(state_dict, strict=False)

    unexpected = list(incompatible.unexpected_keys)
    missing = list(incompatible.missing_keys)

    ignored_aux = [k for k in unexpected if k.startswith(_AUX_KEY_PREFIX)]
    other_unexpected = [k for k in unexpected if not k.startswith(_AUX_KEY_PREFIX)]
    other_missing = [k for k in missing if not k.startswith(_AUX_KEY_PREFIX)]

    if ignored_aux:
        logger.warning(
            "Đã bỏ qua %d khóa phụ lỗi thời (legacy aux key) khi nạp %s.",
            len(ignored_aux),
            context,
        )

    if other_missing or other_unexpected:
        problems: list[str] = []
        if other_missing:
            problems.append(f"thiếu các khóa: {other_missing}")
        if other_unexpected:
            problems.append(f"thừa các khóa không xác định: {other_unexpected}")
        details = "; ".join(problems)
        raise RuntimeError(f"Trọng số mô hình (State dict) không khớp khi tải {context}: {details}")
