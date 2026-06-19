# =============================================================================
# Thư mục: src/training/distributed.py
# Chức năng: Định nghĩa cấu hình và môi trường cho việc huấn luyện song song phân tán (DDP) bằng torchrun.
# Lớp quan trọng:
#   - DistributedContext: Lớp chứa trạng thái cấu hình phân tán (rank, world_size, local_rank).
# Hàm quan trọng:
#   - parse_torchrun_env: Đọc và xác thực các biến môi trường thiết lập bởi torchrun.
# =============================================================================

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class DistributedContext:
    """
    Trạng thái môi trường chạy của huấn luyện đơn tiến trình hoặc song song phân tán DDP.
    """

    enabled: bool = False
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0

    @property
    def is_main_process(self) -> bool:
        """Kiểm tra xem tiến trình hiện tại có phải là tiến trình chính (rank 0) hay không."""
        return self.rank == 0


def single_process_context() -> DistributedContext:
    """Trả về trạng thái cấu hình mặc định cho trường hợp huấn luyện đơn tiến trình."""
    return DistributedContext(enabled=False, rank=0, world_size=1, local_rank=0)


def parse_torchrun_env(env: Mapping[str, str] | None = None) -> DistributedContext:
    """
    Đọc và phân tích các biến môi trường của lệnh torchrun để tạo cấu hình phân tán.

    Các biến môi trường bắt buộc: RANK, WORLD_SIZE, LOCAL_RANK.
    """
    env_map = os.environ if env is None else env
    required = ("RANK", "WORLD_SIZE", "LOCAL_RANK")
    missing = [key for key in required if key not in env_map]
    if missing:
        raise ValueError(
            "Thiếu các biến môi trường cần thiết của torchrun: "
            f"{', '.join(missing)}. Vui lòng chạy kịch bản thông qua lệnh torchrun."
        )

    try:
        rank = int(env_map["RANK"])
        world_size = int(env_map["WORLD_SIZE"])
        local_rank = int(env_map["LOCAL_RANK"])
    except ValueError as exc:
        raise ValueError("Các biến RANK/WORLD_SIZE/LOCAL_RANK bắt buộc phải là số nguyên.") from exc

    if world_size <= 0:
        raise ValueError("Tham số WORLD_SIZE phải lớn hơn 0.")
    if rank < 0 or rank >= world_size:
        raise ValueError(
            f"Tham số RANK={rank} không hợp lệ với giá trị WORLD_SIZE={world_size}."
        )
    if local_rank < 0:
        raise ValueError("Tham số LOCAL_RANK phải lớn hơn hoặc bằng 0.")

    return DistributedContext(
        enabled=(world_size > 1),
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
    )
