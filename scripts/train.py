#!/usr/bin/env python3
# =============================================================================
# Thư mục: scripts/train.py
# Chức năng: Điểm khởi chạy chính (Entry point) cho quá trình huấn luyện mô hình phân đoạn.
#            Hỗ trợ huấn luyện đơn GPU/CPU hoặc song song phân tán (DDP) đa GPU.
# Hàm quan trọng:
#   - build_dataloaders: Khởi tạo và thiết lập các bộ nạp dữ liệu Train/Val DataLoader.
#   - main: Điều phối luồng khởi tạo môi trường, nạp cấu hình, dựng mô hình và chạy huấn luyện.
# =============================================================================

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Đảm bảo đường dẫn gốc của repo nằm trong PYTHONPATH để cho phép import thư mục src
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.data.dataset import ISICDataset
from src.data.transforms import get_transforms
from src.models.segmentation import create_model
from src.training.distributed import (
    DistributedContext,
    parse_torchrun_env,
    single_process_context,
)
from src.training.trainer import Trainer
from src.utils.config import load_config, override_config
from src.utils.logger import Logger
from src.utils.misc import count_parameters, get_device, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Các lớp và hàm bổ trợ
# ---------------------------------------------------------------------------


class _NoOpLogger:
    """Lớp Logger rỗng (No-op logger) cho các tiến trình phụ khi chạy huấn luyện song song phân tán DDP."""

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        _ = metrics, step

    def log_summary(self, summary: dict[str, float]) -> None:
        _ = summary

    def finish(self) -> None:
        return


def _resolve_find_unused_parameters(config) -> bool:
    """Đọc cấu hình find_unused_parameters của DDP với giá trị mặc định là False."""
    return bool(getattr(config.training, "find_unused_parameters", False))


def _init_runtime(device_mode: str) -> tuple[torch.device, DistributedContext]:
    """
    Khởi tạo môi trường chạy: cấu hình đơn GPU/CPU hoặc song song phân tán DDP dựa trên torchrun.
    """
    if device_mode == "single":
        return get_device(), single_process_context()

    if not torch.cuda.is_available():
        raise RuntimeError("Chế độ device-mode=ddp yêu cầu hệ thống phải hỗ trợ CUDA (GPU).")

    ctx = parse_torchrun_env(os.environ)
    if not ctx.enabled:
        raise RuntimeError(
            "Chế độ device-mode=ddp yêu cầu WORLD_SIZE > 1. Vui lòng khởi chạy thông qua lệnh: torchrun --nproc_per_node=2 ..."
        )

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(ctx.local_rank)
    device = torch.device(f"cuda:{ctx.local_rank}")
    return device, ctx


def _cleanup_runtime(ctx: DistributedContext) -> None:
    """Giải phóng tài nguyên và hủy tiến trình phân tán khi kết thúc huấn luyện."""
    if ctx.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def build_dataloaders(
    config,
    dist_ctx: DistributedContext,
) -> tuple[DataLoader, DataLoader, DistributedSampler | None]:
    """Xây dựng và trả về các bộ nạp dữ liệu train/val DataLoader từ cấu hình."""
    root = Path(config.data.root)

    train_ds = ISICDataset(
        img_dir=root / "train" / "images",
        mask_dir=root / "train" / "masks",
        transform=get_transforms("train", config),
    )
    val_ds = ISICDataset(
        img_dir=root / "val" / "images",
        mask_dir=root / "val" / "masks",
        transform=get_transforms("val", config),
    )

    train_sampler: DistributedSampler | None = None
    val_sampler: DistributedSampler | None = None
    if dist_ctx.enabled:
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_ds,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=False,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers and config.data.num_workers > 0,
        drop_last=True,
    )
    val_batch_size = config.training.batch_size * int(
        getattr(config.data, "val_batch_size_multiplier", 2)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers and config.data.num_workers > 0,
    )

    if dist_ctx.is_main_process:
        log.info(f"Tập huấn luyện: {len(train_ds)} mẫu | Tập kiểm định: {len(val_ds)} mẫu")
    return train_loader, val_loader, train_sampler


# ---------------------------------------------------------------------------
# Luồng thực thi chính
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh CLI."""
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình phân đoạn tổn thương vùng da",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Đường dẫn đến file cấu hình YAML.\nVí dụ: configs/experiments/resnet34_unet_v1.yaml",
    )
    parser.add_argument(
        "--resume",
        "-r",
        default=None,
        help="Đường dẫn đến file last_checkpoint.pth để khôi phục trạng thái huấn luyện.",
    )
    parser.add_argument(
        "--device-mode",
        choices=["single", "ddp"],
        default="single",
        help=(
            "single: Huấn luyện trên thiết bị đơn GPU hoặc CPU (mặc định)\n"
            "ddp: Huấn luyện phân tán đa GPU sử dụng torchrun (Kaggle 2xT4)"
        ),
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="key.subkey=value",
        help="Tham số ghi đè dạng dot-notation. Ví dụ: data.root=/kaggle/input/isic",
    )
    return parser.parse_args()


def main() -> None:
    """Luồng thực thi chính."""
    args = parse_args()
    dist_ctx = single_process_context()
    run_logger: Logger | _NoOpLogger = _NoOpLogger()

    try:
        # 1. Nạp và áp dụng các cấu hình ghi đè
        config = load_config(args.config)
        config = override_config(config, args.overrides)

        # Tự động lấy tên file cấu hình làm tên thử nghiệm nếu chưa được định nghĩa
        if not config.logging.experiment_name:
            config["logging"]["experiment_name"] = Path(args.config).stem

        # 2. Thiết lập tính tái tạo kết quả
        set_seed(config.seed, deterministic=bool(getattr(config.training, "deterministic", True)))

        # 3. Khởi tạo môi trường phần cứng và tiến trình chạy
        device, dist_ctx = _init_runtime(args.device_mode)
        if dist_ctx.enabled and not dist_ctx.is_main_process:
            logging.getLogger().setLevel(logging.WARNING)
        if dist_ctx.is_main_process:
            log.info(
                "Thiết bị: %s | chế độ=%s | số tiến trình=%d",
                device,
                args.device_mode,
                dist_ctx.world_size,
            )

        # 4. Thiết lập thư mục lưu kết quả đầu ra
        output_dir = Path(config.output.dir) / config.logging.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        if dist_ctx.is_main_process:
            log.info(f"Thư mục lưu đầu ra: {output_dir}")

        # 5. Khởi tạo bộ ghi nhật ký (chỉ kích hoạt trên tiến trình chính)
        if dist_ctx.is_main_process:
            run_logger = Logger(config, output_dir)

        # 6. Khởi tạo các bộ nạp dữ liệu
        train_loader, val_loader, train_sampler = build_dataloaders(config, dist_ctx)

        # 7. Khởi tạo mô hình mạng
        model = create_model(config).to(device)
        if dist_ctx.enabled:
            model = DDP(
                model,
                device_ids=[dist_ctx.local_rank],
                output_device=dist_ctx.local_rank,
                find_unused_parameters=_resolve_find_unused_parameters(config),
            )

        model_ref = model.module if hasattr(model, "module") else model
        params = count_parameters(model_ref)
        if dist_ctx.is_main_process:
            log.info(
                f"Mô hình: {config.model.name} | encoder: {config.model.encoder_name} | "
                f"số tham số tối ưu={params['trainable']:,} ({params['size_mb']:.1f} MB)"
            )

        # 8. Khởi tạo lớp quản lý huấn luyện (Trainer)
        trainer = Trainer(
            model=model,
            config=config,
            device=device,
            log=run_logger,
            is_distributed=dist_ctx.enabled,
            is_main_process=dist_ctx.is_main_process,
        )

        # 8b. Khôi phục từ checkpoint cũ nếu có cấu hình
        start_epoch = 0
        if args.resume:
            ckpt = trainer.load_checkpoint(args.resume, resume=True)
            start_epoch = ckpt.get("epoch", -1) + 1
            if dist_ctx.is_main_process:
                log.info(f"Tiếp tục huấn luyện từ epoch số {start_epoch + 1}")

        # 9. Thực hiện tối ưu hóa
        summary = trainer.fit(
            train_loader,
            val_loader,
            output_dir,
            start_epoch=start_epoch,
            train_sampler=train_sampler,
        )
        if dist_ctx.enabled:
            dist.barrier()

        # 10. Hoàn tất ghi nhận nhật ký (chỉ tiến trình chính)
        if dist_ctx.is_main_process:
            run_logger.finish()

            best_dice = summary["best_metrics"].get("val_dice")
            best_epoch = summary.get("best_epoch")
            if best_dice is not None and best_epoch is not None:
                log.info(f"Hoàn tất huấn luyện. Kết quả val_dice tốt nhất={best_dice:.4f} tại epoch {best_epoch + 1}")
            elif best_dice is not None:
                log.info(f"Hoàn tất huấn luyện. Kết quả val_dice tốt nhất={best_dice:.4f}")
            else:
                log.warning("Quá trình huấn luyện kết thúc mà không đạt được sự cải thiện nào.")
    finally:
        _cleanup_runtime(dist_ctx)


if __name__ == "__main__":
    main()
