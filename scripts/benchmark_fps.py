#!/usr/bin/env python3
# =============================================================================
# Thư mục: scripts/benchmark_fps.py
# Chức năng: Đo đạc hiệu năng suy luận của mô hình (độ trễ latency và tốc độ xử lý FPS)
#            trên các batch size khác nhau.
# Hàm quan trọng:
#   - _is_out_of_memory_error: Kiểm tra xem lỗi Runtime có phải do tràn bộ nhớ GPU (OOM) hay không.
#   - _benchmark_batch_size: Thực hiện đo lường độ trễ và FPS cho một batch size cụ thể.
# =============================================================================

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import torch
import torch.utils.benchmark as benchmark

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.segmentation import create_model
from src.utils.checkpoint import load_state_dict_with_aux_compat
from src.utils.config import load_config, override_config
from src.utils.misc import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def _is_out_of_memory_error(exc: RuntimeError) -> bool:
    """
    Kiểm tra xem ngoại lệ Runtime có phải do lỗi tràn bộ nhớ (Out-Of-Memory) của CUDA hay không.

    Các tham số đầu vào:
        exc: Ngoại lệ Runtime được ném ra bởi PyTorch.

    Kết quả trả về:
        True nếu là lỗi OOM, ngược lại trả về False.
    """
    oom_error = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
    return isinstance(exc, oom_error) or "out of memory" in str(exc).lower()


def _benchmark_batch_size(
    model: torch.nn.Module,
    device: torch.device,
    height: int,
    width: int,
    batch_size: int,
) -> tuple[float, float] | None:
    """
    Đo đạc hiệu năng cho một batch size cụ thể. Trả về bộ giá trị (độ trễ, FPS).

    Các tham số đầu vào:
        model:      Mô hình phân đoạn đã nạp trọng số.
        device:     Thiết bị phần cứng để chạy đo đạc.
        height:     Chiều cao của ảnh đầu vào.
        width:      Chiều rộng của ảnh đầu vào.
        batch_size: Số lượng ảnh truyền vào trong một lượt suy luận.

    Kết quả trả về:
        Bộ giá trị ``(độ_trễ_ms, FPS)`` nếu đo đạc thành công, hoặc ``None`` nếu xảy ra lỗi tràn bộ nhớ.
    """
    x: torch.Tensor | None = None
    timer: benchmark.Timer | None = None

    try:
        x = torch.randn(batch_size, 3, height, width, device=device)
        timer = benchmark.Timer(
            stmt="with torch.inference_mode(): model(x)",
            globals={"model": model, "x": x, "torch": torch},
        )

        stats = timer.blocked_autorange(min_run_time=2.0)
        mean_latency = stats.mean
        return mean_latency * 1000, batch_size / mean_latency
    except RuntimeError as exc:
        if not _is_out_of_memory_error(exc):
            raise

        log.warning("Bỏ qua batch size %s do lỗi tràn bộ nhớ (OOM): %s", batch_size, exc)
        return None
    finally:
        del timer, x
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh CLI."""
    parser = argparse.ArgumentParser(
        description="Đo đạc hiệu năng suy luận FPS của mô hình",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", "-c", required=True, help="Đường dẫn đến file cấu hình YAML")
    parser.add_argument("--checkpoint", "-k", required=True, help="Đường dẫn đến file best_model.pth")
    parser.add_argument("--device", "-d", default="cpu", help="Thiết bị chạy đo đạc (mặc định: cpu)")
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 50, 100],
        help="Danh sách các batch size để chạy đo đạc (mặc định: 1 50 100)",
    )
    parser.add_argument("overrides", nargs="*", metavar="key.subkey=value")
    return parser.parse_args(argv)


def main() -> None:
    """Luồng thực thi chính."""
    args = parse_args()

    config = load_config(args.config)
    config = override_config(config, args.overrides)

    set_seed(config.seed)

    device = torch.device(args.device)
    log.info(f"Sử dụng thiết bị: {device}")

    # Khởi tạo mô hình
    model = create_model(config)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    load_state_dict_with_aux_compat(model, state, context=str(args.checkpoint))

    model = model.to(device)
    model.eval()

    log.info(f"Đã nạp mô hình lên thiết bị {device} và chuyển sang chế độ eval.")

    H, W = config.data.input_size

    results = []
    for bs in args.batch_sizes:
        log.info(f"Đang đo đạc cho batch size {bs}...")
        result = _benchmark_batch_size(model, device, H, W, bs)
        if result is None:
            continue
        results.append((bs, *result))

    print("\n--- Kết Quả Đo Hiệu Năng (Benchmark Results) ---")
    print(f"Thiết bị chạy : {args.device}")
    print(f"Kích thước ảnh: {H}x{W}")
    print("-" * 50)
    print(f"{'Batch Size':<12} | {'Độ trễ (ms)':>12} | {'Tốc độ (FPS)':>16}")
    print("-" * 50)

    for bs, latency, fps in results:
        print(f"{bs:<12} | {latency:>12.2f} | {fps:>16.2f}")

    print("-" * 50)


if __name__ == "__main__":
    main()
