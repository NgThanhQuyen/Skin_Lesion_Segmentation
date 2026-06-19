#!/usr/bin/env python3
# =============================================================================
# Thư mục: scripts/prepare_data.py
# Chức năng: Tiền xử lý dữ liệu và phân chia bộ dữ liệu hình ảnh da liễu ISIC thành
#            các tập huấn luyện (Train), kiểm định (Val) và kiểm thử (Test).
# Hàm quan trọng:
#   - _get_valid_pairs: Đối chiếu ảnh gốc và mặt nạ nhị phân tương ứng dựa trên tên file.
#   - _split_pairs: Xáo trộn ngẫu nhiên và phân chia các cặp dữ liệu theo tỷ lệ cấu hình.
#   - _copy_split: Sao chép các cặp dữ liệu vào các thư mục phân tách tương ứng.
# =============================================================================

from __future__ import annotations

import argparse
import logging
import random
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Các định dạng ảnh được hỗ trợ đầu vào
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _get_valid_pairs(img_dir: Path, mask_dir: Path) -> tuple[list[tuple[Path, Path]], int]:
    """
    Tìm kiếm và ghép cặp ảnh gốc với mặt nạ tương ứng có cùng tên gốc (stem).

    Các tham số đầu vào:
        img_dir:  Thư mục chứa ảnh gốc.
        mask_dir: Thư mục chứa mặt nạ phân đoạn định dạng PNG.

    Kết quả trả về:
        Bộ giá trị gồm:
        - Danh sách các tuple dạng (đường_dẫn_ảnh, đường_dẫn_mặt_nạ) hợp lệ.
        - Số lượng ảnh bị bỏ qua do không tìm thấy mặt nạ tương ứng.
    """
    pairs: list[tuple[Path, Path]] = []
    skipped = 0

    for img_path in sorted(img_dir.iterdir()):
        if not img_path.is_file() or img_path.suffix.lower() not in _IMAGE_EXTS:
            continue

        stem = img_path.stem
        mask_path = mask_dir / f"{stem}.png"
        if mask_path.exists():
            pairs.append((img_path, mask_path))
        else:
            skipped += 1
            log.warning(f"Không tìm thấy mặt nạ cho ảnh {img_path.name} -> bỏ qua mẫu này")

    return pairs, skipped


def _split_pairs(
    pairs: list[tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[tuple[Path, Path]]]:
    """
    Xáo trộn ngẫu nhiên và phân chia các cặp dữ liệu thành các tập train/val/test.

    Các tham số đầu vào:
        pairs:       Danh sách các cặp (đường_dẫn_ảnh, đường_dẫn_mặt_nạ) hợp lệ.
        train_ratio: Tỷ lệ chia tập huấn luyện (train).
        val_ratio:   Tỷ lệ chia tập kiểm định (val).
        test_ratio:  Tỷ lệ chia tập kiểm thử (test).
        seed:        Hạt giống ngẫu nhiên giúp tái tạo kết quả phân chia.

    Kết quả trả về:
        Từ điển ánh xạ từ tên tập dữ liệu (train/val/test) sang danh sách các cặp tương ứng.
    """
    shuffled = pairs.copy()
    random.Random(seed).shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_pairs = shuffled[:n_train]
    val_pairs = shuffled[n_train : n_train + n_val]
    test_pairs = shuffled[n_train + n_val :]

    assert len(test_pairs) == n_test

    return {
        "train": train_pairs,
        "val": val_pairs,
        "test": test_pairs,
    }


def _copy_split(
    pairs: list[tuple[Path, Path]],
    img_dst: Path,
    mask_dst: Path,
    split_name: str,
) -> None:
    """
    Sao chép các cặp dữ liệu đã phân chia vào đúng thư mục đích.

    Các tham số đầu vào:
        pairs:      Danh sách các cặp cần sao chép.
        img_dst:    Thư mục đích lưu ảnh gốc.
        mask_dst:   Thư mục đích lưu ảnh mặt nạ.
        split_name: Tên tập dữ liệu phục vụ mục đích log tiến độ.
    """
    img_dst.mkdir(parents=True, exist_ok=True)
    mask_dst.mkdir(parents=True, exist_ok=True)

    for i, (img_path, mask_path) in enumerate(pairs, 1):
        shutil.copy2(img_path, img_dst / img_path.name)
        shutil.copy2(mask_path, mask_dst / mask_path.name)
        if i % 200 == 0:
            log.info(f"  {split_name}: đã sao chép {i}/{len(pairs)}...")

    log.info(f"  {split_name}: hoàn tất xử lý {len(pairs)} tệp tin.")


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh CLI."""
    parser = argparse.ArgumentParser(
        description="Chuẩn bị bộ dữ liệu phân đoạn từ một thư mục ảnh gốc và một thư mục mặt nạ.",
    )
    parser.add_argument(
        "--data-dir",
        default="data/data-HA10000-remove-hair",
        help="Thư mục gốc chứa bộ dữ liệu gốc (mặc định: data/data-HA10000-remove-hair)",
    )
    parser.add_argument(
        "--images-subdir",
        default="remove-hair/images",
        help="Đường dẫn thư mục con chứa ảnh gốc tính từ --data-dir (mặc định: remove-hair/images)",
    )
    parser.add_argument(
        "--masks-subdir",
        default="masks",
        help="Đường dẫn thư mục con chứa ảnh mặt nạ tính từ --data-dir (mặc định: masks)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed",
        help="Thư mục lưu kết quả sau phân chia (mặc định: data/processed)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Tỷ lệ tập huấn luyện (mặc định: 0.8)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Tỷ lệ tập kiểm định (mặc định: 0.1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Tỷ lệ tập kiểm thử (mặc định: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Hạt giống ngẫu nhiên cho việc phân chia dữ liệu (mặc định: 42)",
    )
    return parser.parse_args()


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """Xác thực các tỷ lệ phân chia tập dữ liệu đầu vào."""
    ratios = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }
    for name, value in ratios.items():
        if value < 0:
            raise ValueError(f"Giá trị {name} bắt buộc phải lớn hơn hoặc bằng 0, nhận được: {value}")

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"Tổng các tỷ lệ train_ratio + val_ratio + test_ratio phải bằng 1.0 (hiện tại là: {total:.6f})"
        )


def main() -> None:
    """Luồng thực thi chính."""
    args = parse_args()

    data_dir = Path(args.data_dir)
    img_dir = data_dir / args.images_subdir
    mask_dir = data_dir / args.masks_subdir
    out_dir = Path(args.out_dir)

    try:
        _validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)
    except ValueError as exc:
        log.error(str(exc))
        sys.exit(1)

    if not img_dir.exists():
        log.error(f"Không tìm thấy thư mục chứa ảnh gốc: {img_dir}")
        sys.exit(1)
    if not mask_dir.exists():
        log.error(f"Không tìm thấy thư mục chứa ảnh mặt nạ: {mask_dir}")
        sys.exit(1)

    print("=" * 60)
    print("CHUẨN BỊ BỘ DỮ LIỆU TỪ THƯ MỤC ẢNH VÀ MẶT NẠ GỐC")
    print("=" * 60)
    log.info(f"Thư mục ảnh gốc: {img_dir}")
    log.info(f"Thư mục mặt nạ : {mask_dir}")
    log.info(
        "Tỷ lệ phân chia: train=%.2f val=%.2f test=%.2f (hạt giống=%d)",
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.seed,
    )

    pairs, skipped = _get_valid_pairs(img_dir, mask_dir)
    if not pairs:
        log.error("Không tìm thấy cặp ảnh và mặt nạ hợp lệ nào để xử lý.")
        sys.exit(1)

    split_map = _split_pairs(
        pairs,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    split_counts: dict[str, int] = {}
    for split_name, split_pairs in split_map.items():
        split_counts[split_name] = len(split_pairs)
        log.info(f"Đang xử lý tập [{split_name}]: gồm {len(split_pairs)} cặp dữ liệu")
        _copy_split(
            split_pairs,
            out_dir / split_name / "images",
            out_dir / split_name / "masks",
            split_name,
        )

    grand_total = sum(split_counts.values())
    print("=" * 60)
    for split_name in ("train", "val", "test"):
        count = split_counts.get(split_name, 0)
        print(f"  {split_name:<6}: {count:>5}  ({count / grand_total:.1%})")
    print(f"  {'Tổng số':<6}: {grand_total:>5}")
    print("=" * 60)

    log.info(f"Số lượng ảnh bị bỏ qua do thiếu mặt nạ tương ứng: {skipped}")
    log.info(f"Hoàn tất. Thư mục dữ liệu kết quả: {out_dir.resolve()}")
    log.info(
        "Bước tiếp theo: chạy lệnh huấn luyện: python scripts/train.py --config configs/experiments/resnet34_unet_v1.yaml"
    )


if __name__ == "__main__":
    main()
