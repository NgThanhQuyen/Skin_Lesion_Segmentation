# =============================================================================
# Thư mục: scripts/merge_and_split_isic.py
# Chức năng: Tiện ích gộp dữ liệu từ các tập train/val/test hiện có của ISIC, sau đó
#            chia ngẫu nhiên thành 3 tập test độc lập (test_1, test_2, test_3) phục vụ
#            cho các mục đích đánh giá chéo hiệu năng mô hình.
# Hàm quan trọng:
#   - _collect_pairs: Thu gom tất cả các cặp ảnh gốc và mặt nạ hợp lệ từ các thư mục con.
#   - run: Thực thi toàn bộ quy trình xáo trộn, chia tách kích thước và phân phối sao chép tệp tin.
# =============================================================================

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Định dạng hình ảnh được hỗ trợ
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Gộp dữ liệu ISIC từ train/val/test hiện có, sau đó xáo trộn ngẫu nhiên "
            "và phân chia thành 3 phần kiểm thử test_1/test_2/test_3 (mỗi phần gồm thư mục images/ và masks/)."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Đường dẫn thư mục gốc bộ dữ liệu chứa train/, val/, test/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/kaggle/working/isic_2018_task1_resplit"),
        help="Thư mục đầu ra lưu trữ kết quả phân chia cho test_1, test_2, test_3.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Hạt giống ngẫu nhiên giúp tái tạo kết quả phân chia.",
    )
    parser.add_argument(
        "--copy-mode",
        type=str,
        choices=["copy", "move"],
        default="copy",
        help="copy: giữ nguyên dữ liệu gốc, move: di chuyển hẳn tệp tin sang thư mục đầu ra.",
    )
    return parser.parse_args()


def _find_mask_for_image(mask_dir: Path, image_stem: str) -> Path | None:
    """
    Tìm kiếm file mặt nạ tương thích với tên gốc (stem) của ảnh gốc.

    Các tham số đầu vào:
        mask_dir:   Thư mục chứa mặt nạ phân đoạn.
        image_stem: Tên file gốc của ảnh cần đối chiếu.

    Kết quả trả về:
        Đường dẫn file mặt nạ hợp lệ đầu tiên tìm thấy, hoặc None nếu không tồn tại.
    """
    candidates = [p for p in mask_dir.glob(f"{image_stem}.*") if p.is_file()]
    if not candidates:
        return None
    return sorted(candidates)[0]


def _collect_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    """
    Thu thập tất cả các cặp (ảnh, mặt nạ) từ cả 3 tập train, val, test ban đầu.

    Các tham số đầu vào:
        input_dir: Đường dẫn thư mục gốc bộ dữ liệu.

    Kết quả trả về:
        Danh sách các cặp ảnh gốc và mặt nạ tương ứng.
    """
    pairs: list[tuple[Path, Path]] = []
    missing_masks: list[Path] = []

    for split_name in ("train", "val", "test"):
        image_dir = input_dir / split_name / "images"
        mask_dir = input_dir / split_name / "masks"

        if not image_dir.exists() or not mask_dir.exists():
            raise ValueError(
                f"Thiếu các thư mục con cần thiết của tập '{split_name}': {image_dir} hoặc {mask_dir}"
            )

        image_files = sorted(
            [p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTS]
        )

        for image_path in image_files:
            mask_path = _find_mask_for_image(mask_dir, image_path.stem)
            if mask_path is None:
                missing_masks.append(image_path)
                continue
            pairs.append((image_path, mask_path))

    if missing_masks:
        preview = "\n".join(str(p) for p in missing_masks[:10])
        raise ValueError(
            "Một số hình ảnh không tìm thấy mặt nạ tương thích. "
            f"Tổng số ảnh thiếu mặt nạ: {len(missing_masks)}\nVí dụ đường dẫn:\n{preview}"
        )

    if not pairs:
        raise ValueError("Không tìm thấy cặp ảnh và mặt nạ hợp lệ nào.")

    return pairs


def _split_sizes(total: int, n_splits: int) -> list[int]:
    """Tính toán kích thước chia tách cân bằng giữa các tập con."""
    base = total // n_splits
    rem = total % n_splits
    return [base + (1 if i < rem else 0) for i in range(n_splits)]


def _safe_dest_path(dest_dir: Path, file_name: str) -> Path:
    """Tạo đường dẫn đích không bị trùng lặp tên file (tự động tạo tên phụ nếu đã tồn tại)."""
    dest = dest_dir / file_name
    if not dest.exists():
        return dest

    stem = dest.stem
    suffix = dest.suffix
    idx = 1
    while True:
        candidate = dest_dir / f"{stem}__dup{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def _transfer_file(src: Path, dst: Path, mode: str) -> None:
    """Thực hiện truyền tệp tin bằng cách sao chép (copy) hoặc di chuyển (move)."""
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


def run(input_dir: Path, output_dir: Path, seed: int, copy_mode: str) -> None:
    """
    Thực thi toàn bộ kịch bản gộp và phân chia lại dữ liệu.

    Các tham số đầu vào:
        input_dir:   Thư mục gốc chứa train/val/test ban đầu.
        output_dir:  Thư mục đầu ra lưu kết quả phân chia.
        seed:        Hạt giống ngẫu nhiên.
        copy_mode:   Chế độ truyền file: copy hoặc move.
    """
    pairs = _collect_pairs(input_dir)
    rng = random.Random(seed)
    rng.shuffle(pairs)

    sizes = _split_sizes(len(pairs), 3)
    split_names = ["test_1", "test_2", "test_3"]

    for split_name in split_names:
        (output_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split_name / "masks").mkdir(parents=True, exist_ok=True)

    start = 0
    for split_name, size in zip(split_names, sizes, strict=True):
        end = start + size
        split_pairs = pairs[start:end]
        start = end

        image_out = output_dir / split_name / "images"
        mask_out = output_dir / split_name / "masks"

        for img_src, mask_src in split_pairs:
            img_dst = _safe_dest_path(image_out, img_src.name)
            mask_dst = _safe_dest_path(mask_out, mask_src.name)
            _transfer_file(img_src, img_dst, copy_mode)
            _transfer_file(mask_src, mask_dst, copy_mode)

    print("Hoàn tất quy trình phân chia lại dữ liệu")
    print(f"Thư mục nguồn      : {input_dir}")
    print(f"Thư mục kết quả    : {output_dir}")
    print(f"Tổng số cặp xử lý  : {len(pairs)}")
    print(f"Kích thước chia    : test_1={sizes[0]}, test_2={sizes[1]}, test_3={sizes[2]}")
    print(f"Chế độ truyền file : {copy_mode}")


def main() -> None:
    """Luồng thực thi chính."""
    args = _parse_args()
    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        copy_mode=args.copy_mode,
    )


if __name__ == "__main__":
    main()
