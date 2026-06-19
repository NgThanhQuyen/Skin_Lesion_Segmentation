# =============================================================================
# Thư mục: src/data/dataset.py
# Chức năng: Định nghĩa lớp dữ liệu (Dataset) cho bộ dữ liệu phân đoạn tổn thương da ISIC.
# Lớp quan trọng:
#   - ISICDataset: Kế thừa từ torch.utils.data.Dataset để tải ảnh dermoscopy và mặt nạ (mask) tương ứng.
# =============================================================================

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

# Các định dạng ảnh được hỗ trợ (không phân biệt chữ hoa, chữ thường)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ISICDataset(Dataset):
    """
    Lớp Dataset PyTorch cho bài toán phân đoạn dữ liệu ISIC Challenge Task 1.

    Lớp này thực hiện đọc các ảnh dermoscopy RGB và các mặt nạ phân đoạn nhị phân tương ứng.
    Có thể áp dụng các phép biến đổi (transform) của Albumentations đồng thời lên cả ảnh và mặt nạ.

    Các tham số đầu vào:
        img_dir:   Thư mục chứa ảnh gốc.
        mask_dir:  Thư mục chứa mặt nạ phân đoạn nhị phân.
        transform: Đối tượng ``Compose`` của Albumentations để thực hiện tăng cường dữ liệu.
                   Nếu truyền ``None``, dữ liệu thô dạng numpy array sẽ được trả về.

    Kết quả trả về (cho mỗi mẫu):
        image: torch.Tensor kích thước (3, H, W), kiểu float32.
        mask:  torch.Tensor kích thước (1, H, W), kiểu float32, giá trị thuộc khoảng {0, 1}.

    Ngoại lệ:
        FileNotFoundError: Nếu thư mục ``img_dir`` hoặc ``mask_dir`` không tồn tại.
        ValueError:        Nếu không tìm thấy cặp ảnh và mặt nạ hợp lệ nào.
    """

    def __init__(
        self,
        img_dir: str | Path,
        mask_dir: str | Path,
        transform=None,
    ) -> None:
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        # Xác thực sự tồn tại của các thư mục dữ liệu
        if not self.img_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục ảnh: {self.img_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Không tìm thấy thư mục mặt nạ: {self.mask_dir}")

        # Gom nhóm đường dẫn ảnh và mặt nạ thành từng cặp tương ứng
        self.samples: list[tuple[Path, Path]] = self._collect_samples()

        if not self.samples:
            raise ValueError(
                f"Không tìm thấy cặp ảnh và mặt nạ hợp lệ nào.\n"
                f"  Thư mục ảnh:  {self.img_dir}\n"
                f"  Thư mục mặt nạ: {self.mask_dir}\n"
                f"Vui lòng kiểm tra lại sự tồn tại của dữ liệu và định dạng file."
            )

        log.debug(f"ISICDataset: Đã tải {len(self.samples)} mẫu dữ liệu từ {self.img_dir}")

    def _collect_samples(self) -> list[tuple[Path, Path]]:
        """
        Tìm kiếm và ghép cặp (đường dẫn ảnh, đường dẫn mặt nạ) tương ứng.

        Chiến lược ghép cặp (theo thứ tự ưu tiên):
        1. Khớp chính xác tên file: ảnh ``ISIC_0024306.jpg`` -> mặt nạ ``ISIC_0024306.png``
        2. Hậu tố ``_segmentation``: mặt nạ ``ISIC_0024306_segmentation.png``

        Cả hai chiến lược này đều xử lý được cấu trúc thư mục mặc định của ISIC cũng như
        cấu trúc đầu ra đã qua tiền xử lý của script ``prepare_data.py``.
        """
        samples: list[tuple[Path, Path]] = []

        # Tạo bảng tra cứu: tên mặt nạ (dạng chữ thường) -> đường dẫn mặt nạ
        mask_lookup: dict[str, Path] = {}
        for mask_path in sorted(self.mask_dir.iterdir()):
            if mask_path.suffix.lower() in IMAGE_EXTS:
                mask_lookup[mask_path.stem.lower()] = mask_path

        for img_path in sorted(self.img_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTS:
                continue

            stem = img_path.stem.lower()

            # Chiến lược 1 – khớp chính xác tên file
            mask_path = mask_lookup.get(stem)

            # Chiến lược 2 – định dạng đặt tên chính thức của ISIC: <tên_ảnh>_segmentation
            if mask_path is None:
                mask_path = mask_lookup.get(f"{stem}_segmentation")

            if mask_path is None:
                log.warning(f"Không tìm thấy mặt nạ cho ảnh: {img_path.name} — bỏ qua mẫu này.")
                continue

            samples.append((img_path, mask_path))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        """
        Tải và trả về một cặp (ảnh, mặt nạ) dựa vào chỉ mục.

        Kết quả trả về:
            image: torch.Tensor kích thước (3, H, W) kiểu float32
            mask:  torch.Tensor kích thước (1, H, W) kiểu float32, giá trị thuộc khoảng {0.0, 1.0}
        """
        img_path, mask_path = self.samples[index]

        # Đọc ảnh gốc dạng RGB numpy array (H, W, 3) kiểu uint8
        image = np.array(Image.open(img_path).convert("RGB"), dtype=np.uint8)

        # Đọc mặt nạ ở dạng ảnh xám (grayscale) và nhị phân hóa về khoảng {0, 1} kiểu float32
        mask_raw = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        mask = (mask_raw > 127).astype(np.float32)  # (H, W)

        # Áp dụng các phép tăng cường dữ liệu và tiền xử lý nếu có
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]  # Tensor (3, H, W) sau khi áp dụng ToTensorV2
            mask = augmented["mask"]  # Tensor (H, W) sau khi áp dụng ToTensorV2

            # Đảm bảo mặt nạ có đủ chiều kênh: (H, W) -> (1, H, W)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)

        return image, mask

    def __repr__(self) -> str:
        return (
            f"ISICDataset("
            f"n_samples={len(self)}, "
            f"img_dir={self.img_dir}, "
            f"mask_dir={self.mask_dir}, "
            f"transform={'đã thiết lập' if self.transform else 'None'})"
        )
