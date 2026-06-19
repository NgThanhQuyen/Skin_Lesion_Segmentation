# =============================================================================
# Thư mục: src/data/transforms.py
# Chức năng: Định nghĩa pipeline tăng cường và biến đổi dữ liệu (transforms) sử dụng thư viện Albumentations.
# Hàm quan trọng:
#   - get_transforms: Trả về chuỗi biến đổi hình ảnh và mặt nạ tùy thuộc vào phân tách dữ liệu (train/val/test).
# =============================================================================

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Giá trị trung bình (mean) và độ lệch chuẩn (std) của bộ dữ liệu ImageNet (kênh RGB)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Các nhãn đại diện cho tập huấn luyện và tập kiểm định/kiểm thử
_TRAIN_SPLITS = {"train"}
_VAL_SPLITS = {"val", "valid", "validation", "test", "predict"}


def get_transforms(split: str, config) -> A.Compose:
    """
    Trả về một pipeline Albumentations Compose tương ứng với phân tách dữ liệu được yêu cầu.

    Các tham số đầu vào:
        split:  Tên phân tách dữ liệu (như "train", "val", "test", "predict").
                Các giá trị nằm trong _VAL_SPLITS chỉ được áp dụng căn chỉnh kích thước và chuẩn hóa.
                Giá trị "train" sẽ được áp dụng thêm các phép tăng cường hình học và màu sắc.
        config: Đối tượng cấu hình dự án (thuộc lớp src.utils.config.Config).
                Cần cung cấp config.data.input_size dưới dạng [H, W].

    Kết quả trả về:
        Đối tượng albumentations.Compose đã được cấu hình.
        Pipeline luôn trả về một từ điển chứa hai khóa: "image" (Tensor PyTorch float32,
        kích thước C x H x W) và "mask" (Tensor PyTorch float32, kích thước H x W).

    Ngoại lệ:
        TypeError: Nếu kích thước config.data.input_size không chứa đủ 2 phần tử số nguyên.
        ValueError: Nếu giá trị split không nằm trong danh sách được hỗ trợ.
    """
    input_size = config.data.input_size
    if len(input_size) != 2:
        raise TypeError(f"config.data.input_size phải có định dạng [H, W], giá trị nhận được: {input_size}")
    height, width = int(input_size[0]), int(input_size[1])

    # ------------------------------------------------------------------
    # Phần xử lý chung ở cuối pipeline cho mọi tập dữ liệu
    # ------------------------------------------------------------------
    _tail = [
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),  # Chuyển đổi ảnh: HWC uint8 -> CHW float32 tensor
        # Chuyển đổi mặt nạ: HW float32 -> HW float32 tensor (không thêm chiều mới)
    ]

    normalized_split = split.lower()
    allowed_splits = sorted(_TRAIN_SPLITS | _VAL_SPLITS)

    if normalized_split not in _TRAIN_SPLITS and normalized_split not in _VAL_SPLITS:
        raise ValueError(
            f"Phân tách dữ liệu '{split}' không hợp lệ. Các giá trị được phép: {allowed_splits}"
        )

    # ------------------------------------------------------------------
    # Tập kiểm định / kiểm thử — không áp dụng tăng cường dữ liệu
    # ------------------------------------------------------------------
    if normalized_split in _VAL_SPLITS:
        return A.Compose(
            [A.Resize(height, width)] + _tail,
        )

    # ------------------------------------------------------------------
    # Tập huấn luyện — áp dụng các phép tăng cường hình học và màu sắc
    # ------------------------------------------------------------------
    return A.Compose(
        [
            # --- Biến đổi hình học ---
            A.Resize(height, width),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)},
                rotate=(-15, 15),
                border_mode=0,  # Điền giá trị hằng số (màu đen) ở vùng biên
                p=0.5,
            ),
            # --- Biến dạng cấu trúc hình ảnh ---
            A.ElasticTransform(alpha=120.0, sigma=8.0, p=0.2),
            A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.2),
            # --- Biến đổi màu sắc / cường độ sáng (chỉ tác động lên ảnh gốc) ---
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5,
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=10,
                p=0.3,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(p=0.2),
            # --- Phép khử nhiễu / loại bỏ vùng ảnh ---
            A.CoarseDropout(
                num_holes_range=(1, 8),
                hole_height_range=(12, 25),  # Chiếm khoảng 5-10% kích thước ảnh 256px
                hole_width_range=(12, 25),   # Chiếm khoảng 5-10% kích thước ảnh 256px
                fill=0,
                fill_mask=0,
                p=0.2,
            ),
        ]
        + _tail,
    )
