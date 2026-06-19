# =============================================================================
# Thư mục: src/models/segmentation.py
# Chức năng: Bộ sinh mô hình (Model Factory) cho các kiến trúc phân đoạn ảnh.
# Lớp quan trọng:
#   - DeepLabV3Wrapper: Lớp bao bọc đầu ra của torchvision DeepLabV3 để trả về logits thô.
# Hàm quan trọng:
#   - create_model: Hàm chính để xây dựng mô hình dựa trên cấu hình đầu vào.
# =============================================================================

from __future__ import annotations

import logging
import torch
import torch.nn as nn

_log = logging.getLogger(__name__)


# =============================================================================
# Các lớp bao bọc (Wrappers)
# =============================================================================


class DeepLabV3Wrapper(nn.Module):
    """
    Bao bọc đầu ra dạng từ điển của torchvision DeepLabV3 thành tensor logits thô.

    Mặc định torchvision trả về OrderedDict({'out': tensor, 'aux': tensor}),
    nhưng lớp Trainer, hàm loss và các metrics yêu cầu trực tiếp tensor dạng (B, C, H, W).
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


# =============================================================================
# Các hàm xây dựng mô hình (Builders)
# =============================================================================


def _build_unet(config) -> nn.Module:
    """Xây dựng mô hình U-Net sử dụng mạng xương sống (encoder backbone) từ thư viện SMP."""
    import segmentation_models_pytorch as smp

    m = config.model
    return smp.Unet(
        encoder_name=m.encoder_name,
        encoder_weights=m.encoder_weights,
        decoder_attention_type=m.decoder_attention_type,
        decoder_channels=list(m.decoder_channels),
        in_channels=m.in_channels,
        classes=m.classes,
    )


def _build_unet_original(config) -> nn.Module:
    """
    Xây dựng mô hình U-Net nguyên bản (chỉ gồm các khối mã hóa và giải mã cơ bản, không có tiền huấn luyện).

    Các trường cấu hình hỗ trợ:
        - model.in_channels (kiểu số nguyên)
        - model.classes (kiểu số nguyên)
        - model.base_channels (kiểu số nguyên tùy chọn, mặc định=64)

    Các trường kế thừa từ cấu hình gốc bị bỏ qua (chỉ mang tính thông tin):
        - encoder_name
        - encoder_weights
        - decoder_attention_type
        - decoder_channels
    """
    from src.models.unet_original import UNetOriginal

    m = config.model
    base_channels = int(getattr(m, "base_channels", 64))

    ignored_fields = (
        "encoder_name",
        "encoder_weights",
        "decoder_attention_type",
        "decoder_channels",
    )
    for field in ignored_fields:
        value = getattr(m, field, None)
        if value is not None:
            _log.warning(
                "Mô hình unet_original sẽ bỏ qua cấu hình model.%s=%r. "
                "Hãy đặt giá trị thành null trong file YAML cấu hình thử nghiệm để tắt cảnh báo này.",
                field,
                value,
            )

    return UNetOriginal(
        in_channels=int(m.in_channels),
        num_classes=int(m.classes),
        base_channels=base_channels,
    )


def _build_deeplabv3(config) -> nn.Module:
    """
    Xây dựng mô hình DeepLabV3 với mạng xương sống MobileNetV3-Large (từ thư viện torchvision).

    Ánh xạ cấu hình:
        encoder_weights: "imagenet" -> sử dụng trọng số tiền huấn luyện ImageNet cho backbone.
                         null       -> huấn luyện mô hình từ đầu (từ scratch).
        classes:         Số lượng lớp đầu ra (mặc định=1 cho phân đoạn nhị phân).

    Ràng buộc:
        - in_channels phải bằng 3 (do torchvision mặc định cấu hình đầu vào RGB).
        - encoder_name chỉ mang tính thông tin; hàm này luôn sử dụng cấu hình mặc định
          deeplabv3_mobilenet_v3_large bất kể giá trị cấu hình được truyền vào là gì.
    """
    from torchvision.models import MobileNet_V3_Large_Weights
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

    m = config.model

    # Kiểm tra ràng buộc: torchvision không cho phép thay đổi in_channels cho cấu hình này
    if m.in_channels != 3:
        raise ValueError(
            f"Mô hình DeepLabV3 của torchvision chỉ hỗ trợ in_channels=3 (RGB). "
            f"Giá trị nhận được: in_channels={m.in_channels}. "
            "Vui lòng sử dụng các mô hình của thư viện SMP nếu cần số kênh đầu vào khác."
        )

    # Cảnh báo: cấu hình encoder_name bị bỏ qua do hàm này mặc định dùng mobilenet_v3_large
    _EXPECTED_ENCODER = "mobilenet_v3_large"
    actual_encoder = getattr(m, "encoder_name", _EXPECTED_ENCODER)
    if actual_encoder != _EXPECTED_ENCODER:
        _log.warning(
            "Hàm _build_deeplabv3 chỉ hỗ trợ xây dựng deeplabv3_mobilenet_v3_large. "
            "Cấu hình encoder_name=%r bị bỏ qua (kỳ vọng %r). "
            "Hãy viết thêm hàm xây dựng mới nếu bạn cần sử dụng mạng xương sống khác.",
            actual_encoder,
            _EXPECTED_ENCODER,
        )

    # Xử lý trọng số tiền huấn luyện:
    # - "imagenet": Chỉ sử dụng trọng số đã huấn luyện cho backbone.
    # - None:       Huấn luyện từ đầu.
    encoder_weights = m.encoder_weights
    if isinstance(encoder_weights, str):
        encoder_weights = encoder_weights.lower()

    if encoder_weights == "imagenet":
        weights_backbone = MobileNet_V3_Large_Weights.IMAGENET1K_V1
    elif encoder_weights is None:
        weights_backbone = None
    else:
        raise ValueError(
            "Mô hình DeepLabV3 chỉ hỗ trợ model.encoder_weights là 'imagenet' hoặc null. "
            f"Giá trị nhận được: {m.encoder_weights!r}"
        )

    model = deeplabv3_mobilenet_v3_large(
        weights=None,
        weights_backbone=weights_backbone,
        num_classes=m.classes,
        aux_loss=False,
    )

    return DeepLabV3Wrapper(model)


def _build_deeplabv3plus(config) -> nn.Module:
    """
    Xây dựng mô hình DeepLabV3+ sử dụng các mạng xương sống hỗ trợ bởi thư viện SMP.

    Các trường cấu hình hỗ trợ:
        - model.encoder_name
        - model.encoder_weights
        - model.in_channels
        - model.classes
        - model.decoder_channels (kiểu số nguyên, mặc định=256)
        - model.encoder_output_stride (tùy chọn, giá trị bằng 8 hoặc 16)

    Các trường cấu hình kế thừa từ file base bị bỏ qua:
        - decoder_attention_type
    """
    import segmentation_models_pytorch as smp

    m = config.model

    decoder_channels = getattr(m, "decoder_channels", 256)
    if not isinstance(decoder_channels, int):
        raise ValueError(
            "Mô hình DeepLabV3+ yêu cầu model.decoder_channels phải là kiểu số nguyên (int). "
            f"Giá trị nhận được: {decoder_channels!r}"
        )
    if decoder_channels <= 0:
        raise ValueError(
            "Mô hình DeepLabV3+ yêu cầu model.decoder_channels phải > 0. "
            f"Giá trị nhận được: {decoder_channels}"
        )

    output_stride = int(getattr(m, "encoder_output_stride", 16))
    if output_stride not in (8, 16):
        raise ValueError(
            "Mô hình DeepLabV3+ chỉ hỗ trợ model.encoder_output_stride bằng 8 hoặc 16. "
            f"Giá trị nhận được: {output_stride}"
        )

    decoder_attention_type = getattr(m, "decoder_attention_type", None)
    if decoder_attention_type is not None:
        _log.warning(
            "Mô hình deeplabv3plus (SMP) sẽ bỏ qua cấu hình model.decoder_attention_type=%r.",
            decoder_attention_type,
        )

    return smp.DeepLabV3Plus(
        encoder_name=m.encoder_name,
        encoder_weights=m.encoder_weights,
        encoder_output_stride=output_stride,
        decoder_channels=decoder_channels,
        in_channels=m.in_channels,
        classes=m.classes,
    )


def _build_transunet(config) -> nn.Module:
    """
    Xây dựng mô hình TransUNet R50-ViT-B_16.

    Các trường cấu hình hỗ trợ:
        - model.in_channels (bắt buộc bằng 3)
        - model.classes (bắt buộc bằng 1)
        - model.transunet_variant (hiện tại chỉ hỗ trợ giá trị "r50_vit_b16")
        - model.encoder_weights (giá trị là "imagenet" hoặc null)
        - model.pretrained_path (bắt buộc phải điền nếu encoder_weights="imagenet")
        - model.decoder_channels (danh sách/tuple gồm 4 số nguyên dương)
        - model.n_skip (số lượng kết nối tắt, nằm trong khoảng [0, 3])
        - model.skip_channels (danh sách/tuple tùy chọn gồm 4 số nguyên)
        - model.vit_hidden_size (tùy chọn)
        - model.vit_mlp_dim (tùy chọn)
        - model.vit_num_heads (tùy chọn)
        - model.vit_num_layers (tùy chọn)
        - model.vit_dropout_rate (tùy chọn)
        - model.vit_attention_dropout_rate (tùy chọn)

    Các trường cấu hình kế thừa từ file base bị bỏ qua:
        - model.decoder_attention_type
        - model.encoder_name
    """
    from pathlib import Path

    from src.models.transunet import TransUNet, build_r50_vit_b16_config

    m = config.model

    if int(m.in_channels) != 3:
        raise ValueError(
            "Mô hình TransUNet (R50-ViT-B_16) chỉ hỗ trợ cấu hình model.in_channels=3 (RGB). "
            f"Giá trị nhận được: {m.in_channels}"
        )
    if int(m.classes) != 1:
        raise ValueError(
            "Dự án hiện tại dùng phân đoạn nhị phân nên TransUNet yêu cầu "
            f"model.classes=1. Giá trị nhận được: {m.classes}"
        )

    variant = str(getattr(m, "transunet_variant", "r50_vit_b16")).lower()
    if variant != "r50_vit_b16":
        raise ValueError(
            "TransUNet hiện tại chỉ hỗ trợ model.transunet_variant='r50_vit_b16'. "
            f"Giá trị nhận được: {variant!r}"
        )

    decoder_channels = getattr(m, "decoder_channels", [256, 128, 64, 16])
    if not isinstance(decoder_channels, (list, tuple)) or len(decoder_channels) != 4:
        raise ValueError(
            "TransUNet yêu cầu model.decoder_channels phải là danh sách/tuple gồm 4 phần tử. "
            f"Giá trị nhận được: {decoder_channels!r}"
        )
    decoder_channels = tuple(int(v) for v in decoder_channels)
    if any(v <= 0 for v in decoder_channels):
        raise ValueError(
            "TransUNet yêu cầu tất cả các phần tử trong model.decoder_channels phải > 0. "
            f"Giá trị nhận được: {decoder_channels!r}"
        )

    skip_channels = getattr(m, "skip_channels", [512, 256, 64, 16])
    if not isinstance(skip_channels, (list, tuple)) or len(skip_channels) != 4:
        raise ValueError(
            "TransUNet yêu cầu model.skip_channels phải là danh sách/tuple gồm 4 phần tử. "
            f"Giá trị nhận được: {skip_channels!r}"
        )
    skip_channels = tuple(int(v) for v in skip_channels)

    n_skip = int(getattr(m, "n_skip", 3))
    if n_skip < 0 or n_skip > 3:
        raise ValueError(
            "TransUNet yêu cầu model.n_skip nằm trong khoảng [0, 3]. "
            f"Giá trị nhận được: {n_skip}"
        )

    input_size = getattr(config.data, "input_size", [256, 256])
    if not isinstance(input_size, (list, tuple)) or len(input_size) != 2:
        raise ValueError(
            "TransUNet yêu cầu cấu hình data.input_size có dạng [H, W]. "
            f"Giá trị nhận được: {input_size!r}"
        )
    height, width = int(input_size[0]), int(input_size[1])
    if height != width:
        raise ValueError(
            "TransUNet yêu cầu hình ảnh đầu vào phải là hình vuông để tạo lưới token chính xác. "
            f"Giá trị nhận được: data.input_size=[{height}, {width}]"
        )

    decoder_attention_type = getattr(m, "decoder_attention_type", None)
    if decoder_attention_type is not None:
        _log.warning(
            "Mô hình transunet sẽ bỏ qua cấu hình model.decoder_attention_type=%r.",
            decoder_attention_type,
        )

    encoder_name = getattr(m, "encoder_name", None)
    if encoder_name not in (None, "r50_vit_b16"):
        _log.warning(
            "Hàm _build_transunet chỉ hỗ trợ cấu hình R50-ViT-B_16. Cấu hình encoder_name=%r bị bỏ qua.",
            encoder_name,
        )

    model_cfg = build_r50_vit_b16_config(
        n_classes=1,
        decoder_channels=decoder_channels,
        n_skip=n_skip,
        hidden_size=int(getattr(m, "vit_hidden_size", 768)),
        mlp_dim=int(getattr(m, "vit_mlp_dim", 3072)),
        num_heads=int(getattr(m, "vit_num_heads", 12)),
        num_layers=int(getattr(m, "vit_num_layers", 12)),
        dropout_rate=float(getattr(m, "vit_dropout_rate", 0.1)),
        attention_dropout_rate=float(getattr(m, "vit_attention_dropout_rate", 0.0)),
        skip_channels=skip_channels,
    )
    model = TransUNet(config=model_cfg, img_size=(height, width), in_channels=3, vis=False)

    encoder_weights = m.encoder_weights
    if isinstance(encoder_weights, str):
        encoder_weights = encoder_weights.lower()

    if encoder_weights == "imagenet":
        pretrained_path = getattr(m, "pretrained_path", None)
        if pretrained_path is None:
            raise ValueError(
                "Mô hình TransUNet với cấu hình encoder_weights='imagenet' bắt buộc phải có model.pretrained_path "
                "chỉ tới file trọng số định dạng R50+ViT-B_16.npz"
            )
        ckpt_path = Path(pretrained_path)
        if not ckpt_path.exists():
            raise ValueError(
                "Không tìm thấy file trọng số tiền huấn luyện TransUNet tại đường dẫn cấu hình model.pretrained_path: "
                f"{ckpt_path}"
            )
        model.load_pretrained_from_npz(ckpt_path)
    elif encoder_weights is None:
        _log.warning("Mô hình TransUNet sẽ được huấn luyện từ đầu do model.encoder_weights=null.")
    else:
        raise ValueError(
            "TransUNet chỉ hỗ trợ model.encoder_weights là 'imagenet' hoặc null. "
            f"Giá trị nhận được: {m.encoder_weights!r}"
        )

    return model


# =============================================================================
# Đăng ký mô hình (Registry)
# =============================================================================

_REGISTRY = {
    "unet": _build_unet,
    "unet_original": _build_unet_original,
    "deeplabv3": _build_deeplabv3,
    "deeplabv3plus": _build_deeplabv3plus,
    "transunet": _build_transunet,
}


def create_model(config) -> nn.Module:
    """
    Hàm tạo mô hình dựa trên cấu hình đầu vào.

    Tham số đầu vào:
        config: Đối tượng cấu hình chứa trường config.model.name

    Kết quả trả về:
        Đối tượng nn.Module tương ứng với mô hình đã chọn.

    Ngoại lệ:
        ValueError: Nếu tên mô hình chưa được đăng ký trong danh sách hỗ trợ.

    Ví dụ sử dụng:
        model = create_model(config)                    # Chạy trên GPU đơn lẻ
        model = nn.DataParallel(model).to(device)       # Chạy trên môi trường đa GPU
    """
    name = config.model.name
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(f"Mô hình '{name}' chưa được đăng ký. Các mô hình hiện có: {available}")
    return _REGISTRY[name](config)
