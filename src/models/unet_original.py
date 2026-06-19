# =============================================================================
# Thư mục: src/models/unet_original.py
# Chức năng: Định nghĩa cấu trúc mạng U-Net nguyên bản (Original U-Net) từ đầu, không sử dụng backbone có sẵn.
# Lớp quan trọng:
#   - UNetOriginal: Lớp chính chứa luồng mã hóa (encoder), giải mã (decoder) và kết nối tắt (skip connections).
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn


class _DoubleConv(nn.Module):
    """Lớp thực hiện hai phép tích chập liên tiếp (Convolution -> ReLU -> Convolution -> ReLU)."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Down(nn.Module):
    """Khối giảm chiều dữ liệu (Downsampling) trong nhánh mã hóa: MaxPool2d -> _DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Up(nn.Module):
    """Khối tăng chiều dữ liệu (Upsampling) kết hợp kết nối tắt (skip connection) trong nhánh giải mã."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = _DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetOriginal(nn.Module):
    """
    Kiến trúc mạng U-Net gốc (Ronneberger et al., 2015) phục vụ phân đoạn nhị phân.

    Các tham số đầu vào:
        in_channels: Số lượng kênh dữ liệu đầu vào (mặc định RGB=3).
        num_classes: Số lượng kênh đầu ra (số lớp phân đoạn).
        base_channels: Số kênh cơ bản cho khối tích chập đầu tiên của encoder.

    Ngoại lệ:
        ValueError: Nếu các tham số in_channels, num_classes, hoặc base_channels nhỏ hơn hoặc bằng 0.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels phải > 0, giá trị nhận được: {in_channels}")
        if num_classes <= 0:
            raise ValueError(f"num_classes phải > 0, giá trị nhận được: {num_classes}")
        if base_channels <= 0:
            raise ValueError(f"base_channels phải > 0, giá trị nhận được: {base_channels}")

        c = base_channels
        self.inc = _DoubleConv(in_channels, c)
        self.down1 = _Down(c, c * 2)
        self.down2 = _Down(c * 2, c * 4)
        self.down3 = _Down(c * 4, c * 8)
        self.down4 = _Down(c * 8, c * 16)
        self.up1 = _Up(c * 16, c * 8)
        self.up2 = _Up(c * 8, c * 4)
        self.up3 = _Up(c * 4, c * 2)
        self.up4 = _Up(c * 2, c)
        self.outc = nn.Conv2d(c, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
