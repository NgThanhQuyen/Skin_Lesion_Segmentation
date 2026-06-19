#!/usr/bin/env python3
# =============================================================================
# Thư mục: scripts/predict.py
# Chức năng: Chạy suy luận (inference) trên các hình ảnh mới để dự đoán mặt nạ phân đoạn
#            vùng tổn thương da, đồng thời vẽ trực quan hóa kết quả phủ ảnh (overlay).
# Hàm quan trọng:
#   - preprocess: Đọc và tiền xử lý ảnh đơn về tensor kích thước (1, C, H, W).
#   - predict_single: Thực hiện suy luận và phân ngưỡng nhị phân cho một ảnh đơn.
#   - save_overlay: Tạo biểu đồ so sánh: ảnh gốc, mặt nạ dự đoán và ảnh phủ đặc trưng.
# =============================================================================

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset import IMAGE_EXTS
from src.data.transforms import get_transforms
from src.inference.tta import tta_predict
from src.models.segmentation import create_model
from src.utils.checkpoint import load_state_dict_with_aux_compat
from src.utils.config import load_config, override_config
from src.utils.misc import get_device, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Các hàm bổ trợ suy luận
# ---------------------------------------------------------------------------


def preprocess(image_path: Path, transform) -> torch.Tensor:
    """Tải và áp dụng các tiền xử lý cho ảnh đơn -> trả về tensor (1, C, H, W)."""
    image = np.array(Image.open(image_path).convert("RGB"))
    dummy_mask = np.zeros(image.shape[:2], dtype=np.float32)
    out = transform(image=image, mask=dummy_mask)
    tensor = out["image"]  # (C, H, W)
    return tensor.unsqueeze(0)  # (1, C, H, W)


@torch.no_grad()
def predict_single(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
    use_tta: bool = False,
) -> np.ndarray:
    """
    Dự đoán mặt nạ phân đoạn cho một ảnh đơn đầu vào.

    Kết quả trả về:
        Mặt nạ nhị phân định dạng uint8 numpy array (H, W) với các giá trị 0/255.
    """
    x = image_tensor.to(device)

    if use_tta:
        probs = tta_predict(model, x)
    else:
        probs = torch.sigmoid(model(x))

    mask = (probs[0, 0] > threshold).cpu().numpy().astype(np.uint8) * 255
    return mask


def save_overlay(
    image_path: Path,
    mask: np.ndarray,
    save_path: Path,
    alpha: float = 0.4,
) -> None:
    """Vẽ và lưu trữ biểu đồ so sánh: ảnh gốc | mặt nạ dự đoán | ảnh phủ đặc trưng."""
    import matplotlib.pyplot as plt

    orig = np.array(Image.open(image_path).convert("RGB"))
    # Mặt nạ được sinh ra ở kích thước đầu vào của mô hình; cần resize lại về kích thước ảnh gốc
    if mask.shape != orig.shape[:2]:
        orig_h, orig_w = orig.shape[:2]
        mask = np.array(
            Image.fromarray(mask).resize((orig_w, orig_h), resample=Image.Resampling.NEAREST)
        )

    mask_rgb = np.zeros_like(orig)
    mask_rgb[:, :, 0] = mask  # Gán mặt nạ màu đỏ vào kênh màu Red

    h_mask = mask > 127
    overlay = orig.copy().astype(float)
    overlay[h_mask] = (1 - alpha) * orig[h_mask] + alpha * mask_rgb[h_mask]
    overlay = overlay.astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(orig)
    axes[0].set_title("Ảnh gốc")
    axes[0].axis("off")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Mặt nạ dự đoán")
    axes[1].axis("off")
    axes[2].imshow(overlay)
    axes[2].set_title("Ảnh phủ đặc trưng (Overlay)")
    axes[2].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Giao diện dòng lệnh CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh CLI."""
    parser = argparse.ArgumentParser(
        description="Chạy suy luận trên các hình ảnh mới",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", "-c", required=True, help="Đường dẫn đến file cấu hình YAML")
    parser.add_argument("--checkpoint", "-k", required=True, help="Đường dẫn đến file best_model.pth")
    parser.add_argument(
        "--input", "-i", required=True, help="Đường dẫn đến file ảnh đầu vào hoặc thư mục chứa ảnh"
    )
    parser.add_argument(
        "--output", "-o", default="outputs/predictions", help="Thư mục đầu ra lưu trữ mặt nạ dự đoán"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.5, help="Ngưỡng nhị phân hóa (mặc định: 0.5)"
    )
    parser.add_argument(
        "--tta", dest="tta", action="store_true", default=False, help="Kích hoạt kỹ thuật TTA (mặc định: tắt)"
    )
    parser.add_argument(
        "--overlay", action="store_true", default=False, help="Lưu thêm biểu đồ trực quan hóa overlay dạng PNG"
    )
    parser.add_argument("overrides", nargs="*", metavar="key.subkey=value")
    return parser.parse_args()


def main() -> None:
    """Luồng thực thi chính."""
    args = parse_args()

    config = load_config(args.config)
    config = override_config(config, args.overrides)

    set_seed(config.seed, deterministic=bool(getattr(config.training, "deterministic", True)))
    device = get_device()

    # Thu thập danh sách ảnh đầu vào
    input_path = Path(args.input)
    if input_path.is_file():
        image_paths = [input_path]
    elif input_path.is_dir():
        image_paths: list[Path] = []
        for ext in sorted(IMAGE_EXTS):
            image_paths.extend(sorted(input_path.glob(f"*{ext}")))
    else:
        log.error(f"Không tìm thấy đường dẫn đầu vào: {input_path}")
        sys.exit(1)

    if not image_paths:
        exts = ", ".join(sorted(IMAGE_EXTS))
        log.error(f"Không tìm thấy ảnh hợp lệ nào tại: {input_path}. Định dạng hỗ trợ: {exts}")
        sys.exit(1)

    log.info(f"Tìm thấy {len(image_paths)} ảnh | threshold={args.threshold} | TTA={args.tta}")

    # Khởi tạo mô hình
    model = create_model(config).to(device)
    model.eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    load_state_dict_with_aux_compat(model, state, context=str(args.checkpoint))
    log.info(f"Đã nạp thành công checkpoint: {args.checkpoint}")

    # Chuẩn bị thư mục đầu ra
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    transform = get_transforms("val", config)  # Chỉ sử dụng resize và chuẩn hóa, không tăng cường ảnh

    # Tiến hành dự đoán
    for img_path in tqdm(image_paths, desc="Predicting"):
        tensor = preprocess(img_path, transform)
        mask = predict_single(model, tensor, device, args.threshold, args.tta)

        # Lưu mặt nạ dự đoán
        mask_save = out_dir / f"{img_path.stem}_pred_mask.png"
        Image.fromarray(mask).save(mask_save)

        # Lưu biểu đồ trực quan hóa overlay nếu được cấu hình
        if args.overlay:
            overlay_save = out_dir / f"{img_path.stem}_overlay.png"
            save_overlay(img_path, mask, overlay_save)

    log.info(f"Hoàn tất. Kết quả dự đoán đã được lưu tại: {out_dir}")


if __name__ == "__main__":
    main()
