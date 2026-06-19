#!/usr/bin/env python3
# =============================================================================
# Thư mục: scripts/evaluate.py
# Chức năng: Đánh giá hiệu năng mô hình trên tập kiểm thử (Test Set), kết hợp kỹ thuật
#            TTA (Test-Time Augmentation) và tìm kiếm ngưỡng phân ngưỡng tối ưu.
# Hàm quan trọng:
#   - evaluate: Chạy vòng lặp đánh giá trên toàn bộ tập dữ liệu, tính toán hệ số Dice/IoU trên nhiều ngưỡng khác nhau.
#   - main: Đọc cấu hình, khởi tạo mô hình, nạp trọng số và điều phối quá trình kiểm thử.
# =============================================================================

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.dataset import ISICDataset
from src.data.transforms import get_transforms
from src.inference.tta import tta_predict
from src.losses.segmentation import CombinedLoss
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


def _resolve_eval_batch_size(config) -> int:
    """Tính toán batch size kiểm thử dựa trên batch size huấn luyện và hệ số nhân."""
    return config.training.batch_size * int(getattr(config.data, "val_batch_size_multiplier", 2))


# ---------------------------------------------------------------------------
# Vòng lặp đánh giá hiệu năng
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: CombinedLoss,
    split: str = "test",
    threshold: float = 0.5,
    use_tta: bool = True,
) -> dict:
    """Thực thi một vòng kiểm tra đầy đủ trên tập dữ liệu. Trả về từ điển kết quả."""
    model.eval()
    total_loss = 0.0
    thresholds = [round(t, 2) for t in torch.arange(0.3, 0.71, 0.05).tolist()]
    idx_report = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - threshold))
    report_thr = thresholds[idx_report]
    dice_sums = [0.0 for _ in thresholds]
    iou_sums = [0.0 for _ in thresholds]
    dice_values: list[float] = []
    iou_values: list[float] = []
    total_samples = 0

    for images, masks in tqdm(loader, desc="Evaluating"):
        images = images.to(device)
        masks = masks.to(device)

        if use_tta:
            probs = tta_predict(model, images)
            # Tính toán hao hụt dựa trên kết quả suy luận đơn lẻ để đảm bảo tính so sánh tương đồng.
            logits = model(images)
        else:
            logits = model(images)
            probs = torch.sigmoid(logits)

        loss = criterion(logits, masks)

        n = images.size(0)
        total_loss += loss.item() * n
        total_samples += n

        probs_flat = probs.view(n, -1)
        masks_flat = masks.float().view(n, -1)
        target_sum = masks_flat.sum(dim=1)

        # Tính toán các chỉ số thống kê cục bộ tại ngưỡng báo cáo
        pred_flat_report = (probs_flat > report_thr).float()
        intersection_report = (pred_flat_report * masks_flat).sum(dim=1)
        pred_sum_report = pred_flat_report.sum(dim=1)
        dice_report = (2.0 * intersection_report + 1e-7) / (pred_sum_report + target_sum + 1e-7)
        iou_report = (intersection_report + 1e-7) / (
            pred_sum_report + target_sum - intersection_report + 1e-7
        )
        dice_values.extend(dice_report.tolist())
        iou_values.extend(iou_report.tolist())

        # Đánh giá chỉ số trên nhiều ngưỡng khác nhau để tìm ra ngưỡng tối ưu
        for idx, thr in enumerate(thresholds):
            pred_flat = (probs_flat > thr).float()
            intersection = (pred_flat * masks_flat).sum(dim=1)
            pred_sum = pred_flat.sum(dim=1)

            dice = (2.0 * intersection + 1e-7) / (pred_sum + target_sum + 1e-7)
            iou = (intersection + 1e-7) / (pred_sum + target_sum - intersection + 1e-7)

            dice_sums[idx] += float(dice.sum().item())
            iou_sums[idx] += float(iou.sum().item())

    if total_samples == 0:
        raise RuntimeError("Bộ nạp dữ liệu rỗng, không thể chạy đánh giá.")

    def _summarize(values: list[float]) -> dict[str, float]:
        """Tóm tắt các chỉ số thống kê mô tả cơ bản."""
        if not values:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "median": 0.0}
        mean = float(statistics.fmean(values))
        std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
        return {
            "mean": mean,
            "std": std,
            "min": float(min(values)),
            "max": float(max(values)),
            "median": float(statistics.median(values)),
        }

    mean_dice = [s / total_samples for s in dice_sums]
    mean_iou = [s / total_samples for s in iou_sums]
    best_idx = max(range(len(thresholds)), key=lambda i: mean_dice[i])
    split_prefix = split.lower()

    tta_loss_note = (
        "hao hụt được tính dựa trên logits suy luận đơn lẻ để đảm bảo tính so sánh" if use_tta else "hao hụt được tính bình thường"
    )

    return {
        "loss": total_loss / total_samples,
        "dice": mean_dice[idx_report],
        "iou": mean_iou[idx_report],
        "best_threshold": thresholds[best_idx],
        "best_dice_at_best_thr": mean_dice[best_idx],
        "best_iou_at_best_thr": mean_iou[best_idx],
        f"{split_prefix}_dice": mean_dice[idx_report],
        f"{split_prefix}_iou": mean_iou[idx_report],
        f"{split_prefix}_dice_best": mean_dice[best_idx],
        f"{split_prefix}_iou_best": mean_iou[best_idx],
        "threshold_report": report_thr,
        "dice_stats": _summarize(dice_values),
        "iou_stats": _summarize(iou_values),
        "tta_loss_note": tta_loss_note,
    }


# ---------------------------------------------------------------------------
# Giao diện dòng lệnh CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Đọc tham số dòng lệnh CLI."""
    parser = argparse.ArgumentParser(
        description="Đánh giá mô hình phân đoạn trên tập dữ liệu kiểm thử",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", "-c", required=True, help="Đường dẫn đến file cấu hình YAML")
    parser.add_argument(
        "--checkpoint", "-k", required=True, help="Đường dẫn đến file checkpoint best_model.pth"
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Chọn tập dữ liệu để đánh giá (mặc định: test)",
    )
    parser.add_argument(
        "--tta",
        dest="tta",
        action="store_true",
        default=True,
        help="Kích hoạt kỹ thuật Test-Time Augmentation (mặc định: bật)",
    )
    parser.add_argument("--no-tta", dest="tta", action="store_false", help="Tắt kỹ thuật TTA")
    parser.add_argument("overrides", nargs="*", metavar="key.subkey=value")
    return parser.parse_args()


def main() -> None:
    """Luồng thực thi chính."""
    args = parse_args()

    config = load_config(args.config)
    config = override_config(config, args.overrides)
    if not config.logging.experiment_name:
        config["logging"]["experiment_name"] = Path(args.config).stem

    set_seed(config.seed, deterministic=bool(getattr(config.training, "deterministic", True)))
    device = get_device()
    log.info(f"Thiết bị: {device} | TTA: {args.tta} | Tập dữ liệu: {args.split}")
    if args.tta:
        log.info("Đã kích hoạt TTA; hao hụt (loss) sẽ được tính trên logits suy luận đơn để đồng bộ so sánh.")

    # Khởi tạo bộ nạp dữ liệu
    root = Path(config.data.root)
    dataset = ISICDataset(
        img_dir=root / args.split / "images",
        mask_dir=root / args.split / "masks",
        transform=get_transforms("val", config),  # Kiểm thử không dùng biến đổi tăng cường ảnh
    )
    eval_batch_size = _resolve_eval_batch_size(config)
    loader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    log.info(f"Tập [{args.split}]: gồm {len(dataset)} mẫu dữ liệu")

    # Khởi tạo mô hình
    model = create_model(config).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    load_state_dict_with_aux_compat(model, state, context=str(args.checkpoint))
    log.info(f"Đã nạp thành công file trọng số: {args.checkpoint}")

    # Khởi tạo hàm mất mát
    criterion = CombinedLoss(config)

    # Đánh giá hiệu năng
    results = evaluate(model, loader, device, criterion, split=args.split, use_tta=args.tta)

    # In kết quả cuối cùng
    print("\n" + "=" * 60)
    print(f"  KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (TẬP {args.split.upper()})")
    print("=" * 60)
    print(f"  Hao hụt (Loss):          {results['loss']:.4f}")
    print(
        f"  Chỉ số {args.split}_dice tại ngưỡng={results['threshold_report']:.2f}:  "
        f"{results[f'{args.split}_dice']:.4f}"
    )
    print(
        f"  Chỉ số {args.split}_iou  tại ngưỡng={results['threshold_report']:.2f}:  "
        f"{results[f'{args.split}_iou']:.4f}"
    )
    print(f"  Ngưỡng tối ưu tìm thấy:  {results['best_threshold']:.2f}")
    print(f"  Chỉ số {args.split}_dice_best (tốt nhất):       {results[f'{args.split}_dice_best']:.4f}")
    print(f"  Chỉ số {args.split}_iou_best tại ngưỡng tối ưu: {results[f'{args.split}_iou_best']:.4f}")
    print(f"  Sử dụng kỹ thuật TTA:    {args.tta}")
    print("=" * 60)

    if args.split == "test":
        dice_stats = results["dice_stats"]
        iou_stats = results["iou_stats"]
        print("  Thống kê chi tiết hệ số Dice (tính trên từng mẫu tại ngưỡng báo cáo):")
        print(
            f"    trung_bình={dice_stats['mean']:.4f} độ_lệch_chuẩn={dice_stats['std']:.4f} "
            f"nhỏ_nhất={dice_stats['min']:.4f} lớn_nhất={dice_stats['max']:.4f} "
            f"trung_vị={dice_stats['median']:.4f}"
        )
        print("  Thống kê chi tiết chỉ số IoU (tính trên từng mẫu tại ngưỡng báo cáo):")
        print(
            f"    trung_bình={iou_stats['mean']:.4f} độ_lệch_chuẩn={iou_stats['std']:.4f} "
            f"nhỏ_nhất={iou_stats['min']:.4f} lớn_nhất={iou_stats['max']:.4f} "
            f"trung_vị={iou_stats['median']:.4f}"
        )
        print("=" * 60)

    # Lưu kết quả định dạng JSON cùng thư mục với checkpoint
    ckpt_path = Path(args.checkpoint)
    out_path = ckpt_path.parent / f"eval_{args.split}_results.json"
    with open(out_path, "w") as f:
        json.dump({"split": args.split, "tta": args.tta, **results}, f, indent=2)
    log.info(f"Đã lưu kết quả chi tiết tại: {out_path}")

    if args.split == "test":
        stats_path = ckpt_path.parent / f"eval_{args.split}_metric_stats.json"
        with open(stats_path, "w") as f:
            json.dump(
                {
                    "split": args.split,
                    "tta": args.tta,
                    "threshold_report": results["threshold_report"],
                    "dice_stats": results["dice_stats"],
                    "iou_stats": results["iou_stats"],
                },
                f,
                indent=2,
            )
        log.info(f"Đã lưu thống kê chỉ số tại: {stats_path}")


if __name__ == "__main__":
    main()
