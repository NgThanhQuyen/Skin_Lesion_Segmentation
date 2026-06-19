# =============================================================================
# Thư mục: src/utils/logger.py
# Chức năng: Điều phối ghi nhật ký đồng thời lên hệ thống đám mây Weights & Biases (W&B)
#            và lưu trữ các tệp tin kết quả cục bộ (JSON/CSV).
# Lớp quan trọng:
#   - Logger: Lớp chính tích hợp các tính năng ghi nhật ký và tính toán thống kê kết quả.
# =============================================================================

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Logger:
    """
    Trình ghi nhật ký hợp nhất: Tự động ghi nhận thông tin lên W&B và xuất tệp tin cục bộ (JSON/CSV).

    Cách sử dụng:
        log = Logger(config, output_dir)
        log.log({"train_loss": 0.12, "val_dice": 0.89}, step=1)
        log.log_summary({"test_dice": 0.90})
        log.finish()
    """

    def __init__(self, config: Any, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.use_wandb = config.logging.use_wandb
        self.history: list[dict] = []
        self._wandb_run = None

        self._init_wandb(config)

    # ------------------------------------------------------------------
    # Thiết lập khởi tạo
    # ------------------------------------------------------------------

    def _init_wandb(self, config: Any) -> None:
        """Khởi tạo kết nối với hệ thống giám sát Weights & Biases."""
        if not self.use_wandb:
            return

        try:
            import wandb
        except ImportError:
            logger.warning("Thư viện wandb chưa được cài đặt. Hệ thống chuyển về chế độ ghi nhật ký cục bộ.")
            self.use_wandb = False
            return

        experiment_name = config.logging.experiment_name
        entity = config.logging.entity if config.logging.entity else None

        try:
            self._wandb_run = wandb.init(
                project=config.logging.project,
                entity=entity,
                name=experiment_name,
                config=config.to_dict(),
                dir=str(self.output_dir),
            )
            logger.info(f"Đã liên kết thành công với W&B: {self._wandb_run.url}")
        except Exception as e:
            logger.warning(f"Khởi tạo W&B thất bại: {e}. Hệ thống chuyển về chế độ ghi nhật ký cục bộ.")
            self.use_wandb = False

    # ------------------------------------------------------------------
    # Lưu nhật ký
    # ------------------------------------------------------------------

    def log(self, metrics: dict, step: int | None = None) -> None:
        """Ghi các chỉ số đánh giá cho từng epoch hoặc bước lặp."""
        if step is not None:
            metrics = {"step": step, **metrics}
        self.history.append(metrics)

        if self.use_wandb and self._wandb_run:
            self._wandb_run.log(metrics, step=step)

    def log_summary(self, summary: dict) -> None:
        """Ghi nhận chỉ số tóm tắt cuối cùng (kết quả test, chỉ số tốt nhất...)."""
        if self.use_wandb and self._wandb_run:
            for k, v in summary.items():
                self._wandb_run.summary[k] = v

    def log_image(self, key: str, image_path: str | Path) -> None:
        """Gửi hình ảnh kết quả lên giao diện W&B."""
        if self.use_wandb and self._wandb_run:
            import wandb

            self._wandb_run.log({key: wandb.Image(str(image_path))})

    # ------------------------------------------------------------------
    # Lưu trữ cục bộ
    # ------------------------------------------------------------------

    def save_history(self) -> None:
        """Xuất toàn bộ lịch sử huấn luyện ra định dạng tệp CSV và JSON cục bộ."""
        if not self.history:
            return

        import csv

        # Lưu ra tệp JSON
        json_path = self.output_dir / "training_history.json"
        with open(json_path, "w") as f:
            json.dump(self.history, f, indent=2)

        # Lưu ra tệp CSV
        csv_path = self.output_dir / "training_history.csv"
        fieldnames: list[str] = []
        for row in self.history:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.history)

        # Tính toán và xuất thống kê tóm tắt các chỉ số (mean/std/min/max/median)
        metrics_summary = self._build_metrics_summary(self.history)
        if metrics_summary:
            summary_json_path = self.output_dir / "metrics_summary.json"
            with open(summary_json_path, "w") as f:
                json.dump(metrics_summary, f, indent=2)

            summary_csv_path = self.output_dir / "metrics_summary.csv"
            with open(summary_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["metric", "mean", "std", "min", "max", "median"]
                )
                writer.writeheader()
                writer.writerows(metrics_summary)

    def _build_metrics_summary(self, history: list[dict]) -> list[dict[str, float | str]]:
        """
        Tính toán các giá trị Trung bình (Mean), Độ lệch chuẩn (Std), Nhỏ nhất (Min), Lớn nhất (Max)
        và Trung vị (Median) cho mỗi chỉ số ghi nhận trong lịch sử.

        Chỉ áp dụng với các khóa có định dạng số và loại bỏ khóa chỉ mục "step".
        """
        metric_values: dict[str, list[float]] = {}
        for row in history:
            for key, value in row.items():
                if key == "step":
                    continue
                if isinstance(value, (int, float)):
                    metric_values.setdefault(key, []).append(float(value))

        summary: list[dict[str, float | str]] = []
        for metric, values in metric_values.items():
            if not values:
                continue
            mean = float(statistics.fmean(values))
            std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
            summary.append(
                {
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "min": float(min(values)),
                    "max": float(max(values)),
                    "median": float(statistics.median(values)),
                }
            )
        return summary

    # ------------------------------------------------------------------
    # Kết thúc ghi nhật ký
    # ------------------------------------------------------------------

    def finish(self) -> None:
        """Hoàn tất quá trình ghi nhật ký và đóng liên kết với W&B."""
        self.save_history()
        if self.use_wandb and self._wandb_run:
            self._wandb_run.finish()
