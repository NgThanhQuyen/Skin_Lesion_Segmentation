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
        self.use_dagshub = getattr(config.logging, "use_dagshub", False)
        self.upload_model = getattr(config.logging, "upload_model", False)
        self.history: list[dict] = []
        self._wandb_run = None
        self._mlflow_run = None

        self._init_wandb(config)
        self._init_dagshub(config)

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

    def _init_dagshub(self, config: Any) -> None:
        """Khởi tạo kết nối với hệ thống giám sát DagsHub MLflow."""
        if not self.use_dagshub:
            return

        try:
            import dagshub
            import mlflow
        except ImportError:
            logger.warning(
                "Thư viện dagshub hoặc mlflow chưa được cài đặt. "
                "Hệ thống chuyển về chế độ ghi nhật ký cục bộ cho phần DagsHub."
            )
            self.use_dagshub = False
            return

        username = getattr(config.logging, "dagshub_username", None)
        repo = getattr(config.logging, "dagshub_repo", None)

        if not username or not repo:
            logger.warning(
                "Thiếu thông tin cấu hình dagshub_username hoặc dagshub_repo. "
                "Bỏ qua ghi nhận nhật ký DagsHub."
            )
            self.use_dagshub = False
            return

        try:
            # Khởi tạo tích hợp DagsHub với MLflow
            dagshub.init(repo_owner=username, repo_name=repo, mlflow=True)
            
            # Đặt tên Experiment
            mlflow.set_experiment(config.logging.project)
            
            # Bắt đầu run mới
            experiment_name = config.logging.experiment_name or "default_experiment"
            self._mlflow_run = mlflow.start_run(run_name=experiment_name)
            
            # Ghi các thông số cấu hình (params)
            flat_config = self._flatten_dict(config.to_dict())
            mlflow.log_params(flat_config)
            
            logger.info("Đã liên kết thành công với DagsHub MLflow!")
        except Exception as e:
            logger.warning(f"Khởi tạo DagsHub MLflow thất bại: {e}. Hệ thống chuyển về chế độ ghi nhật ký cục bộ cho phần DagsHub.")
            self.use_dagshub = False

    def _flatten_dict(self, d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """Làm phẳng từ điển lồng nhau để ghi nhận cấu hình lên MLflow."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

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

        if self.use_dagshub and self._mlflow_run:
            import mlflow
            # Loại bỏ trường 'step' khỏi metrics trước khi log vào MLflow (vì MLflow truyền step qua đối số)
            mlflow_metrics = {k: v for k, v in metrics.items() if k != "step"}
            mlflow.log_metrics(mlflow_metrics, step=step)

    def log_summary(self, summary: dict) -> None:
        """Ghi nhận chỉ số tóm tắt cuối cùng (kết quả test, chỉ số tốt nhất...)."""
        if self.use_wandb and self._wandb_run:
            for k, v in summary.items():
                self._wandb_run.summary[k] = v

        if self.use_dagshub and self._mlflow_run:
            import mlflow
            for k, v in summary.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"best_{k}", v)

    def log_image(self, key: str, image_path: str | Path) -> None:
        """Gửi hình ảnh kết quả lên giao diện W&B."""
        if self.use_wandb and self._wandb_run:
            import wandb

            self._wandb_run.log({key: wandb.Image(str(image_path))})

        if self.use_dagshub and self._mlflow_run:
            import mlflow
            try:
                from PIL import Image
                img = Image.open(image_path)
                mlflow.log_image(img, f"{key}.png")
            except Exception:
                mlflow.log_artifact(str(image_path), artifact_path="images")

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
        """Hoàn tất quá trình ghi nhật ký và đóng liên kết với W&B/DagsHub."""
        self.save_history()
        if self.use_wandb and self._wandb_run:
            self._wandb_run.finish()

        if self.use_dagshub and self._mlflow_run:
            import mlflow
            try:
                # Tải các tệp tin kết quả quan trọng lên DagsHub dưới dạng artifact
                artifacts = [
                    "training_curves.png",
                    "training_summary.json",
                    "training_history.json",
                    "training_history.csv",
                    "metrics_summary.json",
                    "metrics_summary.csv"
                ]
                if self.upload_model:
                    artifacts.append("best_model.pth")
                else:
                    logger.info("Cấu hình upload_model=False. Bỏ qua tải tệp trọng số best_model.pth lên DagsHub.")

                for art_name in artifacts:
                    art_path = self.output_dir / art_name
                    if art_path.exists():
                        mlflow.log_artifact(str(art_path))
                        logger.info(f"Đã upload thành công artifact: {art_name} lên DagsHub.")
            except Exception as e:
                logger.warning(f"Tải artifacts lên DagsHub thất bại: {e}")
            finally:
                mlflow.end_run()
