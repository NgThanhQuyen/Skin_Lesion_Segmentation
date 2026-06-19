# =============================================================================
# Thư mục: src/training/trainer.py
# Chức năng: Quản lý toàn bộ vòng lặp huấn luyện (training loop), kiểm định (validation),
#            cơ chế dừng sớm (Early Stopping), lưu checkpoint và cập nhật tỷ lệ học (Learning Rate).
# Lớp quan trọng:
#   - Trainer: Lớp chính chứa các luồng điều phối huấn luyện.
# =============================================================================

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.losses import CombinedLoss
from src.metrics import dice_coefficient, iou_score
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.checkpoint import load_state_dict_with_aux_compat
from src.utils.logger import Logger
from src.utils.misc import plot_training_curves

logger = logging.getLogger(__name__)
_VAL_METRIC_SEMANTICS = "macro_per_sample_v1"


def _compute_warmup_lr(
    epoch: int,
    warmup_epochs: int,
) -> float:
    """
    Tính toán hệ số tiến trình khởi động tuyến tính (linear warmup progress factor) trong khoảng [0, 1].

    Hệ số tiến trình khởi động sử dụng công thức (epoch + 1) / warmup_epochs để đảm bảo rằng
    ở epoch khởi động cuối cùng, tỷ lệ học sẽ đạt chính xác giá trị mục tiêu.
    """
    if warmup_epochs <= 0:
        return 1.0
    return min((epoch + 1) / warmup_epochs, 1.0)


class Trainer:
    """
    Trình quản lý huấn luyện (Trainer) cho bài toán phân đoạn nhị phân.

    Các tham số đầu vào:
        model:           Đối tượng nn.Module (đã được chuyển lên thiết bị GPU/CPU tương ứng).
        config:          Đối tượng cấu hình dự án.
        device:          Thiết bị phần cứng để huấn luyện (torch.device).
        log:             Đối tượng Logger ghi nhật ký (Weights & Biases và ghi nhận cục bộ).
        is_distributed:  Cấu hình huấn luyện song song phân tán (DDP) có bật hay không.
        is_main_process: Đánh dấu nếu đây là tiến trình chính (rank 0).

    Cách sử dụng:
        trainer = Trainer(model, config, device, log)
        trainer.fit(train_loader, val_loader, output_dir)
    """

    def __init__(
        self,
        model: nn.Module,
        config: Any,
        device: torch.device,
        log: Logger,
        is_distributed: bool = False,
        is_main_process: bool = True,
    ):
        self.model = model
        self.config = config
        self.device = device
        self.log = log
        self.is_distributed = is_distributed
        self.is_main_process = is_main_process
        self._resume_state: dict[str, Any] = {}

        # Thiết lập hàm mất mát
        self.criterion = CombinedLoss(config)

        # Thiết lập bộ tối ưu hóa (Optimizer)
        self.optimizer = self._build_optimizer()
        self._init_warmup_param_groups(force=True)

        # Thiết lập bộ điều chỉnh tỷ lệ học (LR Scheduler)
        self.scheduler = self._build_scheduler()

        # Bộ phóng đại gradient cho huấn luyện độ chính xác hỗn hợp (AMP)
        self.scaler = GradScaler(device.type, enabled=config.training.mixed_precision)

        # Ngưỡng cắt giảm gradient
        self.grad_clip = config.training.grad_clip

    # ------------------------------------------------------------------
    # Các hàm thiết lập bổ trợ
    # ------------------------------------------------------------------

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Xây dựng bộ tối ưu hóa dựa trên tên cấu hình."""
        cfg = self.config.training
        name = cfg.optimizer.lower()

        params = self._get_param_groups(cfg)

        if name == "adamw":
            return torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if name == "adam":
            return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        if name == "sgd":
            return torch.optim.SGD(
                params,
                lr=cfg.lr,
                weight_decay=cfg.weight_decay,
                momentum=0.9,
            )
        raise ValueError(f"Bộ tối ưu hóa '{name}' chưa được hỗ trợ.")

    def _get_param_groups(self, cfg: Any) -> Any:
        """
        Xây dựng các nhóm tham số (parameter groups) cho bộ tối ưu hóa.

        Khi cấu hình ``training.differential_lr.enabled`` bằng True, các tham số sẽ được tách ra
        làm hai nhóm: encoder (tỷ lệ học thấp hơn) và decoder (tỷ lệ học đầy đủ). Trong trường hợp
        kiến trúc mô hình không được nhận diện, luồng xử lý sẽ quay về cấu hình tỷ lệ học đơn nhóm mặc định.

        Các mô hình được hỗ trợ:
        - Các mô hình SMP: Cung cấp thuộc tính ``.encoder``
        - Lớp DeepLabV3Wrapper (torchvision): Cung cấp thuộc tính ``.model.backbone``
        - Các mô hình song song DDP: Cần giải phóng thuộc tính ``.module`` trước

        Các tham số đầu vào:
            cfg: Nhánh cấu hình ``config.training``.

        Kết quả trả về:
            Hoặc là bộ tham số của mô hình (đối với tỷ lệ học đơn nhóm) hoặc danh sách từ điển
            chứa thông tin của từng nhóm tham số kèm tỷ lệ học tương ứng (đối với tỷ lệ học phân biệt).
        """
        diff_lr_cfg = getattr(cfg, "differential_lr", None)
        if diff_lr_cfg is None or not getattr(diff_lr_cfg, "enabled", False):
            return self.model.parameters()

        # Giải phóng lớp DDP nếu có
        model_ref = self.model.module if hasattr(self.model, "module") else self.model

        encoder: torch.nn.Module | None = None

        # Đối với các mô hình của thư viện SMP (UNet, DeepLabV3+...)
        if hasattr(model_ref, "encoder"):
            encoder = model_ref.encoder
        # Đối với lớp DeepLabV3Wrapper của torchvision (sử dụng backbone làm encoder)
        elif hasattr(model_ref, "model") and hasattr(model_ref.model, "backbone"):
            encoder = model_ref.model.backbone

        if encoder is None:
            logger.warning(
                "Cấu hình differential_lr.enabled=true nhưng mô hình không cung cấp thuộc tính .encoder "
                "hoặc .model.backbone - quay về chế độ huấn luyện với tỷ lệ học đơn nhóm mặc định."
            )
            return self.model.parameters()

        encoder_lr = cfg.lr * float(getattr(diff_lr_cfg, "encoder_lr_scale", 0.1))
        encoder_ids = {id(p) for p in encoder.parameters()}

        decoder_params = [p for p in model_ref.parameters() if id(p) not in encoder_ids]
        encoder_params = list(encoder.parameters())

        logger.info(
            "Tỷ lệ học phân biệt (Differential LR): tỷ lệ học của encoder=%.2e, tỷ lệ học của decoder=%.2e",
            encoder_lr,
            cfg.lr,
        )
        return [
            {"params": encoder_params, "lr": encoder_lr},
            {"params": decoder_params, "lr": cfg.lr},
        ]

    def _build_scheduler(self) -> Any:
        """Xây dựng bộ điều chỉnh tỷ lệ học dựa trên cấu hình."""
        cfg = self.config.lr_schedule
        name = cfg.scheduler.lower()
        if name == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=cfg.mode,
                factor=cfg.factor,
                patience=cfg.patience,
                min_lr=cfg.min_lr,
            )
        if name == "cosine":
            warmup_epochs = int(getattr(cfg, "warmup_epochs", 0) or 0)
            cosine_epochs = max(1, int(self.config.training.max_epochs) - warmup_epochs)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cosine_epochs,
                eta_min=cfg.min_lr,
            )
        raise ValueError(f"Bộ điều chỉnh tỷ lệ học '{name}' chưa được hỗ trợ.")

    def _get_default_base_lrs(self) -> list[float]:
        """Lấy giá trị tỷ lệ học cơ bản cho từng nhóm tham số từ cấu hình."""
        main_lr = float(self.config.training.lr)
        group_count = len(self.optimizer.param_groups)
        if group_count == 1:
            return [main_lr]

        diff_lr_cfg = getattr(self.config.training, "differential_lr", None)
        if group_count == 2 and diff_lr_cfg is not None and getattr(diff_lr_cfg, "enabled", False):
            encoder_lr = main_lr * float(getattr(diff_lr_cfg, "encoder_lr_scale", 0.1))
            return [encoder_lr, main_lr]

        return [float(pg.get("lr", main_lr)) for pg in self.optimizer.param_groups]

    def _init_warmup_param_groups(self, force: bool = False) -> None:
        """Khởi tạo thông tin siêu dữ liệu khởi động (warmup) cho các nhóm tham số tối ưu hóa."""
        main_lr = float(self.config.training.lr)
        if main_lr <= 0:
            raise ValueError("Tham số training.lr phải lớn hơn 0 để sử dụng cơ chế khởi động.")

        warmup_start_lr = float(getattr(self.config.lr_schedule, "warmup_start_lr", 1e-6))
        base_lrs = self._get_default_base_lrs()
        for pg, base_lr in zip(self.optimizer.param_groups, base_lrs, strict=False):
            if force or "warmup_base_lr" not in pg:
                pg["warmup_base_lr"] = float(base_lr)
            if force or "warmup_start_lr" not in pg:
                pg["warmup_start_lr"] = warmup_start_lr * (float(pg["warmup_base_lr"]) / main_lr)

    def _sync_epoch_totals(
        self,
        total_loss: float,
        total_dice: float,
        total_iou: float,
        n_samples: int,
    ) -> tuple[float, float, float, int]:
        """
        Đồng bộ hóa các giá trị tích lũy giữa các tiến trình phân tán (DDP).
        """
        stats = torch.tensor(
            [total_loss, total_dice, total_iou, float(n_samples)],
            device=self.device,
            dtype=torch.float64,
        )
        if self.is_distributed:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return (
            float(stats[0].item()),
            float(stats[1].item()),
            float(stats[2].item()),
            int(stats[3].item()),
        )

    # ------------------------------------------------------------------
    # Quy trình huấn luyện chính
    # ------------------------------------------------------------------

    def train_one_epoch(self, loader: DataLoader) -> dict:
        """Huấn luyện mô hình trong vòng 1 epoch. Trả về từ điển các chỉ số."""
        self.model.train()
        total_loss = total_dice = total_iou = 0.0
        n_samples = 0

        pbar = tqdm(loader, desc="Train", leave=False, disable=not self.is_main_process)
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.config.training.mixed_precision,
            ):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            self.scaler.scale(loss).backward()

            if self.grad_clip:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Tính toán các chỉ số đánh giá (sử dụng detach để không tích lũy đồ thị đạo hàm)
            with torch.no_grad():
                d = dice_coefficient(logits, masks)
                i = iou_score(logits, masks)

            n = images.size(0)
            total_loss += loss.item() * n
            total_dice += d * n
            total_iou += i * n
            n_samples += n

            if self.is_main_process:
                pbar.set_postfix(loss=f"{loss.item():.4f}", dice=f"{d:.4f}")

        total_loss, total_dice, total_iou, n_samples = self._sync_epoch_totals(
            total_loss, total_dice, total_iou, n_samples
        )
        if n_samples == 0:
            raise RuntimeError("Bộ nạp dữ liệu huấn luyện (Train loader) không có mẫu nào.")
        return {
            "train_loss": total_loss / n_samples,
            "train_dice": total_dice / n_samples,
            "train_iou": total_iou / n_samples,
        }

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> dict:
        """
        Kiểm thử mô hình trên tập kiểm định. Trả về từ điển các chỉ số.

        Hệ số Dice và IoU được tính theo dạng macro-average trên từng mẫu dữ liệu để đảm bảo tính
        nhất quán với các chỉ số huấn luyện và kiểm thử cuối cùng.
        """
        self.model.eval()
        total_loss = total_dice = total_iou = 0.0
        n_samples = 0

        pbar = tqdm(loader, desc="Val  ", leave=False, disable=not self.is_main_process)
        for images, masks in pbar:
            images = images.to(self.device)
            masks = masks.to(self.device)

            with torch.amp.autocast(
                device_type=self.device.type,
                enabled=self.config.training.mixed_precision,
            ):
                logits = self.model(images)
                loss = self.criterion(logits, masks)

            n = images.size(0)
            total_loss += loss.item() * n
            d = dice_coefficient(logits, masks)
            i = iou_score(logits, masks)
            total_dice += d * n
            total_iou += i * n
            n_samples += n

            if self.is_main_process:
                pbar.set_postfix(dice=f"{d:.4f}")

        total_loss, total_dice, total_iou, n_samples = self._sync_epoch_totals(
            total_loss, total_dice, total_iou, n_samples
        )

        if n_samples == 0:
            raise RuntimeError("Bộ nạp dữ liệu kiểm định (Validation loader) không có mẫu nào.")
        return {
            "val_loss": total_loss / n_samples,
            "val_dice": total_dice / n_samples,
            "val_iou": total_iou / n_samples,
        }

    # ------------------------------------------------------------------
    # Vòng lặp tối ưu hóa mô hình chính
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        output_dir: Path | str,
        start_epoch: int = 0,
        train_sampler: Any | None = None,
    ) -> dict:
        """
        Vòng lặp huấn luyện chính.

        Các tham số đầu vào:
            train_loader:  Bộ nạp dữ liệu cho tập huấn luyện.
            val_loader:    Bộ nạp dữ liệu cho tập kiểm định.
            output_dir:    Đường dẫn lưu file checkpoint tốt nhất và các file kết quả.
            start_epoch:   Chỉ mục epoch bắt đầu huấn luyện (0-indexed, mặc định bằng 0).
            train_sampler: Bộ lấy mẫu (ví dụ DistributedSampler) để thiết lập epoch cập nhật cho phân tán.

        Kết quả trả về:
            Từ điển chứa thông tin tóm tắt kết quả huấn luyện tốt nhất.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg_es = self.config.early_stopping
        monitor = cfg_es.monitor  # mặc định: "val_dice"

        checkpoint = ModelCheckpoint(
            save_path=output_dir / "best_model.pth",
            mode=cfg_es.mode,
            monitor=monitor,
        )
        early_stop = EarlyStopping(
            patience=cfg_es.patience,
            min_delta=cfg_es.min_delta,
            mode=cfg_es.mode,
            monitor=monitor,
        )

        best_metrics: dict[str, Any] = {}
        if start_epoch > 0:
            prev_best, prev_best_epoch, prev_best_metrics = self._load_previous_best_state(
                output_dir, monitor
            )
            checkpoint.best = prev_best
            checkpoint.best_epoch = prev_best_epoch
            best_metrics = prev_best_metrics
            if self.is_main_process and prev_best is not None:
                epoch_text = (
                    f"epoch {prev_best_epoch + 1}"
                    if prev_best_epoch is not None
                    else "không rõ epoch"
                )
                print(f"  Đã khôi phục trạng thái tốt nhất trước đó {monitor}={prev_best:.4f} ({epoch_text})")

        lr_cfg = self.config.lr_schedule
        sched_monitor = lr_cfg.monitor
        warmup_epochs = int(getattr(lr_cfg, "warmup_epochs", 0) or 0)
        warmup_start_lr = float(getattr(lr_cfg, "warmup_start_lr", 1e-6))
        target_lr = float(self.config.training.lr)

        if self.is_main_process:
            print("=" * 70)
            if start_epoch > 0:
                print(f"  QUÁ TRÌNH HUẤN LUYỆN ĐƯỢC TIẾP TỤC từ epoch {start_epoch + 1}")
            else:
                print("  QUÁ TRÌNH HUẤN LUYỆN BẮT ĐẦU")
            if warmup_epochs > 0:
                print(f"  Giai đoạn khởi động (Warmup): {warmup_epochs} epoch ({warmup_start_lr:.1e} -> {target_lr:.1e})")
            print("=" * 70)

        last_epoch = start_epoch - 1
        for epoch in range(start_epoch, self.config.training.max_epochs):
            last_epoch = epoch
            if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            # --- Điều chỉnh tỷ lệ học trong giai đoạn khởi động (Warmup) ---
            if epoch < warmup_epochs:
                warmup_factor = _compute_warmup_lr(
                    epoch=epoch,
                    warmup_epochs=warmup_epochs,
                )
                for pg in self.optimizer.param_groups:
                    start_lr = float(pg["warmup_start_lr"])
                    base_lr = float(pg["warmup_base_lr"])
                    pg["lr"] = start_lr + (base_lr - start_lr) * warmup_factor

            # --- Huấn luyện và Đánh giá ---
            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            lr_groups = [float(pg["lr"]) for pg in self.optimizer.param_groups]
            lr_main = max(lr_groups) if lr_groups else 0.0
            epoch_metrics = {**train_metrics, **val_metrics, "lr": lr_main}
            if len(lr_groups) >= 2:
                epoch_metrics["lr_encoder"] = lr_groups[0]
                epoch_metrics["lr_decoder"] = lr_groups[1]
            if sched_monitor not in epoch_metrics:
                raise ValueError(
                    f"Trường lr_schedule.monitor='{sched_monitor}' không tồn tại trong từ điển chỉ số epoch. "
                    f"Các trường hiện có: {sorted(epoch_metrics.keys())}"
                )

            # --- Ghi nhật ký ---
            self.log.log(epoch_metrics, step=epoch + 1)

            # --- In tiến độ huấn luyện ---
            if self.is_main_process:
                lr_text = f"lr={lr_main:.2e}"
                if len(lr_groups) >= 2:
                    lr_text = f"lr_enc={lr_groups[0]:.2e} lr_dec={lr_groups[1]:.2e}"
                print(
                    f"Epoch {epoch + 1:03d} | "
                    f"loss={train_metrics['train_loss']:.4f} "
                    f"dice={train_metrics['train_dice']:.4f} | "
                    f"val_loss={val_metrics['val_loss']:.4f} "
                    f"val_dice={val_metrics['val_dice']:.4f} | "
                    f"{lr_text}"
                )

            # --- Kiểm tra và Lưu trữ mô hình ---
            monitor_value = val_metrics[monitor]
            is_best = False
            if self.is_main_process:
                is_best = checkpoint.step(
                    monitor_value,
                    self.model,
                    epoch,
                    extra={
                        "val_iou": val_metrics["val_iou"],
                        "val_metric_semantics": _VAL_METRIC_SEMANTICS,
                    },
                    model_config=self.config.model.to_dict(),
                )
                if is_best:
                    best_metrics = epoch_metrics.copy()
                    print(f"  ✓ Đã lưu mô hình tốt nhất ({monitor}={monitor_value:.4f})")

            # --- Lưu trạng thái checkpoint cuối cùng phục vụ khôi phục ---
            if self.is_main_process:
                self._save_last_checkpoint(
                    output_dir=output_dir,
                    epoch=epoch,
                    checkpoint=checkpoint,
                    best_metrics=best_metrics,
                    monitor=monitor,
                )

            # --- Bộ điều chỉnh tỷ lệ học (bỏ qua trong giai đoạn khởi động) ---
            if epoch >= warmup_epochs:
                sched = self.config.lr_schedule.scheduler.lower()
                if sched == "reduce_on_plateau":
                    self.scheduler.step(epoch_metrics[sched_monitor])
                else:
                    self.scheduler.step()

            # --- Điều kiện dừng sớm (Early Stopping) ---
            should_stop = False
            if self.is_main_process and early_stop.step(monitor_value):
                should_stop = True
                print(
                    f"\nKích hoạt dừng sớm tại epoch {epoch + 1}. "
                    f"Chỉ số {monitor} tốt nhất đạt được: {early_stop.best:.4f}"
                )

            if self.is_distributed:
                stop_tensor = torch.tensor(
                    int(should_stop),
                    device=self.device,
                    dtype=torch.int32,
                )
                dist.broadcast(stop_tensor, src=0)
                should_stop = bool(stop_tensor.item())

            if should_stop:
                break

        total_epochs = max(last_epoch + 1, start_epoch)
        summary = {
            "best_epoch": checkpoint.best_epoch,
            "best_metrics": best_metrics,
            "total_epochs": total_epochs,
        }
        if self.is_main_process:
            print("=" * 70)
            print("  QUÁ TRÌNH HUẤN LUYỆN HOÀN TẤT")
            print("=" * 70)

            # --- Vẽ biểu đồ học tập ---
            curves_path = output_dir / "training_curves.png"
            plot_training_curves(self.log.history, curves_path)

            # --- Lưu tóm tắt kết quả huấn luyện ra file json ---
            with open(output_dir / "training_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            self.log.log_summary(best_metrics)
        return summary

    def _save_last_checkpoint(
        self,
        output_dir: Path,
        epoch: int,
        checkpoint: ModelCheckpoint,
        best_metrics: dict[str, Any],
        monitor: str,
    ) -> None:
        """Lưu trữ toàn bộ trạng thái huấn luyện để khôi phục (ghi đè sau mỗi epoch)."""
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        payload = {
            "epoch": epoch,
            "model_state_dict": model_ref.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "monitor": monitor,
            "best": checkpoint.best,
            "best_epoch": checkpoint.best_epoch,
            "best_metrics": best_metrics,
            "training_history": self.log.history,
            "val_metric_semantics": _VAL_METRIC_SEMANTICS,
        }
        torch.save(payload, output_dir / "last_checkpoint.pth")

    def _load_previous_best_state(
        self,
        output_dir: Path,
        monitor: str,
    ) -> tuple[float | None, int | None, dict[str, Any]]:
        """
        Khôi phục các trạng thái đánh giá tốt nhất trước đó phục vụ cho việc checkpoint an toàn.
        """
        best = self._resume_state.get("best")
        best_epoch = self._resume_state.get("best_epoch")
        best_metrics = self._resume_state.get("best_metrics")
        metric_semantics = self._resume_state.get("val_metric_semantics")
        if not isinstance(best_metrics, dict):
            best_metrics = {}
        else:
            best_metrics = best_metrics.copy()

        best_model_path = output_dir / "best_model.pth"
        if best_model_path.exists():
            try:
                ckpt = torch.load(best_model_path, map_location="cpu", weights_only=True)
                if metric_semantics is None:
                    metric_semantics = ckpt.get("val_metric_semantics")
                if metric_semantics == _VAL_METRIC_SEMANTICS:
                    if best is None and monitor in ckpt:
                        best = float(ckpt[monitor])
                    if best_epoch is None and "epoch" in ckpt:
                        best_epoch = int(ckpt["epoch"])
            except Exception as exc:
                logger.warning("Không thể đọc file best_model.pth để khôi phục trạng thái cũ: %s", exc)

        summary_path = output_dir / "training_summary.json"
        if metric_semantics == _VAL_METRIC_SEMANTICS and summary_path.exists():
            try:
                with open(summary_path, "r") as f:
                    summary = json.load(f)
                if not best_metrics and isinstance(summary.get("best_metrics"), dict):
                    best_metrics = summary["best_metrics"]
                if best_epoch is None and summary.get("best_epoch") is not None:
                    best_epoch = int(summary["best_epoch"])
            except Exception as exc:
                logger.warning(
                    "Không thể đọc file training_summary.json để khôi phục trạng thái cũ: %s", exc
                )

        if metric_semantics != _VAL_METRIC_SEMANTICS:
            logger.warning(
                "Checkpoint dùng để phục hồi sử dụng chỉ số cũ hoặc thiếu metadata. "
                "Đặt lại trạng thái cũ để tránh so sánh nhầm lẫn với %s.",
                _VAL_METRIC_SEMANTICS,
            )
            return None, None, {}

        if best is None and isinstance(best_metrics.get(monitor), (float, int)):
            best = float(best_metrics[monitor])

        return best, best_epoch, best_metrics

    # ------------------------------------------------------------------
    # Nạp và Đọc Checkpoint
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: Path | str, resume: bool = False) -> dict:
        """
        Nạp cấu trúc trọng số mô hình từ tệp tin checkpoint.

        Các tham số đầu vào:
            path:   Đường dẫn tới tệp checkpoint (.pth).
            resume: Nếu bằng True, thực hiện khôi phục thêm trạng thái của optimizer, scheduler và bộ phóng đại scaler
                    để tiếp tục quá trình huấn luyện. Nếu bằng False, chỉ nạp trọng số của mô hình.

        Kết quả trả về:
            Đối tượng từ điển chứa nội dung checkpoint thô.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        load_state_dict_with_aux_compat(self.model, state, context=str(path))

        if resume:
            if "optimizer_state_dict" in ckpt:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                self._init_warmup_param_groups(force=False)
            if "scheduler_state_dict" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "scaler_state_dict" in ckpt:
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            # Khôi phục lịch sử học tập để phục vụ vẽ biểu đồ đầy đủ
            if "training_history" in ckpt and isinstance(ckpt["training_history"], list):
                self.log.history = ckpt["training_history"]
            self._resume_state = {
                "best": ckpt.get("best"),
                "best_epoch": ckpt.get("best_epoch"),
                "best_metrics": ckpt.get("best_metrics", {}),
                "val_metric_semantics": ckpt.get("val_metric_semantics"),
            }
            logger.info(
                "Đã khôi phục toàn bộ trạng thái huấn luyện thành công (optimizer + scheduler + scaler) từ %s tại epoch %d",
                path,
                ckpt.get("epoch", -1) + 1,
            )
        else:
            logger.info(f"Đã nạp trọng số mô hình thành công từ: {path}")

        return ckpt
