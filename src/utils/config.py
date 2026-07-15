# =============================================================================
# Thư mục: src/utils/config.py
# Chức năng: Đọc, nạp cấu hình hệ thống từ file YAML, hỗ trợ kế thừa phân cấp
#            và ghi đè các tham số trực tiếp qua giao diện dòng lệnh (CLI).
# Lớp quan trọng:
#   - Config: Lớp kế thừa từ kiểu dict của Python cho phép truy cập lồng nhau bằng cú pháp dấu chấm (dot-notation).
# Hàm quan trọng:
#   - load_config: Nạp cấu hình từ file YAML, tự động xử lý kế thừa từ cấu hình gốc (_base_).
#   - override_config: Đè các giá trị cấu hình thông qua tham số dòng lệnh CLI.
# =============================================================================

from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any


class Config(dict):
    """
    Từ điển hỗ trợ truy cập bằng dấu chấm (dot-notation access), hỗ trợ cả các khóa lồng nhau.

    Các từ điển lồng nhau được bao bọc (eager wrapping) thành thực thể Config ngay tại thời điểm khởi tạo,
    nhờ đó phép gán hoặc đột biến cấu trúc (ví dụ: config.model.name = "x") luôn hoạt động ổn định.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Chuyển đổi tất cả từ điển con lồng nhau thành thực thể Config
        for k, v in self.items():
            if isinstance(v, dict) and not isinstance(v, Config):
                super().__setitem__(k, Config(v))

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Cấu hình không có thuộc tính '{key}'")

    def __setattr__(self, key: str, val: Any) -> None:
        self[key] = val

    def __setitem__(self, key: str, val: Any) -> None:
        # Tự động bao bọc khi thực hiện phép gán giá trị mới
        if isinstance(val, dict) and not isinstance(val, Config):
            val = Config(val)
        super().__setitem__(key, val)

    def __repr__(self) -> str:
        import json

        return json.dumps(dict(self), indent=2, default=str)

    def to_dict(self) -> dict:
        """Chuyển đổi thực thể Config hiện tại về kiểu từ điển gốc (plain dict) của Python, bao gồm cả các lớp lồng nhau."""
        result = {}
        for k, v in self.items():
            result[k] = v.to_dict() if isinstance(v, Config) else v
        return result


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Gộp (merge) đè các giá trị từ override vào base một cách đệ quy.
    Các giá trị trong override sẽ được ưu tiên hơn ở tất cả các cấp lồng nhau.
    """
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _load_yaml(path: Path) -> dict:
    """Đọc file cấu hình YAML."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: str | Path) -> Config:
    """
    Nạp cấu hình YAML hỗ trợ cơ chế kế thừa phân cấp bằng từ khóa _base_.

    Ví dụ khai báo trong file cấu hình thử nghiệm:
        _base_: ../base.yaml
        training:
            lr: 3.0e-4   # Chỉ đè giá trị này

    Các tham số đầu vào:
        config_path: Đường dẫn tới file cấu hình YAML thử nghiệm.

    Kết quả trả về:
        Đối tượng Config hỗ trợ truy cập bằng dấu chấm.
    """
    config_path = Path(config_path)
    raw = _load_yaml(config_path)

    # Xử lý kế thừa từ cấu hình cơ sở _base_
    if "_base_" in raw:
        base_rel = raw.pop("_base_")
        base_path = (config_path.parent / base_rel).resolve()
        base_cfg = load_config(base_path)  # Gọi đệ quy (hỗ trợ nhiều cấp kế thừa)
        merged = _deep_merge(base_cfg.to_dict(), raw)
    else:
        merged = raw

    return Config(merged)


def _cast_value(value_str: str) -> Any:
    """
    Tự động chuyển đổi kiểu chuỗi ký tự nhận được từ CLI về đúng định dạng kiểu dữ liệu Python tương ứng.
    Ví dụ: "true"/"false" -> bool, "1.5e-4" -> float, "42" -> int, "[1,2]" -> list.
    """
    # Xử lý kiểu Boolean
    if value_str.lower() == "true":
        return True
    if value_str.lower() == "false":
        return False
    if value_str.lower() == "null" or value_str.lower() == "none":
        return None

    # Xử lý kiểu List (ví dụ: [256,256] hoặc [256, 256])
    if value_str.startswith("[") and value_str.endswith("]"):
        inner = value_str[1:-1].split(",")
        return [_cast_value(v.strip()) for v in inner if v.strip()]

    # Xử lý kiểu số nguyên (Int)
    try:
        return int(value_str)
    except ValueError:
        pass

    # Xử lý kiểu số thực (Float, bao gồm cả cách viết e-4)
    try:
        return float(value_str)
    except ValueError:
        pass

    # Mặc định trả về chuỗi ký tự ban đầu
    return value_str


def _set_nested(d: dict, key_path: str, value: Any) -> None:
    """
    Gán giá trị vào từ điển dựa trên đường dẫn dấu chấm.
    Ví dụ: _set_nested(d, "training.lr", 1e-3) -> d["training"]["lr"] = 1e-3.
    """
    keys = key_path.split(".")
    current = d
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def _validate_key_exists(d: dict, key_path: str) -> None:
    """
    Xác thực xem đường dẫn dấu chấm có tồn tại trong từ điển cấu hình hay không.

    Ngoại lệ:
        ValueError: Nếu đường dẫn khóa không tồn tại (giúp phát hiện lỗi chính tả).
    """
    keys = key_path.split(".")
    current = d
    for i, key in enumerate(keys):
        if not isinstance(current, dict) or key not in current:
            traversed = ".".join(keys[: i + 1])
            raise ValueError(
                f"Khóa cấu hình '{key_path}' không tồn tại "
                f"(gặp lỗi tại '{traversed}'). Vui lòng kiểm tra lại tên khóa."
            )
        current = current[key]


def override_config(
    config: Config,
    overrides: list[str],
    strict: bool = True,
) -> Config:
    """
    Ghi đè cấu hình dựa trên các tham số dòng lệnh CLI có dạng "key.subkey=value".

    Ví dụ:
        overrides = [
            "data.root=/kaggle/input/isic-2018",
            "output.dir=/kaggle/working",
            "training.batch_size=32",
            "logging.use_wandb=false",
        ]

    Các tham số đầu vào:
        config:    Đối tượng cấu hình ban đầu đã nạp.
        overrides: Danh sách các chuỗi ghi đè "khóa=giá trị".
        strict:    Nếu bằng True, ném ngoại lệ nếu phát hiện khóa ghi đè không tồn tại trong cấu hình gốc.
                   Đặt bằng False để cho phép khai báo thêm các khóa cấu hình mới.

    Kết quả trả về:
        Thực thể Config mới đã qua ghi đè.
    """
    if not overrides:
        return config

    config_dict = config.to_dict()

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Tham số ghi đè '{item}' không hợp lệ. Định dạng yêu cầu là 'key.subkey=value'")
        key_path, _, value_str = item.partition("=")
        key_path = key_path.strip()

        if strict:
            _validate_key_exists(config_dict, key_path)

        value = _cast_value(value_str)
        _set_nested(config_dict, key_path, value)

    return Config(config_dict)
