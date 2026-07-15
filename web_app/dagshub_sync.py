import os
import sys
import json
import logging
from pathlib import Path
from dotenv import load_dotenv

# Đảm bảo import được các module trong src/ bằng cách thêm thư mục gốc vào sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Load environment variables từ file .env
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(env_path)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dagshub_sync")

def get_fallback_config_path(arch_name: str) -> Path:
    """
    Ánh xạ tên kiến trúc mô hình sang file cấu hình local trong dự án.
    """
    arch_lower = arch_name.lower()
    exp_dir = _REPO_ROOT / "configs" / "experiments"
    
    if "resnet34" in arch_lower:
        path = exp_dir / "resnet34_unet_kaggle_t4.yaml"
        if not path.exists():
            path = exp_dir / "resnet34_unet_v1.yaml"
    elif "transunet" in arch_lower:
        path = exp_dir / "transunet_r50_vitb16_kaggle_t4.yaml"
    elif "deeplabv3" in arch_lower or "mobilenet" in arch_lower:
        path = exp_dir / "mobilenetv3_deeplabv3_v1.yaml"
    elif "unet" in arch_lower and "original" in arch_lower:
        path = exp_dir / "unet_original_v1.yaml"
    else:
        path = _REPO_ROOT / "configs" / "base.yaml"
        
    return path

def sync_best_model():
    """
    Tải về mô hình tốt nhất (.pth) và cấu hình từ DagsHub.
    Trả về:
        dict: Chứa thông tin config (load từ JSON hoặc fallback yaml)
        Path: Đường dẫn tới file weight .pth cục bộ
        str: Mã Run ID của DagsHub
    """
    username = os.getenv("DAGSHUB_USERNAME")
    repo_name = os.getenv("DAGSHUB_REPO")
    token = os.getenv("DAGSHUB_TOKEN")
    
    if not username or not repo_name or not token:
        raise ValueError(
            "Thiếu cấu hình DagsHub trong file web_app/.env. "
            "Vui lòng sao chép file .env.example sang .env và điền đầy đủ thông tin."
        )
        
    # Thư mục cache lưu trữ mô hình
    checkpoint_dir = Path(__file__).resolve().parent / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Khởi tạo đăng nhập DagsHub
    import dagshub
    import mlflow
    
    logger.info("Đang kết nối tới DagsHub...")
    dagshub.auth.add_app_token(token)
    dagshub.init(repo_owner=username, repo_name=repo_name, mlflow=True)
    
    # Kết nối MLflow Client để tìm kiếm run
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    
    exps = client.search_experiments()
    exp_ids = [e.experiment_id for e in exps]
    if not exp_ids:
        raise ValueError("Không tìm thấy Experiment nào trên DagsHub MLflow.")
        
    runs = client.search_runs(
        experiment_ids=exp_ids,
        order_by=["attribute.start_time DESC"]
    )
    
    best_run = None
    for run in runs:
        if run.data.tags.get("mlflow.runName") == "Best_Model_Upload":
            best_run = run
            break
            
    if not best_run:
        raise ValueError("Không tìm thấy Run 'Best_Model_Upload' chứa mô hình tốt nhất trên DagsHub.")
        
    run_id = best_run.info.run_id
    arch_name = best_run.data.params.get("best_model_architecture", "ResNet34 + UNet")
    logger.info(f"Tìm thấy mô hình tốt nhất trên DagsHub: {arch_name} (Run ID: {run_id})")
    
    # Kiểm tra metadata cache cục bộ xem có trùng khớp run_id hay không
    metadata_file = checkpoint_dir / "metadata.json"
    cached_run_id = None
    if metadata_file.exists():
        try:
            with open(metadata_file, "r") as f:
                cached_data = json.load(f)
                cached_run_id = cached_data.get("run_id")
        except Exception:
            pass
            
    weight_path = checkpoint_dir / "best_model.pth"
    config_json_path = checkpoint_dir / "best_config.json"
    
    # Nếu trùng khớp run_id và các tệp đều tồn tại thì nạp từ cache cục bộ
    if cached_run_id == run_id and weight_path.exists() and config_json_path.exists():
        logger.info("Mô hình đã được lưu cục bộ và khớp với DagsHub. Đang tải từ cache...")
        with open(config_json_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return config_dict, weight_path, run_id
        
    # Tiến hành tải tệp weights từ DagsHub
    logger.info("Đang tải tệp trọng số (best_model.pth) từ DagsHub...")
    try:
        downloaded_pth = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="best_model.pth",
            dst_path=str(checkpoint_dir)
        )
        # mlflow trả về đường dẫn tải về. Đảm bảo nó được chuyển đúng tên file trong checkpoint_dir
        if Path(downloaded_pth) != weight_path:
            import shutil
            shutil.move(downloaded_pth, weight_path)
    except Exception as e:
        raise RuntimeError(f"Tải file trọng số thất bại: {e}")
        
    # Thử tải tệp config.json từ DagsHub (nếu có)
    config_dict = None
    try:
        logger.info("Đang kiểm tra danh sách artifacts trên DagsHub...")
        artifacts = client.list_artifacts(run_id)
        has_config = any(art.path == "best_config.json" for art in artifacts)
        
        if has_config:
            logger.info("Tìm thấy best_config.json trên DagsHub. Đang tải...")
            downloaded_json = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path="best_config.json",
                dst_path=str(checkpoint_dir)
            )
            if Path(downloaded_json) != config_json_path:
                import shutil
                shutil.move(downloaded_json, config_json_path)
                
            with open(config_json_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
                logger.info("Đã tải cấu hình best_config.json từ DagsHub thành công.")
        else:
            logger.info("Không tìm thấy cấu hình best_config.json trên DagsHub (sử dụng cơ chế Fallback local).")
    except Exception as e:
        logger.warning(f"Không thể kiểm tra hoặc tải cấu hình từ DagsHub: {e}. Sử dụng cơ chế Fallback local...")
        
    # Cơ chế Fallback nếu không có cấu hình trên DagsHub
    if not config_dict:
        fallback_yaml = get_fallback_config_path(arch_name)
        logger.info(f"Đang sử dụng cấu hình cục bộ làm fallback: {fallback_yaml.name}")
        from src.utils.config import load_config
        config_obj = load_config(fallback_yaml)
        config_dict = config_obj.to_dict()
        
        # Lưu vào cache cục bộ để sử dụng sau này
        with open(config_json_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
    # Ghi đè metadata mới
    with open(metadata_file, "w") as f:
        json.dump({"run_id": run_id, "architecture": arch_name}, f)
        
    logger.info("Đồng bộ mô hình tốt nhất từ DagsHub thành công!")
    return config_dict, weight_path, run_id

if __name__ == "__main__":
    try:
        config, weight, rid = sync_best_model()
        print(f"Thành công! Run ID: {rid}")
        print(f"Đường dẫn weights: {weight}")
        print(f"Tên mô hình trong config: {config['model']['name']}")
    except Exception as e:
        print(f"Lỗi: {e}")
