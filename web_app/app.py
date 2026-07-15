import os
import sys
import time
import io
import base64
import logging
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
import torch

# Đảm bảo import được các module trong src/ bằng cách thêm thư mục gốc vào sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.models.segmentation import create_model
from src.data.transforms import get_transforms
from src.inference.tta import tta_predict
from src.utils.checkpoint import load_state_dict_with_aux_compat
from src.utils.config import Config
from web_app.dagshub_sync import sync_best_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web_app")

app = FastAPI(title="Skin Lesion Segmentation Web App", version="1.0.0")

# Khai báo và cấu hình thư mục React frontend
react_dist_dir = _REPO_ROOT / "web_app" / "frontend" / "dist"
react_assets_dir = react_dist_dir / "assets"
react_assets_dir.mkdir(parents=True, exist_ok=True) # Đảm bảo thư mục tồn tại để tránh lỗi mount khi chưa build
app.mount("/assets", StaticFiles(directory=str(react_assets_dir)), name="assets")

# Biến toàn cục lưu trữ mô hình và cấu hình đang chạy
model: Optional[torch.nn.Module] = None
transform = None
model_config: Optional[Config] = None
run_id: Optional[str] = None
architecture_name: Optional[str] = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
load_error_message: Optional[str] = None

def load_model_pipeline():
    """Luồng tải cấu hình, tải weights và nạp mô hình PyTorch vào bộ nhớ."""
    global model, transform, model_config, run_id, architecture_name, load_error_message
    try:
        config_dict, weight_path, rid = sync_best_model()
        model_config = Config(config_dict)
        run_id = rid
        architecture_name = config_dict.get("logging", {}).get("experiment_name", "ResNet34 + UNet")
        
        # Tạo mô hình
        model = create_model(model_config).to(device)
        model.eval()
        
        # Load weights
        ckpt = torch.load(weight_path, map_location=device, weights_only=True)
        state = ckpt.get("model_state_dict", ckpt)
        load_state_dict_with_aux_compat(model, state, context=str(weight_path))
        
        # Khởi tạo transform tiền xử lý
        transform = get_transforms("val", model_config)
        load_error_message = None
        logger.info(f"Đã nạp thành công mô hình: {architecture_name} từ DagsHub Run: {run_id}")
    except Exception as e:
        load_error_message = str(e)
        logger.error(f"Nạp mô hình thất bại: {e}")

@app.on_event("startup")
async def startup_event():
    # Thử nạp mô hình lúc khởi động, nếu thiếu .env thì nạp chế độ trì hoãn (lazy load)
    load_model_pipeline()

def image_to_base64(img: Image.Image) -> str:
    """Mã hóa ảnh PIL sang chuỗi Base64 PNG."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def get_overlay_image(orig_img: Image.Image, mask_np: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Tạo ảnh phủ màu (Overlay) màu xanh ngọc (Cyan/Teal) sang trọng đè lên ảnh gốc."""
    orig = np.array(orig_img.convert("RGB"))
    h_mask = mask_np > 127
    
    # Màu xanh ngọc tế bào chuyên nghiệp: R=0, G=190, B=210
    overlay_color = np.zeros_like(orig)
    overlay_color[:, :, 1] = 190
    overlay_color[:, :, 2] = 210
    
    overlay = orig.copy().astype(float)
    overlay[h_mask] = (1 - alpha) * orig[h_mask] + alpha * overlay_color[h_mask]
    overlay = overlay.astype(np.uint8)
    
    return Image.fromarray(overlay)

@app.get("/", response_class=HTMLResponse)
async def index():
    """Trang chủ phục vụ ứng dụng React."""
    react_index_path = react_dist_dir / "index.html"
    if not react_index_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Giao diện React chưa được biên dịch. Vui lòng chạy 'npm run build' trong thư mục web_app/frontend."
        )
    try:
        with open(react_index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except Exception as e:
        logger.error(f"Lỗi khi đọc file React index: {e}")
        raise HTTPException(status_code=500, detail=f"Không thể đọc file giao diện: {str(e)}")

@app.post("/api/sync")
async def sync_model():
    """API trigger đồng bộ tải mô hình thủ công từ giao diện."""
    load_model_pipeline()
    if model is None:
        raise HTTPException(status_code=500, detail=f"Không thể đồng bộ mô hình: {load_error_message}")
    return {
        "status": "success",
        "architecture": architecture_name,
        "run_id": run_id
    }

@app.post("/api/predict")
async def predict(file: UploadFile = File(...), use_tta: bool = Form(False), threshold: float = Form(0.5)):
    """API chính nhận ảnh da, chạy mô hình phân đoạn và trả về kết quả dưới dạng Base64."""
    global model, transform, device
    if model is None:
        raise HTTPException(
            status_code=400, 
            detail="Mô hình chưa được nạp. Hãy điền file .env và nhấn nút Đồng bộ hóa mô hình trên web."
        )
        
    start_time = time.time()
    
    try:
        # Đọc ảnh đầu vào
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        orig_w, orig_h = image_pil.size
        
        # Tiền xử lý ảnh
        img_np = np.array(image_pil)
        dummy_mask = np.zeros(img_np.shape[:2], dtype=np.float32)
        transformed = transform(image=img_np, mask=dummy_mask)
        x = transformed["image"].unsqueeze(0).to(device)
        
        # Chạy dự đoán
        with torch.no_grad():
            if use_tta:
                probs = tta_predict(model, x)
            else:
                probs = torch.sigmoid(model(x))
                
            # Đưa kết quả ra numpy và đưa về kích thước gốc của ảnh
            mask_pred = (probs[0, 0] > threshold).cpu().numpy().astype(np.uint8) * 255
            
        # Đưa mặt nạ về kích thước gốc của ảnh
        if mask_pred.shape != (orig_h, orig_w):
            mask_resized = np.array(
                Image.fromarray(mask_pred).resize((orig_w, orig_h), resample=Image.Resampling.NEAREST)
            )
        else:
            mask_resized = mask_pred
            
        # Tạo ảnh phủ đặc trưng (Overlay)
        overlay_pil = get_overlay_image(image_pil, mask_resized)
        mask_pil = Image.fromarray(mask_resized)
        
        # Mã hóa Base64
        mask_base64 = image_to_base64(mask_pil)
        overlay_base64 = image_to_base64(overlay_pil)
        
        # Tính toán chỉ số diện tích tổn thương (%)
        pixels_total = mask_resized.size
        pixels_lesion = np.sum(mask_resized > 127)
        area_percentage = (pixels_lesion / pixels_total) * 100
        
        # Tính toán độ tin cậy dự đoán (confidence score) động
        probs_tensor = probs[0, 0]
        confidence_map = torch.where(probs_tensor > threshold, probs_tensor, 1.0 - probs_tensor)
        prediction_confidence = confidence_map.mean().item() * 100

        inference_time_ms = (time.time() - start_time) * 1000
        
        return {
            "model_name": architecture_name,
            "run_id": run_id,
            "inference_time_ms": round(inference_time_ms, 2),
            "lesion_area_percentage": round(area_percentage, 2),
            "mask_base64": mask_base64,
            "overlay_base64": overlay_base64,
            "model_accuracy": 89.23,
            "prediction_confidence": round(prediction_confidence, 2)
        }
        
    except Exception as e:
        logger.error(f"Lỗi khi thực hiện phân đoạn ảnh: {e}")
        raise HTTPException(status_code=500, detail=f"Xử lý ảnh thất bại: {str(e)}")
