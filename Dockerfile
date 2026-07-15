# =============================================================================
# Stage 1: Build React Frontend
# =============================================================================
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY web_app/frontend/package*.json ./
RUN npm ci

COPY web_app/frontend/ ./
RUN npm run build

# =============================================================================
# Stage 2: Build FastAPI Backend & Run Server
# =============================================================================
FROM python:3.11-slim
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết (nếu có)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt PyTorch CPU trước để giảm dung lượng Docker image tối đa
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Cài đặt các thư viện Python khác
COPY web_app/requirements.txt ./web_app/requirements.txt
RUN pip install --no-cache-dir -r web_app/requirements.txt

# Copy source code và web_app
COPY src/ ./src
COPY configs/ ./configs
COPY web_app/ ./web_app

# Copy kết quả biên dịch React frontend vào thư mục phục vụ tĩnh của FastAPI
COPY --from=frontend-builder /app/frontend/dist/ ./web_app/frontend/dist/

# Thiết lập cổng mạng phục vụ
EXPOSE 8000

# Đặt biến môi trường Python
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Khởi chạy server
CMD ["python", "-m", "uvicorn", "web_app.app:app", "--host", "0.0.0.0", "--port", "8000"]
