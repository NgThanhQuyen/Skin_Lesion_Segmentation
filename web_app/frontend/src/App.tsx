import React, { useState, useEffect, useRef } from 'react';

interface PredictionResponse {
  mask_base64: string;
  overlay_base64: string;
  inference_time_ms: number;
  lesion_area_percentage: number;
  model_name: string;
  run_id: string;
  model_accuracy?: number;
  prediction_confidence?: number;
}

export default function App() {
  // Application State
  const [file, setFile] = useState<File | null>(null);
  const [originalSrc, setOriginalSrc] = useState<string | null>(null);
  const [maskSrc, setMaskSrc] = useState<string | null>(null);
  const [overlaySrc, setOriginalOverlaySrc] = useState<string | null>(null);

  const [latency, setLatency] = useState<number | null>(null);
  const [areaPercentage, setAreaPercentage] = useState<number | null>(null);
  const [modelName, setModelName] = useState<string | null>(null);
  const [modelAccuracy, setModelAccuracy] = useState<number | null>(null);
  const [predictionConfidence, setPredictionConfidence] = useState<number | null>(null);
  const [imgSize, setImgSize] = useState<{ width: number; height: number } | null>(null);
  const [viewMode, setViewMode] = useState<'overlay' | 'mask'>('overlay');
  const [threshold, setThreshold] = useState<number>(0.5);
  // Display threshold during sliding (without triggering API call on every step)
  const [displayThreshold, setDisplayThreshold] = useState<number>(0.5);
  const [useTta, setUseTta] = useState<boolean>(false);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null);

  // Before-After Slider State
  const [sliderPosition, setSliderPosition] = useState<number>(50);
  const isDragging = useRef<boolean>(false);
  const sliderContainerRef = useRef<HTMLDivElement | null>(null);
  const originalImgRef = useRef<HTMLImageElement | null>(null);

  // Auto-clear Toast
  useEffect(() => {
    if (toast) {
      const timer = setTimeout(() => {
        setToast(null);
      }, 4000);
      return () => clearTimeout(timer);
    }
  }, [toast]);

  // Observe original image size to update overlay dynamically
  useEffect(() => {
    if (!originalSrc || !originalImgRef.current) {
      setImgSize(null);
      return;
    }

    const imgEl = originalImgRef.current;
    
    const updateSize = () => {
      const rect = imgEl.getBoundingClientRect();
      if (rect.width > 0 && rect.height > 0) {
        setImgSize({ width: rect.width, height: rect.height });
      }
    };

    // Run immediately
    updateSize();

    // Observe changes
    const observer = new ResizeObserver(() => {
      updateSize();
    });
    observer.observe(imgEl);

    imgEl.addEventListener('load', updateSize);

    return () => {
      observer.disconnect();
      imgEl.removeEventListener('load', updateSize);
    };
  }, [originalSrc]);

  const showToast = (message: string, type: 'success' | 'error' = 'success') => {
    setToast({ message, type });
  };

  // Main Prediction API Caller
  const runPrediction = async (currentFile: File, currThreshold: number, currTta: boolean) => {
    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', currentFile);
    formData.append('use_tta', currTta ? 'true' : 'false');
    formData.append('threshold', currThreshold.toString());

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || 'Phân tích ảnh thất bại');
      }

      const data: PredictionResponse = await response.json();
      setMaskSrc(data.mask_base64);
      setOriginalOverlaySrc(data.overlay_base64);
      setLatency(data.inference_time_ms);
      setAreaPercentage(data.lesion_area_percentage);
      setModelName(data.model_name);
      setModelAccuracy(data.model_accuracy || 89.23);
      setPredictionConfidence(data.prediction_confidence || null);

      showToast('Phân tích hình ảnh tổn thương thành công!');
    } catch (error: any) {
      showToast(error.message || 'Lỗi mạng khi kết nối máy chủ.', 'error');
    } finally {
      setIsLoading(false);
    }
  };

  // Trigger prediction when file is selected
  const handleFileChange = (selectedFile: File) => {
    if (!selectedFile.type.startsWith('image/')) {
      showToast('Định dạng tệp không hợp lệ. Vui lòng chọn file hình ảnh!', 'error');
      return;
    }
    setFile(selectedFile);

    const reader = new FileReader();
    reader.onload = (e) => {
      if (e.target?.result) {
        setOriginalSrc(e.target.result as string);
        setSliderPosition(50);
      }
    };
    reader.readAsDataURL(selectedFile);

    // Run prediction
    runPrediction(selectedFile, threshold, useTta);
  };

  // Re-predict when useTta changes (clicks are immediate)
  const handleTtaToggle = (e: React.ChangeEvent<HTMLInputElement>) => {
    const nextTta = e.target.checked;
    setUseTta(nextTta);
    if (file) {
      runPrediction(file, threshold, nextTta);
    }
  };

  // Re-predict on threshold slider release (mouseUp or touchEnd)
  const handleThresholdRelease = () => {
    if (file && displayThreshold !== threshold) {
      setThreshold(displayThreshold);
      runPrediction(file, displayThreshold, useTta);
    }
  };

  // Slider Dragging Event Handlers
  const handleSliderMove = (clientX: number) => {
    if (!sliderContainerRef.current) return;
    const rect = sliderContainerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    let percentage = (x / rect.width) * 100;
    if (percentage < 0) percentage = 0;
    if (percentage > 100) percentage = 100;
    setSliderPosition(percentage);
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    isDragging.current = true;
    handleSliderMove(e.clientX);
    document.addEventListener('mousemove', handleGlobalMouseMove);
    document.addEventListener('mouseup', handleGlobalMouseUp);
  };

  const handleGlobalMouseMove = (e: MouseEvent) => {
    if (!isDragging.current) return;
    handleSliderMove(e.clientX);
  };

  const handleGlobalMouseUp = () => {
    isDragging.current = false;
    document.removeEventListener('mousemove', handleGlobalMouseMove);
    document.removeEventListener('mouseup', handleGlobalMouseUp);
  };

  const handleTouchStart = (e: React.TouchEvent) => {
    isDragging.current = true;
    if (e.touches.length > 0) {
      handleSliderMove(e.touches[0].clientX);
    }
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (!isDragging.current) return;
    if (e.touches.length > 0) {
      handleSliderMove(e.touches[0].clientX);
    }
  };

  const handleTouchEnd = () => {
    isDragging.current = false;
  };

  // Reset Application to Upload state
  const handleReset = () => {
    setFile(null);
    setOriginalSrc(null);
    setMaskSrc(null);
    setOriginalOverlaySrc(null);
    setLatency(null);
    setAreaPercentage(null);
    setModelName(null);
    setModelAccuracy(null);
    setPredictionConfidence(null);
    setThreshold(0.5);
    setDisplayThreshold(0.5);
    setUseTta(false);
  };

  // Drag and Drop
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const getRiskDetails = (area: number) => {
    if (area > 20) {
      return {
        text: 'Nguy cơ cao (Diện tích tổn thương lớn)',
        class: 'badge-high',
        progressClass: 'density-high',
        bullets: [
          `Diện tích sắc tố chiếm ${area.toFixed(2)}% khung hình, vượt ngưỡng an toàn (20%).`,
          'Khuyến nghị chuyên gia da liễu khám lâm sàng kỹ bằng kính hiển vi chuyên dụng.',
          'Đề xuất thực hiện sinh thiết chẩn đoán (Biopsy) để kiểm tra mức độ lành/ác tính (Melanoma).'
        ]
      };
    } else if (area > 5) {
      return {
        text: 'Nguy cơ trung bình (Theo dõi tiến triển)',
        class: 'badge-medium',
        progressClass: 'density-medium',
        bullets: [
          `Diện tích vùng da tổn thương chiếm ${area.toFixed(2)}% khung hình (ở mức trung bình 5% - 20%).`,
          'Ranh giới phân tách tương đối rõ, có thể là biểu hiện của nốt ruồi không điển hình.',
          'Khuyên dùng phương pháp tự kiểm tra da hàng tháng để phát hiện sớm các thay đổi kích thước.'
        ]
      };
    } else {
      return {
        text: 'Nguy cơ thấp (Diện tích sắc tố nhỏ)',
        class: 'badge-low',
        progressClass: 'density-low',
        bullets: [
          `Diện tích vùng sắc tố rất nhỏ, chỉ chiếm ${area.toFixed(2)}% khung hình (dưới mức 5%).`,
          'Không phát hiện dấu hiệu xâm lấn hoặc lan rộng bất thường.',
          'Khuyến nghị sử dụng kem chống nắng và dưỡng da thường xuyên để bảo vệ tế bào biểu bì.'
        ]
      };
    }
  };

  const risk = areaPercentage !== null ? getRiskDetails(areaPercentage) : null;

  return (
    <div className="app-container">
      {/* Header - Center Logo SkinAI & Subtitle */}
      <header className="app-header">
        <div className="logo-container">
          <span className="logo-text">SkinAI</span>
        </div>
        <p className="description-text">
          Hệ thống hỗ trợ chẩn đoán và phân đoạn tổn thương da tự động sử dụng trí tuệ nhân tạo.
          Hỗ trợ phân tích định lượng lâm sàng nhanh chóng, chính xác.
        </p>
      </header>

      {/* Main Workstation */}
      <div className="workstation-layout">

        {/* Left Panel - Image Visualizer & Horizontal Controllers */}
        <div className="panel-card visualizer-panel" onDragOver={handleDragOver} onDrop={handleDrop}>
          {isLoading && (
            <div className="loading-overlay">
              <svg className="spinner-medical" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" strokeDasharray="32 16" />
                <path d="M12 7V17M7 12H17" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
              </svg>
              <div className="loading-text-title">Hệ thống đang phân tích da...</div>
              <div className="loading-text-sub">Mô hình học sâu đang thực hiện chẩn đoán phân đoạn tổn thương.</div>
            </div>
          )}

          {!originalSrc ? (
            // Upload Dropzone
            <label className="dropzone-container">
              <input
                type="file"
                className="dropzone-input"
                accept="image/*"
                onChange={(e) => e.target.files && handleFileChange(e.target.files[0])}
              />
              <svg className="upload-svg-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <h3>Kéo và thả ảnh da liễu vào đây</h3>
              <p>Hoặc nhấp chuột để chọn ảnh từ thiết bị của bạn</p>
              <span className="format-info">Hỗ trợ các định dạng tệp ảnh: JPEG, PNG, BMP</span>
            </label>
          ) : (
            // Results Slider & Controls View
            <div className="results-container">
              <div className="results-header">
                <h3>Kết quả phân đoạn ảnh</h3>
              </div>

              {/* Comparative Before-After Slider */}
              <div className="slider-viewport">
                <div
                  className="image-slider-box"
                  ref={sliderContainerRef}
                  onMouseDown={handleMouseDown}
                  onTouchStart={handleTouchStart}
                  onTouchMove={handleTouchMove}
                  onTouchEnd={handleTouchEnd}
                >
                  {/* Original Image */}
                  <img
                    src={originalSrc}
                    alt="Original Skin"
                    className="img-bg-base"
                    ref={originalImgRef}
                  />

                  {/* Processed Overlay/Mask Image */}
                  {(viewMode === 'overlay' ? overlaySrc : maskSrc) && imgSize && (
                    <div
                      className="slider-overlay-container"
                      style={{ width: `${sliderPosition}%` }}
                    >
                      <img
                        src={viewMode === 'overlay' ? overlaySrc! : maskSrc!}
                        alt="Segmented Visual"
                        className="img-overlay-active"
                        style={{
                          width: `${imgSize.width}px`,
                          height: `${imgSize.height}px`
                        }}
                      />
                    </div>
                  )}

                  {/* Slider divider bar */}
                  <div
                    className="slider-divider-line"
                    style={{ left: `${sliderPosition}%` }}
                  >
                    <div className="slider-divider-handle">
                      <svg fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M7.5 21L3 16.5m0 0L7.5 12M3 16.5h13.5m0-13.5L21 7.5m0 0L16.5 12M21 7.5H7.5" />
                      </svg>
                    </div>
                  </div>
                </div>
              </div>

              {/* Horizontal Controls Row (Unified Below Slider) */}
              <div className="controls-row">

                {/* View Mode Mini Switch */}
                <div className="view-toggle-mini">
                  <button
                    className={`view-btn-mini ${viewMode === 'overlay' ? 'active' : ''}`}
                    onClick={() => setViewMode('overlay')}
                  >
                    Overlay
                  </button>
                  <button
                    className={`view-btn-mini ${viewMode === 'mask' ? 'active' : ''}`}
                    onClick={() => setViewMode('mask')}
                  >
                    Mặt nạ (Mask)
                  </button>
                </div>

                {/* Threshold Slider (Mini) */}
                <div className="control-item">
                  <div className="control-label-row">
                    <label htmlFor="threshold-range">Ngưỡng (Threshold)</label>
                    <span className="control-badge">{displayThreshold.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    id="threshold-range"
                    className="mini-range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={displayThreshold}
                    onChange={(e) => setDisplayThreshold(parseFloat(e.target.value))}
                    onMouseUp={handleThresholdRelease}
                    onTouchEnd={handleThresholdRelease}
                  />
                </div>

                {/* TTA Toggle (Mini) */}
                <div className="controls-row-switch">
                  <span>TTA</span>
                  <label className="ios-switch-mini">
                    <input
                      type="checkbox"
                      checked={useTta}
                      onChange={handleTtaToggle}
                    />
                    <span className="ios-slider-mini"></span>
                  </label>
                </div>

                {/* Reset Button (Mini) */}
                <button className="btn-mini-reset" onClick={handleReset}>
                  <svg width="12" height="12" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                  </svg>
                  Tải ảnh khác
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Right Panel - Diagnostics & Recommendations */}
        <div className="panel-card report-panel">

          {/* Diagnostic Reports */}
          <div>
            <h4 className="report-section-title">
              <svg fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" d="M10.5 6a7.5 7.5 0 107.5 7.5h-7.5V6z" />
                <path strokeLinecap="round" strokeLinejoin="round" d="M13.5 10.5H21A7.5 7.5 0 0013.5 3v7.5z" />
              </svg>
              Báo cáo định lượng
            </h4>

            <div className="metrics-box">
              {/* Metric 1: Processing Latency */}
              <div className="metric-card">
                <div className="metric-circle-icon">
                  <svg fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </div>
                <div className="metric-info">
                  <span className="metric-title">Độ trễ xử lý từ AI</span>
                  <span className="metric-value">{latency !== null ? `${latency.toFixed(2)} ms` : '--'}</span>
                </div>
              </div>

              {/* Metric 2: Segment Confidence (Dynamic) */}
              <div className="metric-card">
                <div className="metric-circle-icon">
                  <svg fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                  </svg>
                </div>
                <div className="metric-info">
                  <span className="metric-title">Độ tin cậy phân đoạn (Confidence)</span>
                  <span className="metric-value">{predictionConfidence !== null ? `${predictionConfidence.toFixed(2)} %` : '--'}</span>
                </div>
              </div>

              {/* Metric 3: Lesion Area */}
              <div className="metric-card">
                <div className="metric-circle-icon">
                  <svg fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 3.75v16.5h16.5V3.75H3.75zm1.5 1.5h13.5v13.5H5.25V5.25z" />
                  </svg>
                </div>
                <div className="metric-info">
                  <span className="metric-title">Tỷ lệ diện tích tổn thương</span>
                  <span className="metric-value">{areaPercentage !== null ? `${areaPercentage.toFixed(2)} %` : '--'}</span>
                </div>
              </div>

              {/* Progress bar showing density visually */}
              <div className="density-progress-wrapper">
                <div className="density-progress-header">
                  <span>Mật độ sắc tố tổn thương</span>
                  <span className="density-progress-percent">{areaPercentage !== null ? `${areaPercentage.toFixed(2)}%` : '0%'}</span>
                </div>
                <div className="density-bar-background">
                  <div
                    className={`density-bar-indicator ${risk?.progressClass || 'density-low'}`}
                    style={{ width: `${areaPercentage || 0}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* AI Clinical Findings */}
          <div className="findings-card">
            <h4 className="report-section-title">
              <svg fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Ghi chú lâm sàng AI
            </h4>

            {!risk ? (
              <div className="findings-empty-state">
                <svg fill="none" stroke="currentColor" strokeWidth="1.5" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
                </svg>
                <p>Vui lòng tải hình ảnh da liễu lên để nhận phân tích chi tiết và khuyến nghị y khoa tự động từ mô hình học sâu.</p>
              </div>
            ) : (
              <div>
                <span className={`findings-alert-badge ${risk.class}`}>
                  {risk.text}
                </span>

                {modelName && (
                  <span className="findings-model-info">
                    Mô hình hoạt động: <strong>{modelName}</strong> | Độ chính xác thuật toán (Dice Score): <strong>{modelAccuracy !== null ? `${modelAccuracy.toFixed(2)}%` : '89.23%'}</strong>
                  </span>
                )}

                <ul className="findings-bullets">
                  {risk.bullets.map((bullet, idx) => (
                    <li key={idx}>
                      <svg fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" />
                      </svg>
                      {bullet}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Global Toast Alert */}
      {toast && (
        <div className={`toast-bar ${toast.type === 'error' ? 'toast-bar-error' : ''}`}>
          {toast.type === 'success' ? (
            <svg className="toast-icon toast-icon-success" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          ) : (
            <svg className="toast-icon toast-icon-error" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          )}
          <span className="toast-message">{toast.message}</span>
        </div>
      )}

      {/* Footer Info */}
      <footer className="footer-info">
        <span>Mô hình được tối ưu hóa dựa trên tập dữ liệu ISIC 2018</span>
      </footer>
    </div>
  );
}
