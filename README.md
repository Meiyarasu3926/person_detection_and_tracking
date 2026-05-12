# AI-Based Intelligent Camera Security System

## 1. Project Overview
Real-time **Person Detection & Tracking** using **YOLO26 + ByteTrack**.  
The system is powered by a **Streamlit web UI** (`app.py`) that accepts image and video uploads, draws bounding boxes with track IDs, fires per-person alerts, and saves annotated outputs.

## 2. Problem Statement
Detect and continuously track persons in surveillance footage.  
Fire an alert the first time each unique person enters the scene.

## 3. Dataset Used
| Item | Detail |
|------|--------|
| Base weights | COCO (80 classes, pre-trained by Ultralytics) |
| Person class | Class 0 — "person" |
| Custom fine-tune | Optional; place annotated data in `data_sample/` and run training |
| Format | YOLO `.txt` labels or Roboflow export |

> No custom training is required to run inference; `yolo26n.pt` downloads automatically.

## 4. System Architecture
```
Image / Video Upload
        │
        ▼
  ┌─────────────────────┐
  │  app.py             │
  │  Streamlit Web UI   │
  │  (Image & Video tab)│
  └──────────┬──────────┘
             │
             ▼
   YOLO26n  (classes=[0], person only, NMS-free)
             │  bounding boxes + conf
             ▼
   ByteTrack  (persist=True, stable IDs)
             │
             ▼
   draw_box()  ──►  Annotated frame
             │
     ┌───────┴────────┐
     ▼                ▼
  Annotated        Metrics Dashboard
  Image/Video      (Persons · FPS · Inference ms)
  Download
```

## 5. Model Selection
| Model | Why |
|-------|-----|
| `yolo26n.pt` | Latest YOLO generation — NMS-free end-to-end inference, MuSGD optimizer, up to 43% faster CPU inference, better small-object detection via ProgLoss + STAL, COCO-pretrained |

> YOLO26 removes the DFL module and NMS post-processing entirely. Predictions are generated directly, reducing latency and simplifying deployment on edge/low-power devices.

## 6. Installation
```bash
git clone https://github.com/Meiyarasu3926/person_detection_and_tracking.git
cd person_detection_and_tracking
pip install -r requirements.txt
```

**requirements.txt**
```
ultralytics>=8.2.0
opencv-python>=4.9.0
numpy>=1.26.0
streamlit>=1.35.0
Pillow>=10.0.0
```

## 7. How to Run

### Streamlit Web App
```bash
streamlit run app.py
```
Opens at `http://localhost:8501` with two tabs:

| Tab | Feature |
|-----|---------|
| 📷 Image Detection | Upload JPG/PNG → detect persons → download annotated image |
| 🎬 Video Tracking | Upload MP4/AVI/MOV → ByteTrack → live preview → download output video |

**Sidebar controls:** confidence threshold · IoU threshold · max frames · preview interval

## 8. Training Details
Pre-trained COCO weights are used out of the box.  
To fine-tune on a custom dataset:
```bash
# Annotate with labelImg or Roboflow, then:
yolo train model=yolo26n.pt data=data_sample/data.yaml epochs=30 imgsz=640
```

## 9. Inference Pipeline
1. Image/video uploaded via Streamlit UI
2. `model.track(frame, persist=True, classes=[0])` — detects + tracks persons only
3. Bounding boxes drawn with unique color per track ID; alert fired on first appearance
4. Live preview updated every N frames; metrics shown in dashboard cards
5. Annotated image/video available for download

## 10. Results

| Output | Location |
|--------|----------|
| Annotated image | In-browser preview + download |
| Annotated video | In-browser preview + download |
| Metrics dashboard | Persons · Inference ms · Avg FPS |

Sample alert:
```
ALERT: Person #3 detected — 91.2% confidence  [frame 42]
```

## 11. Evaluation Metrics

### YOLO26n — COCO Benchmark
| Model | Size (px) | mAP val 50-95 | mAP val 50-95 (e2e) | Speed CPU ONNX (ms) | Speed T4 TensorRT10 (ms) | Params (M) | FLOPs (B) |
|-------|-----------|---------------|----------------------|----------------------|--------------------------|------------|-----------|
| YOLO26n | 640 | **40.9** | **40.1** | 38.9 ± 0.7 | 1.7 ± 0.0 | 2.4 | 5.4 |

### Runtime Metrics (displayed live in Streamlit UI)
| Metric | Description |
|--------|-------------|
| Persons detected | Count of persons in current frame / image |
| Inference time (ms) | Time taken per frame by YOLO26n |
| Avg FPS | Average frames processed per second over the video |
| Unique persons tracked | Total distinct track IDs seen across all frames |

### Detection Quality
| Metric | How it is measured |
|--------|--------------------|
| Precision | TP / (TP + FP) at IoU ≥ 0.5 |
| Recall | TP / (TP + FN) at IoU ≥ 0.5 |
| F1-score | Harmonic mean of Precision and Recall |
| mAP@50-95 | 40.9 (COCO val, YOLO26n standard) |

> The Streamlit dashboard shows per-session FPS and inference time. For full Precision/Recall/F1 evaluation against annotated ground truth, compare `results/detections.csv` against your annotation file using IoU matching at threshold 0.5.

## 12. Limitations
- Occlusion can cause track-ID switches
- `yolo26n` (nano) may miss very small or distant persons; use `yolo26s/m` for higher accuracy
- No re-identification across camera cuts or scene changes
- Streamlit video preview is near-real-time, not true real-time streaming

## 13. Future Improvements
- Fine-tune on domain-specific CCTV datasets using MuSGD optimizer
- Upgrade to `yolo26s` or `yolo26m` for higher accuracy at moderate compute cost
- Add Re-ID module (e.g. StrongSORT) for cross-camera tracking
- Combine with anomaly / fight detection head
- Add webcam live-stream support directly in Streamlit
- Deploy on edge hardware (Jetson Nano) — YOLO26's NMS-free design simplifies TensorRT export

## 14. References
- Ultralytics YOLO26 — https://docs.ultralytics.com/models/yolo26
- Ultralytics GitHub — https://github.com/ultralytics/ultralytics
- ByteTrack — https://arxiv.org/abs/2110.06864
- MuSGD / Muon Optimizer — https://arxiv.org/abs/2502.16982
- COCO Dataset — https://cocodataset.org
- Streamlit — https://streamlit.io
