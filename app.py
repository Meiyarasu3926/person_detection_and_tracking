"""
app.py — Streamlit UI for Person Detection & Tracking
Run: streamlit run app.py
"""

import cv2
import time
import tempfile
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SecureVision · Person Tracker",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0a0c10 !important;
    font-family: 'Rajdhani', sans-serif;
    color: #c8d6e5;
}
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #1e2a38;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── Title ── */
.sv-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2rem;
    color: #00e5ff;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-bottom: 2px;
}
.sv-sub {
    font-size: 0.85rem;
    color: #4a6274;
    letter-spacing: 6px;
    text-transform: uppercase;
    margin-bottom: 24px;
}

/* ── Metric cards ── */
.metric-card {
    background: #0d1117;
    border: 1px solid #1a2535;
    border-left: 3px solid #00e5ff;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 12px;
}
.metric-label {
    font-size: 0.7rem;
    letter-spacing: 3px;
    color: #4a6274;
    text-transform: uppercase;
}
.metric-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.8rem;
    color: #00e5ff;
    line-height: 1.2;
}

/* ── Alert box ── */
.alert-box {
    background: #120a0a;
    border: 1px solid #ff3b3b55;
    border-left: 3px solid #ff3b3b;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 6px 0;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #ff6b6b;
}

/* ── Upload zone ── */
[data-testid="stFileUploader"] {
    background: #0d1117;
    border: 1px dashed #1e3a4a !important;
    border-radius: 8px;
    padding: 8px;
}
[data-testid="stFileUploader"] label {
    color: #4a8fa8 !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-family: 'Rajdhani', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    letter-spacing: 2px;
    color: #4a6274 !important;
    text-transform: uppercase;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00e5ff !important;
    border-bottom: 2px solid #00e5ff !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent;
    border: 1px solid #00e5ff;
    color: #00e5ff;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-radius: 4px;
    padding: 8px 24px;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #00e5ff15;
    border-color: #00e5ff;
    color: #fff;
}

/* ── Sliders / selects ── */
[data-testid="stSlider"] label,
[data-testid="stSelectbox"] label { color: #4a6274 !important; }

/* ── Section header ── */
.section-hdr {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 4px;
    color: #00e5ff88;
    text-transform: uppercase;
    border-bottom: 1px solid #1a2535;
    padding-bottom: 6px;
    margin: 20px 0 12px;
}

/* ── Status bar ── */
.status-bar {
    background: #0d1117;
    border: 1px solid #1a2535;
    border-radius: 4px;
    padding: 8px 14px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.75rem;
    color: #4a6274;
    margin-bottom: 12px;
}
.status-bar .ok  { color: #00e5a0; }
.status-bar .bad { color: #ff6b6b; }
</style>
""", unsafe_allow_html=True)


# ── Model loader (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return YOLO("models/yolo26n.pt")

PERSON_CLASS = 0


# ── Helpers ───────────────────────────────────────────────────────────────────
COLORS = [(0, 229, 255), (0, 255, 160), (255, 165, 0), (200, 100, 255)]

def draw_box(frame, x1, y1, x2, y2, label, color=(0, 229, 255)):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 12, 16), 2)

def bgr_to_rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sv-title">⬡ SECUREVISION</div>', unsafe_allow_html=True)
    st.markdown('<div class="sv-sub">Person Tracker v1.0</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">Detection Config</div>', unsafe_allow_html=True)
    conf_thresh = st.slider("Confidence Threshold", 0.1, 0.95, 0.40, 0.05)
    iou_thresh  = st.slider("IoU Threshold (NMS)", 0.1, 0.9, 0.50, 0.05)

    st.markdown('<div class="section-hdr">Video Options</div>', unsafe_allow_html=True)
    max_frames  = st.slider("Max frames to process", 50, 500, 200, 50)
    show_every  = st.slider("Preview every N frames", 1, 30, 5)

    st.markdown('<div class="section-hdr">Model</div>', unsafe_allow_html=True)
    st.markdown('<div class="status-bar">YOLOv8n · COCO · ByteTrack<br>'
                '<span class="ok">● LOADED</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-hdr">About</div>', unsafe_allow_html=True)
    st.caption("Detects & tracks persons in images and video using YOLOv8 + ByteTrack. "
               "Fires alerts for each unique individual detected.")


# ── Main header ───────────────────────────────────────────────────────────────
st.markdown('<div class="sv-title">⬡ SECUREVISION</div>', unsafe_allow_html=True)
st.markdown('<div class="sv-sub">AI-Powered Person Detection & Tracking</div>',
            unsafe_allow_html=True)

model = load_model()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_img, tab_vid = st.tabs(["📷  IMAGE DETECTION", "🎬  VIDEO TRACKING"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — IMAGE
# ══════════════════════════════════════════════════════════════════════════════
with tab_img:
    col_up, col_out = st.columns([1, 1], gap="large")

    with col_up:
        st.markdown('<div class="section-hdr">Upload Image</div>',
                    unsafe_allow_html=True)
        img_file = st.file_uploader(
            "Drop an image here", type=["jpg", "jpeg", "png", "bmp", "webp", "avif"],
            key="img_upload", label_visibility="collapsed")

        if img_file:
            pil_img = Image.open(img_file).convert("RGB")
            st.image(pil_img, caption="Original", use_container_width=True)

            run_btn = st.button("🎯  RUN DETECTION", key="run_img")
        else:
            st.info("Upload a JPG / PNG image to start detection.")
            run_btn = False

    with col_out:
        st.markdown('<div class="section-hdr">Detection Output</div>',
                    unsafe_allow_html=True)

        if img_file and run_btn:
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            t0 = time.time()
            results = model(frame, conf=conf_thresh, iou=iou_thresh,
                            classes=[PERSON_CLASS], verbose=False)
            inf_ms  = (time.time() - t0) * 1000

            boxes   = results[0].boxes
            n_persons = 0
            alerts = []

            if boxes is not None and len(boxes):
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    color = COLORS[i % len(COLORS)]
                    label = f"Person  {conf:.2f}"
                    draw_box(frame, x1, y1, x2, y2, label, color)
                    n_persons += 1
                    alerts.append(
                        f"ALERT: Person #{i+1} detected — {conf*100:.1f}% confidence")

            out_rgb = bgr_to_rgb(frame)
            st.image(out_rgb, caption="Detected", use_container_width=True)

            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="metric-card"><div class="metric-label">Persons</div>'
                        f'<div class="metric-value">{n_persons}</div></div>',
                        unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-card"><div class="metric-label">Inference</div>'
                        f'<div class="metric-value">{inf_ms:.0f}ms</div></div>',
                        unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-card"><div class="metric-label">Threshold</div>'
                        f'<div class="metric-value">{conf_thresh}</div></div>',
                        unsafe_allow_html=True)

            # Alerts
            if alerts:
                st.markdown('<div class="section-hdr">Alerts</div>',
                            unsafe_allow_html=True)
                for a in alerts:
                    st.markdown(f'<div class="alert-box">⚠ {a}</div>',
                                unsafe_allow_html=True)
            else:
                st.success("✅ No persons detected.")

            # Download
            _, buf = cv2.imencode(".jpg", frame)
            st.download_button("⬇ Download Result", buf.tobytes(),
                               file_name="detection_result.jpg",
                               mime="image/jpeg")

        elif not img_file:
            st.markdown('<div class="status-bar">Waiting for image upload…</div>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VIDEO
# ══════════════════════════════════════════════════════════════════════════════
with tab_vid:
    col_vup, col_vout = st.columns([1, 1], gap="large")

    with col_vup:
        st.markdown('<div class="section-hdr">Upload Video</div>',
                    unsafe_allow_html=True)
        vid_file = st.file_uploader(
            "Drop a video here", type=["mp4", "avi", "mov", "mkv", "webm"],
            key="vid_upload", label_visibility="collapsed")

        if vid_file:
            st.video(vid_file)
            run_vid_btn = st.button("🎯  RUN TRACKING", key="run_vid")
        else:
            st.info("Upload a video file (MP4 / AVI / MOV) to start tracking.")
            run_vid_btn = False

    with col_vout:
        st.markdown('<div class="section-hdr">Tracking Output</div>',
                    unsafe_allow_html=True)

        if vid_file and run_vid_btn:
            # Save upload to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(vid_file.read())
            tfile.flush()

            cap = cv2.VideoCapture(tfile.name)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_src = cap.get(cv2.CAP_PROP_FPS) or 25

            # Output video writer
            out_tmp  = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            out_path = out_tmp.name
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps_src, (w, h))

            # Progress UI
            progress_bar  = st.progress(0, text="Processing…")
            preview_slot  = st.empty()
            status_slot   = st.empty()

            unique_ids: set = set()
            alerts: list   = []
            frame_idx = 0
            process_up_to = min(max_frames, total_frames or max_frames)
            total_inf_ms  = 0.0

            while True:
                ret, frame = cap.read()
                if not ret or frame_idx >= process_up_to:
                    break

                t0 = time.time()
                results = model.track(frame, persist=True,
                                      classes=[PERSON_CLASS],
                                      conf=conf_thresh,
                                      iou=iou_thresh,
                                      verbose=False)
                total_inf_ms += (time.time() - t0) * 1000

                boxes = results[0].boxes
                if boxes is not None and len(boxes):
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        tid  = int(box.id[0]) if box.id is not None else -1
                        color = COLORS[tid % len(COLORS)] if tid >= 0 else COLORS[0]
                        draw_box(frame, x1, y1, x2, y2,
                                 f"#{tid}  {conf:.2f}", color)

                        if tid not in unique_ids:
                            unique_ids.add(tid)
                            msg = (f"ALERT: Person #{tid} detected — "
                                   f"{conf*100:.1f}% conf  [frame {frame_idx}]")
                            alerts.append(msg)

                # Overlay frame counter
                cv2.putText(frame, f"FRAME {frame_idx}", (10, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 1)

                writer.write(frame)

                # Live preview every N frames
                if frame_idx % show_every == 0:
                    preview_slot.image(bgr_to_rgb(frame),
                                       caption=f"Frame {frame_idx}",
                                       use_container_width=True)

                pct = int((frame_idx + 1) / process_up_to * 100)
                progress_bar.progress(pct,
                    text=f"Processing frame {frame_idx+1}/{process_up_to}…")
                frame_idx += 1

            cap.release()
            writer.release()
            progress_bar.progress(100, text="✅ Done!")

            avg_fps = frame_idx / (total_inf_ms / 1000 + 1e-9)

            # Metrics
            status_slot.empty()
            m1, m2, m3 = st.columns(3)
            m1.markdown(
                f'<div class="metric-card"><div class="metric-label">Unique Persons</div>'
                f'<div class="metric-value">{len(unique_ids)}</div></div>',
                unsafe_allow_html=True)
            m2.markdown(
                f'<div class="metric-card"><div class="metric-label">Frames Processed</div>'
                f'<div class="metric-value">{frame_idx}</div></div>',
                unsafe_allow_html=True)
            m3.markdown(
                f'<div class="metric-card"><div class="metric-label">Avg FPS</div>'
                f'<div class="metric-value">{avg_fps:.1f}</div></div>',
                unsafe_allow_html=True)

            # Alerts
            if alerts:
                st.markdown('<div class="section-hdr">Alerts Log</div>',
                            unsafe_allow_html=True)
                for a in alerts[:20]:   # cap display at 20
                    st.markdown(f'<div class="alert-box">⚠ {a}</div>',
                                unsafe_allow_html=True)
                if len(alerts) > 20:
                    st.caption(f"… and {len(alerts)-20} more alerts")

            # Download processed video
            with open(out_path, "rb") as vf:
                st.download_button("⬇ Download Tracked Video", vf.read(),
                                   file_name="tracked_output.mp4",
                                   mime="video/mp4")

        elif not vid_file:
            st.markdown('<div class="status-bar">Waiting for video upload…</div>',
                        unsafe_allow_html=True)
