"""
Emergency Vehicle Detection & Smart Signal Preemption
Streamlit Frontend Application

A professional traffic-management dashboard for detecting emergency vehicles
in traffic images and simulating adaptive signal preemption at a four-way
intersection.

IMPORTANT: This is a research prototype and simulation.
It does NOT control real traffic signals.
"""
import sys
import time
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    EMERGENCY_CLASS_IDS,
    EMERGENCY_CLASS_NAMES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    EMERGENCY_CONFIDENCE_THRESHOLD,
    DIRECTIONS,
    NORMAL_GREEN_DURATION,
    NORMAL_YELLOW_DURATION,
    ALL_RED_CLEARANCE_DURATION,
    EMERGENCY_GREEN_DURATION,
)
from src.inference import VehicleDetector
from src.decision_engine import DecisionEngine, DetectionResult
from src.signal_controller import (
    SignalController,
    SignalState,
    IntersectionPhase,
)


# ═══════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Emergency Vehicle Detection & Smart Signal Preemption",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
        text-align: center;
    }
    .main-header h1 {
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
    }
    .main-header p {
        font-size: 1rem;
        opacity: 0.85;
        margin-top: 0.5rem;
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.2rem;
    }
    .status-online { background: #27ae60; color: white; }
    .status-offline { background: #e74c3c; color: white; }
    .status-warning { background: #f39c12; color: white; }

    /* Signal lights */
    .signal-container {
        text-align: center;
        padding: 1rem;
        background: #1a1a2e;
        border-radius: 12px;
        margin: 0.5rem;
    }
    .signal-light {
        width: 50px;
        height: 50px;
        border-radius: 50%;
        margin: 8px auto;
        border: 3px solid #333;
        transition: all 0.3s ease;
    }
    .signal-red-on { background: #ff0000; box-shadow: 0 0 20px #ff0000; }
    .signal-red-off { background: #4a0000; }
    .signal-yellow-on { background: #ffcc00; box-shadow: 0 0 20px #ffcc00; }
    .signal-yellow-off { background: #4a4a00; }
    .signal-green-on { background: #00ff00; box-shadow: 0 0 20px #00ff00; }
    .signal-green-off { background: #004a00; }

    .direction-label {
        color: white;
        font-weight: 700;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }

    /* Emergency alert */
    .emergency-alert {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.2rem;
        font-weight: 700;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }

    .normal-status {
        background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1rem 0;
    }

    .uncertain-status {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1rem 0;
    }

    /* Vehicle count cards */
    .vehicle-card {
        background: #f8f9fa;
        border-left: 4px solid #3498db;
        padding: 0.75rem 1rem;
        margin: 0.3rem 0;
        border-radius: 0 8px 8px 0;
    }
    .vehicle-card-emergency {
        background: #fdf2f2;
        border-left: 4px solid #e74c3c;
    }

    /* Phase indicator */
    .phase-step {
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        border-left: 4px solid #bdc3c7;
    }
    .phase-active {
        background: #eafaf1;
        border-left-color: #27ae60;
        font-weight: 600;
    }
    .phase-emergency {
        background: #fdf2f2;
        border-left-color: #e74c3c;
        font-weight: 600;
    }

    /* Hide default streamlit elements for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SESSION STATE INITIALIZATION
# ═══════════════════════════════════════════════════════════════════
def init_session_state():
    defaults = {
        "detector": None,
        "model_loaded": False,
        "model_error": None,
        "uploaded_image": None,
        "original_image_np": None,
        "detection_result": None,
        "annotated_image": None,
        "decision": None,
        "analysis_done": False,
        "current_phase_index": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ═══════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_detector():
    """Load the YOLO model (cached across reruns)."""
    detector = VehicleDetector()
    success = detector.load_model()
    return detector, success


# ═══════════════════════════════════════════════════════════════════
# SIGNAL RENDERING
# ═══════════════════════════════════════════════════════════════════
def render_signal_light(direction: str, state: SignalState, is_emergency: bool = False):
    """Render a single traffic signal with three lights."""
    red_class = "signal-red-on" if state == SignalState.RED else "signal-red-off"
    yellow_class = "signal-yellow-on" if state == SignalState.YELLOW else "signal-yellow-off"
    green_class = "signal-green-on" if state == SignalState.GREEN else "signal-green-off"

    emoji = "🚨" if is_emergency and state == SignalState.GREEN else ""

    html = f"""
    <div class="signal-container">
        <div class="direction-label">{emoji} {direction} {emoji}</div>
        <div class="signal-light {red_class}"></div>
        <div class="signal-light {yellow_class}"></div>
        <div class="signal-light {green_class}"></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_intersection_signals(snapshot, emergency_direction=None):
    """Render the four-way intersection signal dashboard."""
    # Top: North
    col_spacer1, col_north, col_spacer2 = st.columns([1, 1, 1])
    with col_north:
        is_emg = (emergency_direction == "North")
        render_signal_light("North", snapshot.north, is_emg)

    # Middle: West and East
    col_west, col_center, col_east = st.columns([1, 1, 1])
    with col_west:
        is_emg = (emergency_direction == "West")
        render_signal_light("West", snapshot.west, is_emg)
    with col_center:
        st.markdown("""
        <div style="text-align:center; padding:2rem;">
            <div style="font-size:2rem;">🚦</div>
            <div style="color:#666; font-size:0.8rem;">INTERSECTION</div>
        </div>
        """, unsafe_allow_html=True)
    with col_east:
        is_emg = (emergency_direction == "East")
        render_signal_light("East", snapshot.east, is_emg)

    # Bottom: South
    col_spacer3, col_south, col_spacer4 = st.columns([1, 1, 1])
    with col_south:
        is_emg = (emergency_direction == "South")
        render_signal_light("South", snapshot.south, is_emg)


# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🚨 Emergency Vehicle Detection & Smart Signal Preemption</h1>
    <p>AI-powered traffic management simulation for heterogeneous traffic conditions</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# SIDEBAR: SYSTEM STATUS & CONTROLS
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ System Controls")

    # Model status
    st.markdown("#### 📦 Model Status")
    detector, model_ok = load_detector()
    st.session_state.detector = detector
    st.session_state.model_loaded = model_ok

    if model_ok:
        st.markdown('<span class="status-badge status-online">✓ Model Loaded</span>',
                     unsafe_allow_html=True)
        st.caption(f"Model: {BEST_MODEL_PATH.name}")
    else:
        st.markdown('<span class="status-badge status-offline">✗ Model Not Available</span>',
                     unsafe_allow_html=True)
        error_msg = detector.load_error if detector else "Unknown error"
        st.error(f"Error: {error_msg}")

    st.divider()

    # Approach direction
    st.markdown("#### 🧭 Camera Approach Direction")
    st.caption(
        "Select the known camera position at the intersection. "
        "This represents where the camera is located, NOT a direction "
        "inferred from the image."
    )
    approach_direction = st.selectbox(
        "Approach Direction",
        options=DIRECTIONS,
        index=0,
        label_visibility="collapsed",
    )

    # Current green direction (for simulation)
    st.markdown("#### 🚦 Current Green Direction")
    st.caption("Which direction currently has the green signal (for simulation).")
    current_green = st.selectbox(
        "Current Green",
        options=DIRECTIONS,
        index=2,  # Default: East
        label_visibility="collapsed",
    )

    st.divider()

    # Confidence threshold
    st.markdown("#### 🎯 Detection Settings")
    conf_threshold = st.slider(
        "Detection Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=DEFAULT_CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Minimum confidence to show a detection",
    )

    emg_conf_threshold = st.slider(
        "Emergency Preemption Threshold",
        min_value=0.20,
        max_value=0.90,
        value=EMERGENCY_CONFIDENCE_THRESHOLD,
        step=0.05,
        help="Minimum confidence for automatic emergency preemption",
    )

    st.divider()

    # Reset button
    if st.button("🔄 Reset Application", use_container_width=True):
        for key in ["uploaded_image", "original_image_np", "detection_result",
                     "annotated_image", "decision", "analysis_done", "current_phase_index"]:
            st.session_state[key] = None if key != "analysis_done" else False
            if key == "current_phase_index":
                st.session_state[key] = 0
        st.rerun()

    st.divider()
    st.markdown("#### ℹ️ About")
    st.caption(
        "This system detects vehicles in traffic images using a YOLO model "
        "fine-tuned on 8 vehicle classes including ambulances and police cars. "
        "When an emergency vehicle is detected, it simulates adaptive signal "
        "preemption at a four-way intersection.\n\n"
        "**This is a research prototype. It does NOT control real traffic signals.**"
    )


# ═══════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════

# ── Section A: Image Upload ─────────────────────────────────────
st.markdown("## 📸 Upload Traffic Image")

upload_col, preview_col = st.columns([1, 1])

with upload_col:
    uploaded_file = st.file_uploader(
        "Upload a traffic intersection image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG",
        key="file_uploader",
    )

    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file)
            # Convert to RGB if needed
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")

            # Convert to numpy (BGR for OpenCV)
            img_np = np.array(pil_image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            st.session_state.original_image_np = img_bgr
            st.session_state.uploaded_image = pil_image

            st.success(f"✓ Image loaded: {uploaded_file.name} ({pil_image.size[0]}×{pil_image.size[1]})")
        except Exception as e:
            st.error(f"Failed to load image: {e}")

with preview_col:
    if st.session_state.uploaded_image is not None:
        st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_container_width=True)

# ── Analyze Button ──────────────────────────────────────────────
st.divider()

analyze_disabled = (
    st.session_state.original_image_np is None
    or not st.session_state.model_loaded
)

analyze_btn = st.button(
    "🔍 Analyze Image",
    type="primary",
    use_container_width=True,
    disabled=analyze_disabled,
)

if analyze_disabled and not st.session_state.model_loaded:
    st.warning("⚠️ Model not loaded. Cannot perform analysis.")
elif analyze_disabled:
    st.info("📤 Please upload an image to begin analysis.")

# ── Run Analysis ────────────────────────────────────────────────
if analyze_btn and st.session_state.original_image_np is not None:
    with st.spinner("🔍 Running vehicle detection..."):
        detector = st.session_state.detector

        # Run detection
        detection = detector.detect(
            st.session_state.original_image_np,
            confidence_threshold=conf_threshold,
        )
        st.session_state.detection_result = detection

        # Annotate image
        annotated = detector.annotate_image(
            st.session_state.original_image_np,
            detection,
        )
        # Convert BGR to RGB for display
        st.session_state.annotated_image = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        # Run decision engine
        engine = DecisionEngine(
            emergency_confidence_threshold=emg_conf_threshold,
        )
        decision = engine.evaluate(
            detection=detection,
            approach_direction=approach_direction,
            current_green_direction=current_green,
        )
        st.session_state.decision = decision
        st.session_state.analysis_done = True
        st.session_state.current_phase_index = 0

    st.rerun()

# ═══════════════════════════════════════════════════════════════════
# RESULTS SECTION
# ═══════════════════════════════════════════════════════════════════
if st.session_state.analysis_done and st.session_state.detection_result is not None:
    detection = st.session_state.detection_result
    decision = st.session_state.decision

    st.markdown("---")
    st.markdown("## 📊 Detection Results")

    # ── Annotated Image ─────────────────────────────────────────
    img_col, stats_col = st.columns([3, 2])

    with img_col:
        if st.session_state.annotated_image is not None:
            st.image(
                st.session_state.annotated_image,
                caption="Annotated Detection Results",
                use_container_width=True,
            )

    with stats_col:
        # Summary metrics
        st.markdown("### 📈 Summary")

        m1, m2, m3 = st.columns(3)
        m1.metric("Total Vehicles", detection.total_vehicles)
        m2.metric("Emergency", detection.emergency_count)
        m3.metric(
            "Max Emg. Conf.",
            f"{detection.max_emergency_confidence*100:.1f}%"
            if detection.emergency_count > 0 else "N/A"
        )

        # Vehicle counts by class
        st.markdown("### 🚗 Vehicle Counts")
        for class_name, count in sorted(
            detection.vehicle_counts.items(), key=lambda x: -x[1]
        ):
            if count > 0:
                is_emergency = class_name in EMERGENCY_CLASS_NAMES
                card_class = "vehicle-card-emergency" if is_emergency else ""
                icon = "🚨" if is_emergency else "🚗"
                st.markdown(
                    f'<div class="vehicle-card {card_class}">'
                    f'{icon} <strong>{class_name}</strong>: {count}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Emergency Status ────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🚨 Emergency Status")

    if decision is not None:
        if decision.should_preempt:
            st.markdown(
                f'<div class="emergency-alert">'
                f'🚨 EMERGENCY VEHICLE DETECTED — {decision.emergency_type.upper()} '
                f'({decision.emergency_confidence*100:.1f}% confidence) — '
                f'PREEMPTION RECOMMENDED FOR {decision.approach_direction.upper()}'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif decision.is_uncertain:
            st.markdown(
                f'<div class="uncertain-status">'
                f'⚠️ UNCERTAIN DETECTION — Manual verification required'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="normal-status">'
                '✅ No Emergency Vehicle Detected — Normal Signal Operation'
                '</div>',
                unsafe_allow_html=True,
            )

        # Decision explanation
        st.markdown("### 📋 Decision Explanation")
        st.info(decision.explanation)

    # ── Signal Dashboard ────────────────────────────────────────
    st.markdown("---")
    st.markdown("## 🚦 Four-Way Intersection Signal Dashboard")

    if decision is not None and decision.signal_sequence:
        # Phase navigation
        num_phases = len(decision.signal_sequence)

        if decision.should_preempt:
            st.markdown(
                f"**Signal transition sequence** ({num_phases} phases). "
                f"Use the controls below to step through the phases."
            )

            # Phase stepper
            phase_col1, phase_col2, phase_col3 = st.columns([1, 3, 1])

            with phase_col1:
                if st.button("⏮ Previous", disabled=st.session_state.current_phase_index <= 0):
                    st.session_state.current_phase_index -= 1
                    st.rerun()

            with phase_col3:
                if st.button("Next ⏭", disabled=st.session_state.current_phase_index >= num_phases - 1):
                    st.session_state.current_phase_index += 1
                    st.rerun()

            with phase_col2:
                idx = st.session_state.current_phase_index
                idx = max(0, min(idx, num_phases - 1))
                st.markdown(f"**Phase {idx + 1} of {num_phases}**")

            current_snapshot = decision.signal_sequence[idx]
        else:
            # Normal operation — show current state
            current_snapshot = decision.signal_sequence[0]
            idx = 0
            num_phases = 1

        # Render the intersection
        emg_dir = decision.approach_direction if decision.should_preempt else None
        render_intersection_signals(current_snapshot, emg_dir)

        # Phase description
        st.markdown(
            f'<div class="phase-step {"phase-emergency" if decision.should_preempt else "phase-active"}">'
            f'📍 <strong>Phase {idx + 1}:</strong> {current_snapshot.description}'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Phase timeline
        if decision.should_preempt and num_phases > 1:
            st.markdown("### 📋 Phase Transition Timeline")
            for i, snap in enumerate(decision.signal_sequence):
                is_current = (i == idx)
                phase_class = "phase-active" if is_current else "phase-step"
                if snap.phase in (IntersectionPhase.EMERGENCY_GREEN,):
                    phase_class = "phase-emergency" if is_current else "phase-step"
                marker = "▶" if is_current else "○"
                st.markdown(
                    f'<div class="{phase_class}">'
                    f'{marker} <strong>Phase {i+1}</strong> ({snap.duration}s): '
                    f'{snap.description}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        # No sequence — show default normal operation
        controller = SignalController()
        normal_snap = controller.get_normal_cycle_snapshot()
        render_intersection_signals(normal_snap)
        st.markdown(
            '<div class="phase-step phase-active">'
            '📍 Normal signal operation — no emergency preemption.'
            '</div>',
            unsafe_allow_html=True,
        )

    # ── Detection Details Table ─────────────────────────────────
    if detection.all_detections:
        st.markdown("---")
        st.markdown("## 📋 All Detections")

        import pandas as pd
        det_data = []
        for d in detection.all_detections:
            det_data.append({
                "Class": d["class"],
                "Confidence": f"{d['confidence']*100:.1f}%",
                "Emergency": "🚨 Yes" if d["class_id"] in EMERGENCY_CLASS_IDS else "No",
                "BBox (x1,y1,x2,y2)": str(d["bbox"]),
            })
        df = pd.DataFrame(det_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
