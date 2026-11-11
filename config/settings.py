"""
Project-wide settings and constants for Emergency Vehicle Detection
and Adaptive Signal Preemption.
"""
import os
from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SRC_DIR = PROJECT_ROOT / "src"
STATIC_DIR = PROJECT_ROOT / "static"

DATA_YAML = CONFIG_DIR / "data.yaml"
BEST_MODEL_PATH = MODEL_DIR / "best.pt"

DATASET_ROOT = PROJECT_ROOT / "data" / "split_dataset"

# ─── Class Configuration ────────────────────────────────────────────
NUM_CLASSES = 8
CLASS_NAMES = {
    0: "car",
    1: "motorcycle",
    2: "autorickshaw",
    3: "truck",
    4: "bus",
    5: "bicycle",
    6: "ambulance",
    7: "police_car",
}

EMERGENCY_CLASS_IDS = {6, 7}  # ambulance, police_car
EMERGENCY_CLASS_NAMES = {"ambulance", "police_car"}

# ─── Detection Settings ─────────────────────────────────────────────
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
EMERGENCY_CONFIDENCE_THRESHOLD = 0.40  # Higher bar for emergency decisions
IMAGE_SIZE = 640
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ─── Colour Palette (BGR for OpenCV, hex for frontend) ──────────────
# Normal vehicle bounding box colour
NORMAL_VEHICLE_COLOR_BGR = (0, 200, 0)       # Green
NORMAL_VEHICLE_COLOR_HEX = "#00C800"

# Emergency vehicle bounding box colour
EMERGENCY_VEHICLE_COLOR_BGR = (0, 0, 255)    # Red
EMERGENCY_VEHICLE_COLOR_HEX = "#FF0000"

# ─── Signal Controller Settings ─────────────────────────────────────
DIRECTIONS = ["North", "South", "East", "West"]

# Phase durations in seconds (for simulation display)
NORMAL_GREEN_DURATION = 30
NORMAL_YELLOW_DURATION = 5
ALL_RED_CLEARANCE_DURATION = 3
EMERGENCY_GREEN_DURATION = 45
EMERGENCY_YELLOW_DURATION = 5

# ─── Training Settings (defaults) ───────────────────────────────────
TRAIN_EPOCHS = 100
TRAIN_BATCH_SIZE = 16
TRAIN_IMG_SIZE = 640
TRAIN_PATIENCE = 15  # Early stopping patience
RANDOM_SEED = 42
YOLO_MODEL_VARIANT = "yolo11n.pt"  # Lightweight YOLO11 nano
