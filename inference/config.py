"""Configuration constants and defaults for YOLO inference.

Contains class IDs, class names, emergency vehicle definitions,
accepted camera directions, and default thresholds.
"""

from pathlib import Path
from typing import Dict, Final, Set, Tuple

# Base project directory
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent

# Default model path
DEFAULT_MODEL_PATH: Final[Path] = BASE_DIR / "models" / "best.pt"

# Default directory for saving annotated images
DEFAULT_OUTPUT_DIR: Final[Path] = BASE_DIR / "output" / "annotated"

# The 8 vehicle classes defined for the project
CLASS_MAPPING: Final[Dict[int, str]] = {
    0: "car",
    1: "motorcycle",
    2: "autorickshaw",
    3: "truck",
    4: "bus",
    5: "bicycle",
    6: "ambulance",
    7: "police_car",
}

# Inverted mapping: name -> id
CLASS_NAME_TO_ID: Final[Dict[str, int]] = {v: k for k, v in CLASS_MAPPING.items()}

# Emergency class designations
EMERGENCY_CLASSES: Final[Set[str]] = {"ambulance", "police_car"}
EMERGENCY_CLASS_IDS: Final[Set[int]] = {
    class_id
    for class_id, name in CLASS_MAPPING.items()
    if name in EMERGENCY_CLASSES
}

# Permitted camera directions
VALID_CAMERA_DIRECTIONS: Final[Tuple[str, ...]] = (
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
)

# Supported image file extensions
SUPPORTED_IMAGE_EXTENSIONS: Final[Tuple[str, ...]] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)

# Inference defaults
DEFAULT_CONF_THRESHOLD: Final[float] = 0.25
DEFAULT_IOU_THRESHOLD: Final[float] = 0.45
DEFAULT_DEVICE: Final[str] = "cpu"
