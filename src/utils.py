"""
Utility functions for the Emergency Vehicle Detection project.
"""
import hashlib
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image


def validate_image_file(file_path: str) -> tuple[bool, str]:
    """
    Validate that a file is a readable image with supported extension.

    Returns:
        (is_valid, message)
    """
    path = Path(file_path)
    supported = {".jpg", ".jpeg", ".png"}

    if not path.exists():
        return False, f"File not found: {file_path}"

    ext = path.suffix.lower()
    if ext not in supported:
        return False, f"Unsupported format '{ext}'. Supported: {', '.join(supported)}"

    try:
        img = Image.open(file_path)
        img.verify()
        return True, "Valid image"
    except Exception as e:
        return False, f"Cannot open image: {e}"


def load_image_cv2(file_path: str) -> Optional[np.ndarray]:
    """Load an image using OpenCV. Returns None on failure."""
    try:
        img = cv2.imread(file_path)
        if img is None:
            return None
        return img
    except Exception:
        return None


def load_image_pil(file_path: str) -> Optional[Image.Image]:
    """Load an image using PIL. Returns None on failure."""
    try:
        img = Image.open(file_path)
        img.load()  # Force load
        return img
    except Exception:
        return None


def compute_file_md5(file_path: str) -> str:
    """Compute MD5 hash of a file."""
    md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5.update(chunk)
    return md5.hexdigest()


def image_to_bytes(image: np.ndarray, ext: str = ".jpg") -> bytes:
    """Encode a CV2 image to bytes."""
    _, buffer = cv2.imencode(ext, image)
    return buffer.tobytes()


def format_confidence(confidence: float) -> str:
    """Format confidence score as percentage string."""
    return f"{confidence * 100:.1f}%"


def get_class_color_bgr(class_id: int) -> tuple[int, int, int]:
    """
    Return a BGR colour for the given class ID.
    Emergency vehicles get red, others get distinct colours.
    """
    emergency_ids = {6, 7}
    if class_id in emergency_ids:
        return (0, 0, 255)  # Red for emergency

    palette = {
        0: (0, 200, 0),      # car - green
        1: (200, 200, 0),    # motorcycle - cyan
        2: (200, 100, 0),    # autorickshaw - blue-ish
        3: (0, 150, 200),    # truck - orange-ish
        4: (200, 0, 200),    # bus - magenta
        5: (100, 200, 100),  # bicycle - light green
    }
    return palette.get(class_id, (200, 200, 200))
