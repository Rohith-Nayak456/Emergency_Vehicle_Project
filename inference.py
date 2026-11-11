"""Top-level inference interface for Emergency Vehicle Detection.

Callable directly from the Streamlit application or scripts:
    from inference import run_inference
    result = run_inference("path/to/image.jpg", "NORTH")

Conforms strictly to the required Phase 4 interface specification.
"""

import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is on sys.path for direct script execution
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from inference import (
    CLASS_MAPPING,
    EMERGENCY_CLASSES,
    VALID_CAMERA_DIRECTIONS,
    YOLOInferenceEngine,
    get_inference_engine,
    run_inference as _package_run_inference,
)


def run_inference(
    image_path: str,
    camera_direction: str,
    raise_on_error: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Execute inference for a single still image.

    Args:
        image_path: Path to the still image file.
        camera_direction: Camera direction ('NORTH', 'SOUTH', 'EAST', 'WEST').
        raise_on_error: If True, raises exceptions. If False (default), returns
                        a structured error dictionary with status="error".
        **kwargs: Optional configuration parameters passed to inference engine.

    Returns:
        Exact required dictionary structure:
        {
            "camera": "NORTH",
            "emergency_detected": True,
            "emergency_type": "ambulance",
            "confidence": 0.94,
            "detections": [
                {
                    "class_id": 6,
                    "class_name": "ambulance",
                    "confidence": 0.94,
                    "bbox": [100, 120, 350, 280],
                    "is_emergency": True
                }
            ],
            "annotated_image_path": "path/to/annotated.jpg",
            "status": "success"
        }
    """
    return _package_run_inference(
        image_path=image_path,
        camera_direction=camera_direction,
        raise_on_error=raise_on_error,
        **kwargs,
    )


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run Emergency Vehicle YOLO inference on a single image."
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--camera",
        required=True,
        choices=["NORTH", "SOUTH", "EAST", "WEST"],
        help="Camera direction",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional custom model weights path (default: models/best.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    args = parser.parse_args()

    result = run_inference(
        image_path=args.image,
        camera_direction=args.camera,
        model_path=args.model,
        conf_threshold=args.conf,
    )
    print(json.dumps(result, indent=2))
