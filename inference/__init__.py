"""Emergency vehicle inference and prediction package.

Phase 4: Model Inference and Prediction for Heterogeneous Traffic Conditions.
"""

from inference.config import (
    CLASS_MAPPING,
    CLASS_NAME_TO_ID,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_DEVICE,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    EMERGENCY_CLASSES,
    EMERGENCY_CLASS_IDS,
    VALID_CAMERA_DIRECTIONS,
)
from inference.detector import (
    YOLOInferenceEngine,
    get_inference_engine,
    reset_inference_engine,
)
from inference.exceptions import (
    InferenceBaseError,
    InferenceExecutionError,
    InvalidCameraDirectionError,
    InvalidImageError,
    ModelLoadError,
    ModelNotFoundError,
)


def run_inference(
    image_path: str,
    camera_direction: str,
    raise_on_error: bool = False,
    **kwargs,
) -> dict:
    """Execute inference conforming to required Phase 4 interface.

    Args:
        image_path: Path to the input image file.
        camera_direction: Camera direction (NORTH, SOUTH, EAST, WEST).
        raise_on_error: If True, raises exceptions directly. If False (default),
                        returns standardized error dictionary with status="error".
        **kwargs: Optional engine parameters (model_path, conf_threshold, etc.).

    Returns:
        Exact required dictionary structure:
        {
            "camera": "NORTH",
            "emergency_detected": True,
            "emergency_type": "ambulance",
            "confidence": 0.94,
            "detections": [...],
            "annotated_image_path": "path/to/annotated.jpg",
            "status": "success"
        }
    """
    engine = get_inference_engine(**kwargs) if not kwargs else YOLOInferenceEngine(**kwargs)

    try:
        return engine.predict(image_path=image_path, camera_direction=camera_direction)
    except Exception as exc:
        if raise_on_error:
            raise

        # Check if camera direction is valid or string
        cam = str(camera_direction) if camera_direction is not None else "UNKNOWN"
        return {
            "camera": cam,
            "emergency_detected": False,
            "emergency_type": None,
            "confidence": 0.0,
            "detections": [],
            "annotated_image_path": None,
            "status": "error",
            "error_message": str(exc),
        }


__all__ = [
    "run_inference",
    "YOLOInferenceEngine",
    "get_inference_engine",
    "reset_inference_engine",
    "CLASS_MAPPING",
    "CLASS_NAME_TO_ID",
    "EMERGENCY_CLASSES",
    "EMERGENCY_CLASS_IDS",
    "VALID_CAMERA_DIRECTIONS",
    "DEFAULT_MODEL_PATH",
    "DEFAULT_OUTPUT_DIR",
    "DEFAULT_CONF_THRESHOLD",
    "DEFAULT_IOU_THRESHOLD",
    "DEFAULT_DEVICE",
    "InferenceBaseError",
    "ModelNotFoundError",
    "ModelLoadError",
    "InvalidCameraDirectionError",
    "InvalidImageError",
    "InferenceExecutionError",
]
