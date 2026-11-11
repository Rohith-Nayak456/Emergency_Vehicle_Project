"""Core YOLO inference engine for emergency vehicle detection.

Provides model loading with caching, CPU support, detection extraction,
emergency vehicle filtering, and annotated image generation.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from inference.annotator import annotate_image
from inference.config import (
    CLASS_MAPPING,
    DEFAULT_CONF_THRESHOLD,
    DEFAULT_DEVICE,
    DEFAULT_IOU_THRESHOLD,
    DEFAULT_MODEL_PATH,
    DEFAULT_OUTPUT_DIR,
    EMERGENCY_CLASSES,
    EMERGENCY_CLASS_IDS,
)
from inference.exceptions import (
    InferenceExecutionError,
    ModelLoadError,
    ModelNotFoundError,
)
from inference.validator import validate_camera_direction, validate_image_file


class YOLOInferenceEngine:
    """Manages YOLO model lifecycle and runs single-image inference."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = DEFAULT_CONF_THRESHOLD,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        device: str = DEFAULT_DEVICE,
        output_dir: Optional[Union[str, Path]] = None,
        model: Optional[Any] = None,
    ) -> None:
        """Initialize the YOLO inference engine.

        Args:
            model_path: Path to weights file (default: models/best.pt).
            conf_threshold: Confidence threshold for predictions (default: 0.25).
            iou_threshold: NMS IoU threshold (default: 0.45).
            device: Inference compute device (default: 'cpu').
            output_dir: Directory where annotated images will be saved.
            model: Optional pre-loaded model or mock model instance.
        """
        env_path = os.environ.get("YOLO_MODEL_PATH")
        if model_path is not None:
            self.model_path = Path(model_path).resolve()
        elif env_path:
            self.model_path = Path(env_path).resolve()
        else:
            self.model_path = Path(DEFAULT_MODEL_PATH).resolve()

        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.device = device
        self.output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR).resolve()

        # Cached model instance
        self.model: Optional[Any] = model

    def load_model(self, force_reload: bool = False) -> Any:
        """Load YOLO model weights into memory with caching.

        Args:
            force_reload: If True, reloads model even if cached.

        Returns:
            The loaded YOLO model instance.

        Raises:
            ModelNotFoundError: If the weights file does not exist.
            ModelLoadError: If loading the weights into memory fails.
        """
        if self.model is not None and not force_reload:
            return self.model

        if not self.model_path.exists():
            raise ModelNotFoundError(
                f"Model weights file not found at '{self.model_path}'. "
                f"Ensure Teammate 1 has provided best.pt or specify a valid path."
            )

        try:
            from ultralytics import YOLO  # lazy import
            model = YOLO(str(self.model_path))
            self.model = model
            return self.model
        except Exception as exc:
            raise ModelLoadError(
                f"Failed to load YOLO model from '{self.model_path}': {exc}"
            ) from exc

    def process_results(self, raw_results: Any) -> List[Dict[str, Any]]:
        """Parse raw YOLO prediction results into standardized detection dictionaries.

        Args:
            raw_results: YOLO output results (typically a list of Results objects).

        Returns:
            List of detection dicts:
            [
                {
                    "class_id": int,
                    "class_name": str,
                    "confidence": float,
                    "bbox": [x1, y1, x2, y2],
                    "is_emergency": bool
                }, ...
            ]
        """
        detections: List[Dict[str, Any]] = []

        if not raw_results:
            return detections

        first_result = raw_results[0] if isinstance(raw_results, (list, tuple)) else raw_results

        boxes = getattr(first_result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        # Extract model class names mapping if present
        model_names = getattr(self.model, "names", {}) or getattr(first_result, "names", {})

        for box in boxes:
            try:
                # Bounding box coords [x1, y1, x2, y2]
                xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(box.xyxy[0])
                bbox = [int(round(float(c))) for c in xyxy]

                # Confidence
                if hasattr(box.conf[0], "item"):
                    conf = float(box.conf[0].item())
                else:
                    conf = float(box.conf[0])
                conf = round(conf, 4)

                # Class id & name
                if hasattr(box.cls[0], "item"):
                    class_id = int(box.cls[0].item())
                else:
                    class_id = int(box.cls[0])

                class_name = model_names.get(class_id, CLASS_MAPPING.get(class_id, f"class_{class_id}"))

                # Emergency classification
                is_emergency = (
                    class_name in EMERGENCY_CLASSES
                    or class_id in EMERGENCY_CLASS_IDS
                )

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox": bbox,
                    "is_emergency": is_emergency,
                })
            except Exception as exc:
                # Log or skip individual corrupt box rather than crashing entire batch
                continue

        return detections

    @staticmethod
    def evaluate_emergency(detections: List[Dict[str, Any]]) -> Tuple[bool, Optional[str], float]:
        """Determine top emergency detection from detection list.

        If multiple emergency vehicles exist, selects the highest-confidence
        emergency detection for summary fields.

        Args:
            detections: List of detection dictionaries.

        Returns:
            Tuple of (emergency_detected, emergency_type, confidence).
        """
        emergency_detections = [d for d in detections if d.get("is_emergency", False)]

        if not emergency_detections:
            return False, None, 0.0

        # Select highest-confidence emergency detection
        top_emergency = max(emergency_detections, key=lambda d: d.get("confidence", 0.0))
        return (
            True,
            top_emergency.get("class_name"),
            top_emergency.get("confidence", 0.0),
        )

    def predict(
        self,
        image_path: Union[str, Path],
        camera_direction: str,
    ) -> Dict[str, Any]:
        """Execute inference on a single image and return standardized dictionary.

        Args:
            image_path: Path to the input image file.
            camera_direction: Camera direction (NORTH, SOUTH, EAST, WEST).

        Returns:
            Standardized result dictionary matching required interface.

        Raises:
            InvalidCameraDirectionError: If camera direction is unsupported.
            InvalidImageError: If image is missing, corrupt, or invalid.
            ModelNotFoundError: If weights are missing.
            ModelLoadError: If loading model fails.
            InferenceExecutionError: If YOLO prediction fails.
        """
        # Validate inputs
        validated_direction = validate_camera_direction(camera_direction)
        validated_image = validate_image_file(image_path)

        # Ensure model is ready
        model = self.load_model()

        # Run inference
        try:
            raw_results = model.predict(
                source=str(validated_image),
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            raise InferenceExecutionError(
                f"YOLO prediction failed on '{validated_image.name}': {exc}"
            ) from exc

        # Process detections
        detections = self.process_results(raw_results)

        # Evaluate emergency status
        emergency_detected, emergency_type, top_conf = self.evaluate_emergency(detections)

        # Generate annotated image
        annotated_image_path = annotate_image(
            image_path=validated_image,
            detections=detections,
            output_dir=self.output_dir,
        )

        return {
            "camera": validated_direction,
            "emergency_detected": emergency_detected,
            "emergency_type": emergency_type,
            "confidence": top_conf,
            "detections": detections,
            "annotated_image_path": annotated_image_path,
            "status": "success",
        }


# Module-level singleton engine for reuse
_DEFAULT_ENGINE: Optional[YOLOInferenceEngine] = None


def get_inference_engine(
    model_path: Optional[Union[str, Path]] = None,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    device: str = DEFAULT_DEVICE,
    output_dir: Optional[Union[str, Path]] = None,
    force_new: bool = False,
) -> YOLOInferenceEngine:
    """Retrieve or create the shared YOLOInferenceEngine singleton.

    Avoids reloading model into memory repeatedly.
    """
    global _DEFAULT_ENGINE
    if _DEFAULT_ENGINE is None or force_new:
        _DEFAULT_ENGINE = YOLOInferenceEngine(
            model_path=model_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            device=device,
            output_dir=output_dir,
        )
    return _DEFAULT_ENGINE


def reset_inference_engine() -> None:
    """Reset the cached engine singleton (useful during tests)."""
    global _DEFAULT_ENGINE
    _DEFAULT_ENGINE = None
