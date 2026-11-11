"""
Inference module for Emergency Vehicle Detection.

Loads the trained YOLO model, runs object detection on uploaded images,
draws annotated bounding boxes, and returns structured results.
"""
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import (
    BEST_MODEL_PATH,
    CLASS_NAMES,
    EMERGENCY_CLASS_IDS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    IMAGE_SIZE,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from src.utils import get_class_color_bgr, format_confidence
from src.decision_engine import DetectionResult


class VehicleDetector:
    """
    YOLO-based vehicle detector with emergency vehicle identification.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        image_size: int = IMAGE_SIZE,
    ):
        self.model_path = model_path or str(BEST_MODEL_PATH)
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.model = None
        self._model_loaded = False
        self._load_error = None

    def load_model(self) -> bool:
        """
        Load the YOLO model. Returns True if successful.
        """
        try:
            from ultralytics import YOLO
            model_file = Path(self.model_path)
            if not model_file.exists():
                self._load_error = f"Model file not found: {self.model_path}"
                return False

            self.model = YOLO(str(model_file))
            self._model_loaded = True
            self._load_error = None
            return True
        except ImportError:
            self._load_error = (
                "ultralytics package not installed. "
                "Run: pip install ultralytics"
            )
            return False
        except Exception as e:
            self._load_error = f"Failed to load model: {e}"
            return False

    @property
    def is_loaded(self) -> bool:
        return self._model_loaded

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    def detect(
        self,
        image: np.ndarray,
        confidence_threshold: Optional[float] = None,
    ) -> DetectionResult:
        """
        Run detection on an image.

        Args:
            image: BGR numpy array (OpenCV format).
            confidence_threshold: Override default confidence threshold.

        Returns:
            DetectionResult with all detections and emergency analysis.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._model_loaded or self.model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() first. "
                f"Error: {self._load_error}"
            )

        conf = confidence_threshold or self.confidence_threshold

        # Run inference
        results = self.model.predict(
            source=image,
            conf=conf,
            imgsz=self.image_size,
            verbose=False,
        )

        all_detections = []
        emergency_vehicles = []
        vehicle_counts = {name: 0 for name in CLASS_NAMES.values()}

        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf_score = float(boxes.conf[i].item())
                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    bbox = [int(b) for b in bbox]

                    class_name = CLASS_NAMES.get(cls_id, f"unknown_{cls_id}")

                    det = {
                        "class": class_name,
                        "class_id": cls_id,
                        "confidence": conf_score,
                        "bbox": bbox,  # [x1, y1, x2, y2]
                    }
                    all_detections.append(det)
                    vehicle_counts[class_name] = vehicle_counts.get(class_name, 0) + 1

                    if cls_id in EMERGENCY_CLASS_IDS:
                        emergency_vehicles.append(det)

        emergency_detected = len(emergency_vehicles) > 0
        max_emergency_conf = (
            max(e["confidence"] for e in emergency_vehicles)
            if emergency_vehicles
            else 0.0
        )

        return DetectionResult(
            emergency_detected=emergency_detected,
            emergency_vehicles=emergency_vehicles,
            all_detections=all_detections,
            vehicle_counts=vehicle_counts,
            total_vehicles=len(all_detections),
            emergency_count=len(emergency_vehicles),
            max_emergency_confidence=max_emergency_conf,
        )

    def annotate_image(
        self,
        image: np.ndarray,
        detection: DetectionResult,
        draw_labels: bool = True,
        draw_confidence: bool = True,
        emergency_highlight: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes on the image with class labels and confidence.
        Emergency vehicles are highlighted with distinct styling.

        Args:
            image: Original BGR image.
            detection: DetectionResult from detect().
            draw_labels: Whether to draw class names.
            draw_confidence: Whether to draw confidence scores.
            emergency_highlight: Whether to use special styling for emergency vehicles.

        Returns:
            Annotated image (copy of original).
        """
        annotated = image.copy()

        for det in detection.all_detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_id = det["class_id"]
            class_name = det["class"]
            confidence = det["confidence"]
            is_emergency = cls_id in EMERGENCY_CLASS_IDS

            # Colour and thickness
            if is_emergency and emergency_highlight:
                color = (0, 0, 255)  # Red
                thickness = 3
            else:
                color = get_class_color_bgr(cls_id)
                thickness = 2

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Prepare label
            label_parts = []
            if draw_labels:
                display_name = class_name.upper() if is_emergency else class_name
                label_parts.append(display_name)
            if draw_confidence:
                label_parts.append(format_confidence(confidence))
            label = " ".join(label_parts)

            if label:
                # Label background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6 if is_emergency else 0.5
                label_thickness = 2 if is_emergency else 1
                (tw, th), _ = cv2.getTextSize(
                    label, font, font_scale, label_thickness
                )

                # Position label above bounding box
                label_y = max(y1 - 8, th + 4)
                label_x = x1

                # Background rectangle
                cv2.rectangle(
                    annotated,
                    (label_x, label_y - th - 4),
                    (label_x + tw + 4, label_y + 4),
                    color,
                    -1,  # filled
                )
                # Text
                cv2.putText(
                    annotated,
                    label,
                    (label_x + 2, label_y),
                    font,
                    font_scale,
                    (255, 255, 255),  # White text
                    label_thickness,
                )

            # Emergency badge
            if is_emergency and emergency_highlight:
                badge = f"⚠ EMERGENCY: {class_name.upper()}"
                badge_font_scale = 0.5
                (bw, bh), _ = cv2.getTextSize(
                    badge, cv2.FONT_HERSHEY_SIMPLEX, badge_font_scale, 1
                )
                badge_y = y2 + bh + 10
                cv2.rectangle(
                    annotated,
                    (x1, y2 + 4),
                    (x1 + bw + 8, badge_y + 4),
                    (0, 0, 200),
                    -1,
                )
                cv2.putText(
                    annotated, badge, (x1 + 4, badge_y),
                    cv2.FONT_HERSHEY_SIMPLEX, badge_font_scale,
                    (255, 255, 255), 1,
                )

        return annotated
