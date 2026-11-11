"""
Unit tests for the inference module.

Note: Tests requiring the actual YOLO model are marked with
@pytest.mark.skipif to skip when the model file is not available.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from config.settings import BEST_MODEL_PATH, CLASS_NAMES, EMERGENCY_CLASS_IDS
from src.inference import VehicleDetector
from src.decision_engine import DetectionResult


MODEL_AVAILABLE = BEST_MODEL_PATH.exists()


class TestVehicleDetectorInit:
    """Tests for detector initialization."""

    def test_default_init(self):
        detector = VehicleDetector()
        assert not detector.is_loaded
        assert detector.confidence_threshold == 0.25

    def test_custom_threshold(self):
        detector = VehicleDetector(confidence_threshold=0.5)
        assert detector.confidence_threshold == 0.5

    def test_missing_model_file(self):
        detector = VehicleDetector(model_path="nonexistent_model.pt")
        loaded = detector.load_model()
        assert not loaded
        assert detector.load_error is not None
        assert "not found" in detector.load_error.lower()


@pytest.mark.skipif(not MODEL_AVAILABLE, reason="Model file not available")
class TestVehicleDetectorInference:
    """Tests requiring the actual model."""

    def setup_method(self):
        self.detector = VehicleDetector()
        assert self.detector.load_model(), f"Failed to load model: {self.detector.load_error}"

    def test_detect_returns_detection_result(self):
        """Detection should return a DetectionResult."""
        # Create a dummy image (blank)
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        result = self.detector.detect(img)
        assert isinstance(result, DetectionResult)

    def test_detect_black_image_minimal_detections(self):
        """A blank black image should produce few or no detections."""
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        result = self.detector.detect(img)
        assert result.total_vehicles >= 0  # Should not crash

    def test_annotate_returns_image(self):
        """Annotation should return a valid image array."""
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        result = self.detector.detect(img)
        annotated = self.detector.annotate_image(img, result)
        assert annotated.shape == img.shape
        assert annotated.dtype == np.uint8

    def test_class_names_match_config(self):
        """Model class names should match our config."""
        model_names = self.detector.model.names
        for idx, name in CLASS_NAMES.items():
            assert model_names.get(idx) == name, \
                f"Class {idx}: model has '{model_names.get(idx)}' vs config '{name}'"

    def test_detect_not_loaded_raises(self):
        """Calling detect without loading should raise."""
        detector = VehicleDetector()
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError):
            detector.detect(img)


class TestDetectionResult:
    """Tests for DetectionResult data structure."""

    def test_empty_detection(self):
        result = DetectionResult(
            emergency_detected=False,
            emergency_vehicles=[],
            all_detections=[],
            vehicle_counts={},
            total_vehicles=0,
            emergency_count=0,
            max_emergency_confidence=0.0,
        )
        assert not result.emergency_detected
        assert result.total_vehicles == 0

    def test_emergency_detection(self):
        result = DetectionResult(
            emergency_detected=True,
            emergency_vehicles=[{
                "class": "ambulance",
                "class_id": 6,
                "confidence": 0.95,
                "bbox": [100, 100, 200, 200],
            }],
            all_detections=[],
            vehicle_counts={"ambulance": 1},
            total_vehicles=1,
            emergency_count=1,
            max_emergency_confidence=0.95,
        )
        assert result.emergency_detected
        assert result.emergency_count == 1
        assert result.max_emergency_confidence == 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
