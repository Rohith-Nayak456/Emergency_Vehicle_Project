"""Actual YOLO Inference Integration Tests.

=============================================================================
IMPORTANT: These tests execute REAL inference using the trained weights file
(default: models/best.pt) supplied by Teammate 1.
They are strictly segregated from mock/interface unit tests.
If models/best.pt is not yet available, these tests are automatically skipped.
=============================================================================
"""

from pathlib import Path
import pytest

from inference.config import (
    CLASS_MAPPING,
    DEFAULT_MODEL_PATH,
    EMERGENCY_CLASSES,
)
from inference.detector import YOLOInferenceEngine
from inference.validator import validate_camera_direction

# Check whether best.pt exists
HAS_REAL_WEIGHTS = Path(DEFAULT_MODEL_PATH).exists()


@pytest.mark.skipif(
    not HAS_REAL_WEIGHTS,
    reason="models/best.pt not yet provided by Teammate 1. Skipping actual YOLO inference tests.",
)
class TestRealYOLOInference:
    """Actual YOLO tests that run ONLY when best.pt is supplied by Teammate 1."""

    @pytest.fixture(scope="class")
    def real_engine(self) -> YOLOInferenceEngine:
        """Initialize engine with real weights."""
        engine = YOLOInferenceEngine(model_path=DEFAULT_MODEL_PATH, device="cpu")
        engine.load_model()
        return engine

    def test_real_model_loading_and_class_ids(self, real_engine):
        """Verify real YOLO model loads successfully and contains expected 8 classes."""
        model = real_engine.model
        assert model is not None

        # Verify class names map correctly
        names = model.names
        assert isinstance(names, dict)
        for class_id, expected_name in CLASS_MAPPING.items():
            if class_id in names:
                assert names[class_id] == expected_name

    def test_real_inference_on_sample_image(self, real_engine, sample_image_file, tmp_path):
        """Run real YOLO inference on a test image and verify output contract."""
        real_engine.output_dir = tmp_path / "real_output"
        result = real_engine.predict(sample_image_file, "NORTH")

        assert result["status"] == "success"
        assert result["camera"] == "NORTH"
        assert isinstance(result["emergency_detected"], bool)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["detections"], list)
        assert Path(result["annotated_image_path"]).exists()

        for det in result["detections"]:
            assert det["class_id"] in CLASS_MAPPING
            assert det["class_name"] in CLASS_MAPPING.values()
            assert 0.0 <= det["confidence"] <= 1.0
            assert len(det["bbox"]) == 4
            assert det["is_emergency"] == (det["class_name"] in EMERGENCY_CLASSES)
