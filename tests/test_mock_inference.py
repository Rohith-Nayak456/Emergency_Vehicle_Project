"""Interface and Mock-based Unit Tests for Phase 4: Model Inference.

NOTE: These tests use controlled mock predictions to verify the interface contract,
dictionary structure, validation logic, edge cases, and output formatting.
Mock predictions are NOT actual model results.
"""

from pathlib import Path
import pytest

from inference import (
    CLASS_MAPPING,
    EMERGENCY_CLASSES,
    VALID_CAMERA_DIRECTIONS,
    YOLOInferenceEngine,
    run_inference,
)
from inference.exceptions import (
    InvalidCameraDirectionError,
    InvalidImageError,
    ModelNotFoundError,
)


class TestCameraDirectionValidation:
    """Tests 1 & 2: Camera direction validation."""

    @pytest.mark.parametrize("direction", ["NORTH", "SOUTH", "EAST", "WEST", "north", " south "])
    def test_01_valid_camera_direction(self, direction, sample_image_file, mock_normal_vehicles_model, tmp_path):
        """Test 1: All valid camera directions are accepted and normalized."""
        engine = YOLOInferenceEngine(
            model=mock_normal_vehicles_model,
            output_dir=tmp_path / "out",
        )
        result = engine.predict(sample_image_file, direction)
        assert result["camera"] == direction.strip().upper()
        assert result["status"] == "success"

    @pytest.mark.parametrize("invalid_dir", ["UP", "DOWN", "NORTHWEST", "CENTRAL", "", 123, None])
    def test_02_invalid_camera_direction(self, invalid_dir, sample_image_file, mock_normal_vehicles_model, tmp_path):
        """Test 2: Invalid camera directions raise error or return status='error'."""
        engine = YOLOInferenceEngine(
            model=mock_normal_vehicles_model,
            output_dir=tmp_path / "out",
        )
        # Should raise with raise_on_error=True
        with pytest.raises(InvalidCameraDirectionError):
            engine.predict(sample_image_file, invalid_dir)

        # Should return error contract with run_inference (raise_on_error=False)
        result = run_inference(
            str(sample_image_file),
            invalid_dir,
            model=mock_normal_vehicles_model,
            output_dir=tmp_path / "out",
        )
        assert result["status"] == "error"
        assert result["emergency_detected"] is False
        assert "error_message" in result


class TestOutputDictionaryContract:
    """Test 3: Verification of the required output dictionary structure."""

    def test_03_correct_output_dictionary_keys_and_types(
        self, sample_image_file, mock_ambulance_model, tmp_path
    ):
        """Test 3: Return dictionary must match the exact required structure."""
        engine = YOLOInferenceEngine(
            model=mock_ambulance_model,
            output_dir=tmp_path / "out",
        )
        result = engine.predict(sample_image_file, "NORTH")

        # Required fields must not be renamed or removed
        required_fields = {
            "camera",
            "emergency_detected",
            "emergency_type",
            "confidence",
            "detections",
            "annotated_image_path",
            "status",
        }
        assert required_fields.issubset(result.keys())
        assert result["camera"] == "NORTH"
        assert isinstance(result["emergency_detected"], bool)
        assert isinstance(result["confidence"], float)
        assert isinstance(result["detections"], list)
        assert isinstance(result["status"], str)
        assert result["status"] == "success"

        # Verify detection items structure
        for det in result["detections"]:
            assert "class_id" in det
            assert "class_name" in det
            assert "confidence" in det
            assert "bbox" in det
            assert "is_emergency" in det
            assert isinstance(det["class_id"], int)
            assert isinstance(det["class_name"], str)
            assert isinstance(det["confidence"], float)
            assert isinstance(det["bbox"], list)
            assert len(det["bbox"]) == 4
            assert isinstance(det["is_emergency"], bool)


class TestVehicleDetections:
    """Tests 4, 5, 6, 7, 8, 11: Specific vehicle class identification."""

    def test_04_normal_vehicle_detection(
        self, sample_image_file, mock_normal_vehicles_model, tmp_path
    ):
        """Test 4: Normal vehicles detected without triggering emergency status."""
        engine = YOLOInferenceEngine(
            model=mock_normal_vehicles_model,
            output_dir=tmp_path / "out",
        )
        result = engine.predict(sample_image_file, "SOUTH")

        assert result["emergency_detected"] is False
        assert result["emergency_type"] is None
        assert result["confidence"] == 0.0
        assert len(result["detections"]) == 3
        # Ensure all detected classes are non-emergency
        detected_names = [d["class_name"] for d in result["detections"]]
        assert "car" in detected_names
        assert "bus" in detected_names
        assert "motorcycle" in detected_names
        assert all(not d["is_emergency"] for d in result["detections"])

    def test_05_ambulance_detection(
        self, sample_image_file, mock_ambulance_model, tmp_path
    ):
        """Test 5: Ambulance presence flags emergency_detected and emergency_type."""
        engine = YOLOInferenceEngine(
            model=mock_ambulance_model,
            output_dir=tmp_path / "out",
        )
        result = engine.predict(sample_image_file, "EAST")

        assert result["emergency_detected"] is True
        assert result["emergency_type"] == "ambulance"
        assert result["confidence"] == 0.94
        assert any(d["class_name"] == "ambulance" and d["is_emergency"] for d in result["detections"])

    def test_06_police_car_detection(
        self, sample_image_file, mock_police_car_model, tmp_path
    ):
        """Test 6: Police car presence flags emergency_detected and emergency_type."""
        engine = YOLOInferenceEngine(
            model=mock_police_car_model,
            output_dir=tmp_path / "out",
        )
        result = engine.predict(sample_image_file, "WEST")

        assert result["emergency_detected"] is True
        assert result["emergency_type"] == "police_car"
        assert result["confidence"] == 0.92
        assert any(d["class_name"] == "police_car" and d["is_emergency"] for d in result["detections"])

    def test_07_multiple_emergency_detections(
        self, sample_image_file, mock_multiple_emergency_model, tmp_path
    ):
        """Test 7: When multiple emergencies exist, selects highest confidence emergency."""
        # mock_multiple_emergency_model has ambulance (0.96) and police_car (0.85)
        engine = YOLOInferenceEngine(
            model=mock_multiple_emergency_model,
            output_dir=tmp_path / "out",
        )
        result = engine.predict(sample_image_file, "NORTH")

        assert result["emergency_detected"] is True
        # Highest confidence (ambulance 0.96 vs police_car 0.85)
        assert result["emergency_type"] == "ambulance"
        assert result["confidence"] == 0.96
        # Both emergency vehicles must still be present in the detections list
        emergency_in_list = [d for d in result["detections"] if d["is_emergency"]]
        assert len(emergency_in_list) == 2
        classes_in_list = {d["class_name"] for d in emergency_in_list}
        assert classes_in_list == {"ambulance", "police_car"}

    def test_08_no_emergency(
        self, sample_image_file, mock_normal_vehicles_model, tmp_path
    ):
        """Test 8: No emergency vehicles present in scene."""
        engine = YOLOInferenceEngine(
            model=mock_normal_vehicles_model,
            output_dir=tmp_path / "out",
        )
        result = engine.predict(sample_image_file, "SOUTH")

        assert result["emergency_detected"] is False
        assert result["emergency_type"] is None
        assert result["confidence"] == 0.0

    def test_11_empty_detections(
        self, sample_image_file, mock_empty_detections_model, tmp_path
    ):
        """Test 11: Empty scene with zero detections returns success with empty list."""
        engine = YOLOInferenceEngine(
            model=mock_empty_detections_model,
            output_dir=tmp_path / "out",
        )
        result = engine.predict(sample_image_file, "WEST")

        assert result["status"] == "success"
        assert result["emergency_detected"] is False
        assert result["emergency_type"] is None
        assert result["confidence"] == 0.0
        assert result["detections"] == []
        assert Path(result["annotated_image_path"]).exists()


class TestErrorHandlingAndEdgeCases:
    """Tests 9, 10: Validation of image and model errors."""

    def test_09_invalid_missing_or_corrupt_image(
        self, tmp_path, corrupt_image_file, unsupported_format_file, mock_normal_vehicles_model
    ):
        """Test 9: Missing image, corrupt image, and unsupported format handling."""
        engine = YOLOInferenceEngine(
            model=mock_normal_vehicles_model,
            output_dir=tmp_path / "out",
        )

        # 9a. Non-existent file
        non_existent = tmp_path / "does_not_exist.jpg"
        with pytest.raises(InvalidImageError, match="does not exist"):
            engine.predict(non_existent, "NORTH")

        # 9b. Corrupted file
        with pytest.raises(InvalidImageError, match="Corrupt or unreadable"):
            engine.predict(corrupt_image_file, "NORTH")

        # 9c. Unsupported format
        with pytest.raises(InvalidImageError, match="Unsupported image extension"):
            engine.predict(unsupported_format_file, "NORTH")

        # Check error contract with run_inference (raise_on_error=False)
        result = run_inference(
            str(corrupt_image_file),
            "NORTH",
            model=mock_normal_vehicles_model,
        )
        assert result["status"] == "error"
        assert result["emergency_detected"] is False
        assert "Corrupt or unreadable" in result["error_message"]

    def test_10_missing_model_weights(self, sample_image_file, tmp_path):
        """Test 10: Missing model weights file raises ModelNotFoundError."""
        missing_weights = tmp_path / "non_existent_weights.pt"
        engine = YOLOInferenceEngine(
            model_path=missing_weights,
            output_dir=tmp_path / "out",
        )

        with pytest.raises(ModelNotFoundError, match="Model weights file not found"):
            engine.predict(sample_image_file, "NORTH")

        # Run inference wrapper returns status="error"
        result = run_inference(
            str(sample_image_file),
            "NORTH",
            model_path=missing_weights,
        )
        assert result["status"] == "error"
        assert "Model weights file not found" in result["error_message"]


class TestAnnotatedImageCreation:
    """Test 12: Annotated image generation and safety."""

    def test_12_annotated_image_creation(
        self, sample_image_file, mock_ambulance_model, tmp_path
    ):
        """Test 12: Verifies annotated image is created and original is not overwritten."""
        original_bytes_before = sample_image_file.read_bytes()
        out_dir = tmp_path / "annotated_output"

        engine = YOLOInferenceEngine(
            model=mock_ambulance_model,
            output_dir=out_dir,
        )
        result = engine.predict(sample_image_file, "NORTH")

        annotated_path = Path(result["annotated_image_path"])
        assert annotated_path.exists()
        assert annotated_path.is_file()
        assert annotated_path != sample_image_file
        # Verify original was not modified
        assert sample_image_file.read_bytes() == original_bytes_before
        assert annotated_path.stat().st_size > 0
