"""
Unit tests for Emergency Decision Engine (Teammate 3 - Phase 5).
Covers validation rules, error handling, thresholding, and fail-safes.
"""

import pytest

from src.decision_engine import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    process_emergency_decision,
)


@pytest.fixture
def valid_ambulance_payload():
    """Standard valid ambulance inference payload from Teammate 2."""
    return {
        "camera": "NORTH",
        "emergency_detected": True,
        "emergency_type": "ambulance",
        "confidence": 0.94,
        "detections": [
            {
                "class_id": 6,
                "class_name": "ambulance",
                "confidence": 0.94,
                "bbox": [50, 60, 200, 220],
                "is_emergency": True,
            }
        ],
        "annotated_image_path": "output/annotated_sample.jpg",
        "status": "success",
    }


@pytest.fixture
def valid_police_car_payload():
    """Standard valid police car inference payload from Teammate 2."""
    return {
        "camera": "WEST",
        "emergency_detected": True,
        "emergency_type": "police_car",
        "confidence": 0.91,
        "detections": [
            {
                "class_id": 7,
                "class_name": "police_car",
                "confidence": 0.91,
                "bbox": [10, 20, 150, 180],
                "is_emergency": True,
            }
        ],
        "annotated_image_path": "output/annotated_police.jpg",
        "status": "success",
    }


# -----------------------------------------------------------------------------
# Test 3: Ambulance detection
# -----------------------------------------------------------------------------
def test_ambulance_detection(valid_ambulance_payload):
    result = process_emergency_decision(valid_ambulance_payload)
    assert result["preemption_requested"] is True
    assert result["approach"] == "NORTH"
    assert result["emergency_type"] == "ambulance"
    assert result["confidence"] == 0.94
    assert "Valid ambulance detection" in result["reason"]


# -----------------------------------------------------------------------------
# Test 4: Police car detection
# -----------------------------------------------------------------------------
def test_police_car_detection(valid_police_car_payload):
    result = process_emergency_decision(valid_police_car_payload)
    assert result["preemption_requested"] is True
    assert result["approach"] == "WEST"
    assert result["emergency_type"] == "police_car"
    assert result["confidence"] == 0.91
    assert "Valid police_car detection" in result["reason"]


# -----------------------------------------------------------------------------
# Test 5: Low-confidence detection
# -----------------------------------------------------------------------------
def test_low_confidence_detection(valid_ambulance_payload):
    valid_ambulance_payload["confidence"] = 0.35  # Below default 0.50 threshold
    result = process_emergency_decision(valid_ambulance_payload)
    assert result["preemption_requested"] is False
    assert result["approach"] == "NORTH"
    assert result["confidence"] == 0.35
    assert "below threshold" in result["reason"]


def test_custom_confidence_threshold(valid_ambulance_payload):
    valid_ambulance_payload["confidence"] = 0.72
    # Reject at higher threshold (0.80)
    res_high = process_emergency_decision(valid_ambulance_payload, confidence_threshold=0.80)
    assert res_high["preemption_requested"] is False

    # Accept at lower threshold (0.70)
    res_low = process_emergency_decision(valid_ambulance_payload, confidence_threshold=0.70)
    assert res_low["preemption_requested"] is True


# -----------------------------------------------------------------------------
# Test 6: Normal vehicle detection (not an emergency)
# -----------------------------------------------------------------------------
def test_normal_vehicle_detection():
    payload = {
        "camera": "SOUTH",
        "emergency_detected": False,
        "emergency_type": None,
        "confidence": 0.0,
        "detections": [
            {
                "class_id": 0,
                "class_name": "car",
                "confidence": 0.88,
                "bbox": [30, 40, 160, 170],
                "is_emergency": False,
            }
        ],
        "annotated_image_path": "output/annotated_car.jpg",
        "status": "success",
    }
    result = process_emergency_decision(payload)
    assert result["preemption_requested"] is False
    assert result["approach"] == "SOUTH"
    assert "No emergency vehicle detected" in result["reason"]


# -----------------------------------------------------------------------------
# Test 7: Invalid direction
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("invalid_cam", ["NORTHEAST", "CENTER", "NORTH-WEST", "", 123])
def test_invalid_direction(valid_ambulance_payload, invalid_cam):
    valid_ambulance_payload["camera"] = invalid_cam
    result = process_emergency_decision(valid_ambulance_payload)
    assert result["preemption_requested"] is False
    assert "camera direction" in result["reason"].lower()


def test_case_insensitive_direction(valid_ambulance_payload):
    valid_ambulance_payload["camera"] = "  south  "
    result = process_emergency_decision(valid_ambulance_payload)
    assert result["preemption_requested"] is True
    assert result["approach"] == "SOUTH"


# -----------------------------------------------------------------------------
# Test 8: Invalid emergency class
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("invalid_class", ["firetruck", "helicopter", "tank", "bicycle"])
def test_invalid_emergency_class(valid_ambulance_payload, invalid_class):
    valid_ambulance_payload["emergency_type"] = invalid_class
    result = process_emergency_decision(valid_ambulance_payload)
    assert result["preemption_requested"] is False
    assert "Invalid emergency class" in result["reason"]


# -----------------------------------------------------------------------------
# Test 9: Failed inference
# -----------------------------------------------------------------------------
@pytest.mark.parametrize("bad_status", ["error", "failed", "timeout", "model_unavailable"])
def test_failed_inference(valid_ambulance_payload, bad_status):
    valid_ambulance_payload["status"] = bad_status
    result = process_emergency_decision(valid_ambulance_payload)
    assert result["preemption_requested"] is False
    assert "Inference status was not successful" in result["reason"]


# -----------------------------------------------------------------------------
# Test 10: Missing input fields
# -----------------------------------------------------------------------------
@pytest.mark.parametrize(
    "missing_field",
    ["status", "camera", "emergency_detected", "emergency_type", "confidence"],
)
def test_missing_input_fields(valid_ambulance_payload, missing_field):
    del valid_ambulance_payload[missing_field]
    result = process_emergency_decision(valid_ambulance_payload)
    assert result["preemption_requested"] is False
    assert f"Missing required field: '{missing_field}'" in result["reason"]


# -----------------------------------------------------------------------------
# Additional fail-safe edge cases
# -----------------------------------------------------------------------------
def test_non_dict_payload():
    assert process_emergency_decision(None)["preemption_requested"] is False
    assert process_emergency_decision("not a dict")["preemption_requested"] is False
    assert process_emergency_decision([1, 2, 3])["preemption_requested"] is False


def test_invalid_confidence_type(valid_ambulance_payload):
    valid_ambulance_payload["confidence"] = "high"
    result = process_emergency_decision(valid_ambulance_payload)
    assert result["preemption_requested"] is False
    assert "Invalid confidence value" in result["reason"]


def test_boolean_confidence_rejected(valid_ambulance_payload):
    valid_ambulance_payload["confidence"] = True  # bool is a subclass of int in python
    result = process_emergency_decision(valid_ambulance_payload)
    assert result["preemption_requested"] is False
    assert "Invalid confidence value" in result["reason"]
