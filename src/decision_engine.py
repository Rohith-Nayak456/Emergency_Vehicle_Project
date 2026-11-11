"""
Emergency Decision Engine Module.

Phase 5: Emergency Decision Engine and Traffic Signal Controller.
Processes inference results from the YOLO detection module (Teammate 2)
and determines whether an emergency signal preemption should be requested.
"""

from typing import Any, Dict, Optional, Set

# Valid approach directions at a four-way intersection
CAMERAS: Set[str] = {"NORTH", "SOUTH", "EAST", "WEST"}

# Supported emergency vehicle classes
EMERGENCY_CLASSES: Set[str] = {"ambulance", "police_car"}

# Default confidence threshold for requesting preemption
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.50

# Required fields in the inference dictionary contract
REQUIRED_INFERENCE_FIELDS = (
    "status",
    "camera",
    "emergency_detected",
    "emergency_type",
    "confidence",
)


def _build_decision(
    preemption_requested: bool,
    approach: Optional[str],
    emergency_type: Optional[str],
    confidence: float,
    reason: str,
) -> Dict[str, Any]:
    """Helper to build a standardized decision response dictionary."""
    return {
        "preemption_requested": preemption_requested,
        "approach": approach,
        "emergency_type": emergency_type,
        "confidence": confidence,
        "reason": reason,
    }


def process_emergency_decision(
    inference_result: Any,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """
    Process YOLO inference output and determine whether to request signal preemption.

    Validation steps:
    1. Check payload is a dictionary.
    2. Check that all required fields are present.
    3. Check inference status is 'success'.
    4. Check camera direction is valid (NORTH, SOUTH, EAST, WEST).
    5. Check emergency_detected flag is True.
    6. Check that emergency_type is in allowed classes ('ambulance', 'police_car').
    7. Check confidence is a valid number >= confidence_threshold.

    Args:
        inference_result: Dictionary returned by the inference module.
        confidence_threshold: Minimum confidence score required to trigger preemption.

    Returns:
        dict: Standardized decision dictionary:
            {
                "preemption_requested": bool,
                "approach": Optional[str],
                "emergency_type": Optional[str],
                "confidence": float,
                "reason": str,
            }
    """
    # 1. Payload type validation
    if not isinstance(inference_result, dict):
        return _build_decision(
            preemption_requested=False,
            approach=None,
            emergency_type=None,
            confidence=0.0,
            reason="Invalid payload: inference result must be a dictionary",
        )

    # 2. Check for missing required fields
    for field in REQUIRED_INFERENCE_FIELDS:
        if field not in inference_result:
            return _build_decision(
                preemption_requested=False,
                approach=None,
                emergency_type=None,
                confidence=0.0,
                reason=f"Missing required field: '{field}'",
            )

    # 3. Check inference status
    status = inference_result.get("status")
    if status != "success":
        return _build_decision(
            preemption_requested=False,
            approach=None,
            emergency_type=None,
            confidence=0.0,
            reason=f"Inference status was not successful: '{status}'",
        )

    # 4. Check camera direction
    raw_camera = inference_result.get("camera")
    if not isinstance(raw_camera, str):
        return _build_decision(
            preemption_requested=False,
            approach=None,
            emergency_type=None,
            confidence=0.0,
            reason=f"Invalid camera direction type: expected string, got {type(raw_camera).__name__}",
        )

    camera = raw_camera.strip().upper()
    if camera not in CAMERAS:
        return _build_decision(
            preemption_requested=False,
            approach=None,
            emergency_type=None,
            confidence=0.0,
            reason=f"Invalid camera direction: '{raw_camera}'. Expected one of {sorted(CAMERAS)}",
        )

    # 5. Check emergency_detected flag
    emergency_detected = inference_result.get("emergency_detected")
    if not isinstance(emergency_detected, bool) or not emergency_detected:
        return _build_decision(
            preemption_requested=False,
            approach=camera,
            emergency_type=None,
            confidence=0.0,
            reason="No emergency vehicle detected",
        )

    # 6. Check emergency_type class
    raw_emergency_type = inference_result.get("emergency_type")
    if not isinstance(raw_emergency_type, str):
        return _build_decision(
            preemption_requested=False,
            approach=camera,
            emergency_type=None,
            confidence=0.0,
            reason=f"Invalid emergency_type type: expected string, got {type(raw_emergency_type).__name__}",
        )

    emergency_type = raw_emergency_type.strip().lower()
    if emergency_type not in EMERGENCY_CLASSES:
        return _build_decision(
            preemption_requested=False,
            approach=camera,
            emergency_type=emergency_type,
            confidence=0.0,
            reason=f"Invalid emergency class: '{raw_emergency_type}'. Expected one of {sorted(EMERGENCY_CLASSES)}",
        )

    # 7. Check confidence score
    confidence = inference_result.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
        return _build_decision(
            preemption_requested=False,
            approach=camera,
            emergency_type=emergency_type,
            confidence=0.0,
            reason=f"Invalid confidence value: '{confidence}'. Must be a number",
        )

    confidence_val = float(confidence)
    if confidence_val < confidence_threshold:
        return _build_decision(
            preemption_requested=False,
            approach=camera,
            emergency_type=emergency_type,
            confidence=confidence_val,
            reason=f"Confidence {confidence_val:.2f} is below threshold {confidence_threshold:.2f}",
        )

    # All validations passed
    return _build_decision(
        preemption_requested=True,
        approach=camera,
        emergency_type=emergency_type,
        confidence=confidence_val,
        reason=f"Valid {emergency_type} detection above threshold ({confidence_val:.2f} >= {confidence_threshold:.2f})",
    )
