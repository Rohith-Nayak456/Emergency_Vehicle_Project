"""
Emergency Vehicle Detection and Adaptive Signal Preemption Package.
"""

from src.decision_engine import (
    CAMERAS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    EMERGENCY_CLASSES,
    REQUIRED_INFERENCE_FIELDS,
    process_emergency_decision,
)
from src.signal_controller import (
    DEFAULT_ALL_RED_DURATION,
    DEFAULT_APPROACHES,
    DEFAULT_EMERGENCY_GREEN_DURATION,
    DEFAULT_GREEN_DURATION,
    DEFAULT_YELLOW_DURATION,
    ControllerMode,
    SignalColor,
    SignalPhase,
    TrafficSignalController,
    advance_signal,
    clear_emergency,
    get_controller,
    get_signal_state,
    process_and_apply_decision,
    request_emergency_preemption,
    reset_controller,
)

__all__ = [
    # Decision Engine
    "process_emergency_decision",
    "CAMERAS",
    "EMERGENCY_CLASSES",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "REQUIRED_INFERENCE_FIELDS",
    # Signal Controller
    "TrafficSignalController",
    "SignalColor",
    "SignalPhase",
    "ControllerMode",
    "DEFAULT_APPROACHES",
    "DEFAULT_GREEN_DURATION",
    "DEFAULT_YELLOW_DURATION",
    "DEFAULT_ALL_RED_DURATION",
    "DEFAULT_EMERGENCY_GREEN_DURATION",
    # Functional Interface
    "get_signal_state",
    "advance_signal",
    "request_emergency_preemption",
    "clear_emergency",
    "process_and_apply_decision",
    "reset_controller",
    "get_controller",
]
