"""
Four-Way Traffic Signal Controller Module with Adaptive Emergency Preemption.

Phase 5: Emergency Decision Engine and Traffic Signal Controller.
Implements:
- 4-way round-robin normal traffic signal cycling (NORTH, EAST, SOUTH, WEST).
- Safe signal transitions: GREEN -> YELLOW -> ALL_RED -> next GREEN.
- Emergency vehicle preemption with configurable priority, clearance, and timings.
- Deterministic simulation clock advancement (no real-time sleeps).
- Safe handling of multiple and conflicting emergency requests (FIFO queue).
"""

from collections import deque
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

from src.decision_engine import (
    CAMERAS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    EMERGENCY_CLASSES,
    process_emergency_decision,
)


class SignalColor(str, Enum):
    """Signal light colors."""
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


class SignalPhase(str, Enum):
    """Controller internal phase states."""
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ALL_RED = "ALL_RED"


class ControllerMode(str, Enum):
    """Operational mode of the traffic controller."""
    NORMAL = "NORMAL"
    EMERGENCY = "EMERGENCY"


# Default approach cycle sequence
DEFAULT_APPROACHES: Tuple[str, ...] = ("NORTH", "EAST", "SOUTH", "WEST")

# Default phase timings (in simulation seconds)
DEFAULT_GREEN_DURATION: float = 10.0
DEFAULT_YELLOW_DURATION: float = 3.0
DEFAULT_ALL_RED_DURATION: float = 2.0
DEFAULT_EMERGENCY_GREEN_DURATION: float = 15.0


class TrafficSignalController:
    """
    Simulates a four-way traffic signal controller supporting emergency preemption.

    Guarantees:
    - Never switch directly from GREEN to GREEN.
    - Transitions always follow: GREEN -> YELLOW -> ALL_RED -> next GREEN.
    - Never allow conflicting GREEN signals (only one approach GREEN at a time).
    - Emergency clearance resets to normal cycling safely.
    - Deterministic clock advancement enables fast and reliable unit testing.
    """

    def __init__(
        self,
        green_duration: float = DEFAULT_GREEN_DURATION,
        yellow_duration: float = DEFAULT_YELLOW_DURATION,
        all_red_duration: float = DEFAULT_ALL_RED_DURATION,
        emergency_green_duration: float = DEFAULT_EMERGENCY_GREEN_DURATION,
        approaches: Tuple[str, ...] = DEFAULT_APPROACHES,
        initial_approach: str = "NORTH",
    ) -> None:
        """
        Initialize the four-way traffic signal controller.

        Args:
            green_duration: Normal green phase duration in seconds.
            yellow_duration: Yellow change phase duration in seconds.
            all_red_duration: All-red clearance phase duration in seconds.
            emergency_green_duration: Emergency green hold duration in seconds.
            approaches: Sequence of approach names in cycle order.
            initial_approach: The starting green approach.
        """
        if green_duration <= 0 or yellow_duration <= 0 or all_red_duration <= 0 or emergency_green_duration <= 0:
            raise ValueError("All signal durations must be strictly positive numbers")

        normalized_approaches = tuple(a.strip().upper() for a in approaches)
        for app in normalized_approaches:
            if app not in CAMERAS:
                raise ValueError(f"Invalid approach: '{app}'. Must be one of {sorted(CAMERAS)}")

        initial_approach = initial_approach.strip().upper()
        if initial_approach not in normalized_approaches:
            raise ValueError(f"Initial approach '{initial_approach}' not found in approaches {normalized_approaches}")

        self.green_duration = float(green_duration)
        self.yellow_duration = float(yellow_duration)
        self.all_red_duration = float(all_red_duration)
        self.emergency_green_duration = float(emergency_green_duration)

        self.approaches: Tuple[str, ...] = normalized_approaches
        self.current_approach_index: int = self.approaches.index(initial_approach)
        self.last_normal_approach_index: int = self.current_approach_index

        self.current_approach: str = initial_approach
        self.phase: SignalPhase = SignalPhase.GREEN
        self.mode: ControllerMode = ControllerMode.NORMAL

        self.time_in_phase: float = 0.0
        self.target_duration: float = self.green_duration
        self.total_time_elapsed: float = 0.0

        # Preemption and queue state
        self.emergency_queue: Deque[Dict[str, Any]] = deque()
        self.active_emergency: Optional[Dict[str, Any]] = None

        # Next transition target
        self.next_approach: Optional[str] = None
        self.next_mode: ControllerMode = ControllerMode.NORMAL

    # -------------------------------------------------------------------------
    # Public State API
    # -------------------------------------------------------------------------

    def get_signal_state(self) -> Dict[str, str]:
        """
        Return the current signal color for all four approaches.

        Format:
            {
                "NORTH": "GREEN",
                "SOUTH": "RED",
                "EAST": "RED",
                "WEST": "RED"
            }
        """
        state = {approach: SignalColor.RED.value for approach in self.approaches}

        if self.phase == SignalPhase.GREEN:
            state[self.current_approach] = SignalColor.GREEN.value
        elif self.phase == SignalPhase.YELLOW:
            state[self.current_approach] = SignalColor.YELLOW.value
        elif self.phase == SignalPhase.ALL_RED:
            # All approaches remain RED during clearance
            pass

        return state

    def get_detailed_state(self) -> Dict[str, Any]:
        """
        Return rich telemetry for frontends and logging.
        Includes time remaining, current mode, active emergency, and queue.
        """
        time_remaining = max(0.0, self.target_duration - self.time_in_phase)
        return {
            "signals": self.get_signal_state(),
            "current_phase": self.phase.value,
            "current_approach": self.current_approach,
            "mode": self.mode.value,
            "time_in_phase": round(self.time_in_phase, 2),
            "target_duration": round(self.target_duration, 2),
            "time_remaining": round(time_remaining, 2),
            "total_time_elapsed": round(self.total_time_elapsed, 2),
            "is_emergency_active": self.mode == ControllerMode.EMERGENCY,
            "active_emergency": self.active_emergency,
            "queued_emergencies_count": len(self.emergency_queue),
            "emergency_queue": list(self.emergency_queue),
        }

    # -------------------------------------------------------------------------
    # Clock Advancement (Deterministic Simulation)
    # -------------------------------------------------------------------------

    def advance_signal(self, delta_time: float = 1.0) -> Dict[str, str]:
        """
        Advance the simulation clock deterministically by delta_time seconds.
        Handles phase timeouts, intermediate safe transitions, and carryover.

        Args:
            delta_time: Elapsed seconds to advance (default 1.0s).

        Returns:
            Dict[str, str]: Current signal state after time advancement.
        """
        if delta_time < 0:
            raise ValueError(f"delta_time cannot be negative: {delta_time}")

        remaining_time = float(delta_time)

        while remaining_time > 0:
            time_needed_in_current_phase = self.target_duration - self.time_in_phase

            if remaining_time < time_needed_in_current_phase:
                self.time_in_phase += remaining_time
                self.total_time_elapsed += remaining_time
                remaining_time = 0.0
            else:
                self.total_time_elapsed += time_needed_in_current_phase
                remaining_time -= time_needed_in_current_phase
                self.time_in_phase = self.target_duration
                self._transition_to_next_phase()

        return self.get_signal_state()

    # -------------------------------------------------------------------------
    # Internal Transitions
    # -------------------------------------------------------------------------

    def _get_next_normal_approach(self) -> str:
        """Calculate the next approach in the round-robin sequence."""
        next_idx = (self.last_normal_approach_index + 1) % len(self.approaches)
        return self.approaches[next_idx]

    def _transition_to_next_phase(self) -> None:
        """
        Execute deterministic state transitions:
        - GREEN -> YELLOW (safety change interval)
        - YELLOW -> ALL_RED (safety intersection clearance)
        - ALL_RED -> GREEN (service next normal or emergency approach)
        """
        if self.phase == SignalPhase.GREEN:
            # 1. GREEN becomes YELLOW
            self.phase = SignalPhase.YELLOW
            self.time_in_phase = 0.0
            self.target_duration = self.yellow_duration

        elif self.phase == SignalPhase.YELLOW:
            # 2. YELLOW becomes ALL_RED
            self.phase = SignalPhase.ALL_RED
            self.time_in_phase = 0.0
            self.target_duration = self.all_red_duration

            # Resolve next approach to receive GREEN after clearance
            if self.emergency_queue:
                next_req = self.emergency_queue.popleft()
                self.active_emergency = next_req
                self.next_approach = next_req["approach"]
                self.next_mode = ControllerMode.EMERGENCY
            else:
                self.active_emergency = None
                self.next_approach = self._get_next_normal_approach()
                self.next_mode = ControllerMode.NORMAL

        elif self.phase == SignalPhase.ALL_RED:
            # 3. ALL_RED becomes GREEN for next approach
            target_approach = self.next_approach or self._get_next_normal_approach()
            self.current_approach = target_approach
            self.phase = SignalPhase.GREEN
            self.mode = self.next_mode
            self.time_in_phase = 0.0

            if self.mode == ControllerMode.EMERGENCY:
                self.target_duration = self.emergency_green_duration
            else:
                self.target_duration = self.green_duration
                self.current_approach_index = self.approaches.index(self.current_approach)
                self.last_normal_approach_index = self.current_approach_index

    # -------------------------------------------------------------------------
    # Emergency Preemption and Clearance
    # -------------------------------------------------------------------------

    def request_emergency_preemption(
        self,
        approach: str,
        emergency_type: str = "ambulance",
        confidence: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Request emergency preemption for a specific approach.

        Transitions:
        - If requested approach is already GREEN in EMERGENCY mode:
          Extends / refreshes emergency green duration.
        - If requested approach is already GREEN in NORMAL mode:
          Immediately switches to EMERGENCY mode and holds green for emergency duration.
        - If another approach is GREEN:
          Safely initiates transition (current approach switches to YELLOW -> ALL_RED -> requested GREEN).
          No green signals ever conflict.
        - If preemption is already active for another approach:
          Queues the new request in FIFO order to service after current emergency clears.

        Args:
            approach: Approach direction ('NORTH', 'SOUTH', 'EAST', 'WEST').
            emergency_type: 'ambulance' or 'police_car'.
            confidence: Detection confidence score (0.0 to 1.0).

        Returns:
            Dict[str, Any]: Preemption request result status.
        """
        # Validate approach
        if not isinstance(approach, str):
            return {
                "status": "error",
                "approach": None,
                "message": "Approach must be a string",
                "signal_state": self.get_signal_state(),
            }

        norm_approach = approach.strip().upper()
        if norm_approach not in self.approaches:
            return {
                "status": "error",
                "approach": norm_approach,
                "message": f"Invalid approach '{norm_approach}'. Valid: {self.approaches}",
                "signal_state": self.get_signal_state(),
            }

        # Validate emergency type
        if not isinstance(emergency_type, str):
            return {
                "status": "error",
                "approach": norm_approach,
                "message": "Emergency type must be a string",
                "signal_state": self.get_signal_state(),
            }

        norm_type = emergency_type.strip().lower()
        if norm_type not in EMERGENCY_CLASSES:
            return {
                "status": "error",
                "approach": norm_approach,
                "message": f"Invalid emergency type '{norm_type}'. Allowed: {sorted(EMERGENCY_CLASSES)}",
                "signal_state": self.get_signal_state(),
            }

        # Validate confidence
        try:
            conf_val = float(confidence)
        except (ValueError, TypeError):
            conf_val = 0.0

        emergency_record = {
            "approach": norm_approach,
            "emergency_type": norm_type,
            "confidence": conf_val,
        }

        # Case 1: The requested approach is already GREEN
        if self.phase == SignalPhase.GREEN and self.current_approach == norm_approach:
            self.mode = ControllerMode.EMERGENCY
            self.active_emergency = emergency_record
            self.target_duration = self.emergency_green_duration
            self.time_in_phase = 0.0  # Reset timer to give full emergency window
            return {
                "status": "active",
                "approach": norm_approach,
                "emergency_type": norm_type,
                "confidence": conf_val,
                "message": f"Preemption active: approach {norm_approach} already GREEN, extended emergency duration",
                "signal_state": self.get_signal_state(),
            }

        # Case 2: An emergency is already active for a DIFFERENT approach
        if self.mode == ControllerMode.EMERGENCY and self.current_approach != norm_approach:
            # Check if this approach is already queued
            for item in self.emergency_queue:
                if item["approach"] == norm_approach:
                    if conf_val > item.get("confidence", 0.0):
                        item["confidence"] = conf_val
                    return {
                        "status": "queued",
                        "approach": norm_approach,
                        "emergency_type": norm_type,
                        "confidence": conf_val,
                        "message": f"Emergency for {norm_approach} is already queued",
                        "signal_state": self.get_signal_state(),
                    }

            self.emergency_queue.append(emergency_record)
            return {
                "status": "queued",
                "approach": norm_approach,
                "emergency_type": norm_type,
                "confidence": conf_val,
                "message": f"Queued emergency preemption for {norm_approach} (position {len(self.emergency_queue)})",
                "signal_state": self.get_signal_state(),
            }

        # Case 3: In NORMAL mode, a different approach is currently GREEN
        if self.mode == ControllerMode.NORMAL and self.phase == SignalPhase.GREEN:
            # Enqueue the emergency request
            self.emergency_queue.append(emergency_record)
            # Immediately trigger safe change: GREEN -> YELLOW
            self.phase = SignalPhase.YELLOW
            self.time_in_phase = 0.0
            self.target_duration = self.yellow_duration
            return {
                "status": "transitioning",
                "approach": norm_approach,
                "emergency_type": norm_type,
                "confidence": conf_val,
                "message": f"Initiated safe preemption: current GREEN ({self.current_approach}) switched to YELLOW",
                "signal_state": self.get_signal_state(),
            }

        # Case 4: Currently in YELLOW or ALL_RED transition
        # Approach will be served immediately upon ALL_RED completion
        # Check if already queued
        for item in self.emergency_queue:
            if item["approach"] == norm_approach:
                return {
                    "status": "queued",
                    "approach": norm_approach,
                    "emergency_type": norm_type,
                    "confidence": conf_val,
                    "message": f"Emergency for {norm_approach} already queued during transition",
                    "signal_state": self.get_signal_state(),
                }

        self.emergency_queue.append(emergency_record)
        return {
            "status": "transitioning",
            "approach": norm_approach,
            "emergency_type": norm_type,
            "confidence": conf_val,
            "message": f"Preemption requested during clearance; {norm_approach} queued for next GREEN",
            "signal_state": self.get_signal_state(),
        }

    def clear_emergency(self, clear_queue: bool = True) -> Dict[str, Any]:
        """
        Signal that the emergency vehicle has cleared the intersection.
        Initiates a safe transition away from emergency green:
        - Current approach switches to YELLOW (yellow_duration).
        - Then all directions switch to ALL_RED (all_red_duration).
        - Resumes normal traffic cycling (or next queued emergency if clear_queue is False).

        Args:
            clear_queue: If True, also clears any pending queued emergencies.

        Returns:
            Dict[str, Any]: Clearance result status.
        """
        if clear_queue:
            self.emergency_queue.clear()

        # If currently in EMERGENCY GREEN, transition safely through YELLOW
        if self.mode == ControllerMode.EMERGENCY and self.phase == SignalPhase.GREEN:
            self.phase = SignalPhase.YELLOW
            self.time_in_phase = 0.0
            self.target_duration = self.yellow_duration
            self.active_emergency = None
            return {
                "status": "clearing",
                "message": f"Emergency cleared for {self.current_approach}; initiated safe YELLOW transition",
                "signal_state": self.get_signal_state(),
            }

        # If already transitioning or in normal mode
        self.active_emergency = None
        if not self.emergency_queue:
            self.next_mode = ControllerMode.NORMAL

        return {
            "status": "cleared",
            "message": "Emergency cleared; returning safely to normal operation",
            "signal_state": self.get_signal_state(),
        }

    def process_and_apply(
        self,
        inference_result: Any,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Convenience method: Evaluate an inference result and automatically apply
        preemption if valid and above threshold.
        """
        decision = process_emergency_decision(inference_result, confidence_threshold)
        preemption_result = None

        if decision["preemption_requested"]:
            preemption_result = self.request_emergency_preemption(
                approach=decision["approach"],
                emergency_type=decision["emergency_type"],
                confidence=decision["confidence"],
            )

        return {
            "decision": decision,
            "preemption": preemption_result,
            "signal_state": self.get_signal_state(),
        }


# =============================================================================
# Module-Level Default Controller & Functional Interface
# =============================================================================

_default_controller: Optional[TrafficSignalController] = None


def get_controller() -> TrafficSignalController:
    """Get or lazily initialize the default global controller singleton."""
    global _default_controller
    if _default_controller is None:
        _default_controller = TrafficSignalController()
    return _default_controller


def reset_controller(
    green_duration: float = DEFAULT_GREEN_DURATION,
    yellow_duration: float = DEFAULT_YELLOW_DURATION,
    all_red_duration: float = DEFAULT_ALL_RED_DURATION,
    emergency_green_duration: float = DEFAULT_EMERGENCY_GREEN_DURATION,
    initial_approach: str = "NORTH",
) -> TrafficSignalController:
    """Reset the default controller instance to initial state."""
    global _default_controller
    _default_controller = TrafficSignalController(
        green_duration=green_duration,
        yellow_duration=yellow_duration,
        all_red_duration=all_red_duration,
        emergency_green_duration=emergency_green_duration,
        initial_approach=initial_approach,
    )
    return _default_controller


def request_emergency_preemption(
    approach: str,
    emergency_type: str = "ambulance",
    confidence: float = 1.0,
) -> Dict[str, Any]:
    """Module-level helper: request preemption on default controller."""
    return get_controller().request_emergency_preemption(
        approach=approach,
        emergency_type=emergency_type,
        confidence=confidence,
    )


def get_signal_state() -> Dict[str, str]:
    """Module-level helper: get signal state from default controller."""
    return get_controller().get_signal_state()


def advance_signal(delta_time: float = 1.0) -> Dict[str, str]:
    """Module-level helper: advance default controller clock."""
    return get_controller().advance_signal(delta_time=delta_time)


def clear_emergency(clear_queue: bool = True) -> Dict[str, Any]:
    """Module-level helper: clear emergency on default controller."""
    return get_controller().clear_emergency(clear_queue=clear_queue)


def process_and_apply_decision(
    inference_result: Any,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> Dict[str, Any]:
    """Module-level helper: evaluate decision and apply to default controller."""
    return get_controller().process_and_apply(
        inference_result=inference_result,
        confidence_threshold=confidence_threshold,
    )
