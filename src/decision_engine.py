"""
Emergency decision engine.

Takes detection results and produces a signal recommendation.
Separated from the signal controller to maintain single-responsibility.

IMPORTANT LIMITATION:
A single still image cannot determine vehicle speed, direction of travel,
or time-to-arrival. The approach direction is user-selected and represents
a known camera position, NOT a direction inferred from pixels.
"""
from dataclasses import dataclass
from typing import Optional

from src.signal_controller import SignalController, SignalSnapshot, DIRECTIONS


@dataclass
class DetectionResult:
    """Structured result from the inference module."""
    emergency_detected: bool
    emergency_vehicles: list[dict]  # [{class, confidence, bbox}]
    all_detections: list[dict]      # [{class, confidence, bbox}]
    vehicle_counts: dict[str, int]  # {class_name: count}
    total_vehicles: int
    emergency_count: int
    max_emergency_confidence: float


@dataclass
class PreemptionDecision:
    """The system's recommendation regarding signal preemption."""
    should_preempt: bool
    confidence_level: str           # "high", "medium", "low", "none"
    emergency_type: Optional[str]   # "ambulance", "police_car", or None
    emergency_confidence: float
    approach_direction: Optional[str]
    explanation: str
    signal_sequence: list[SignalSnapshot]
    is_uncertain: bool


class DecisionEngine:
    """
    Evaluates detection results and produces a preemption recommendation.

    Decision rules:
    1. No emergency vehicle detected -> normal operation, no preemption.
    2. Emergency detected but confidence < threshold -> uncertain, no auto-preempt.
    3. Emergency detected with high confidence + valid approach -> preempt.
    4. No approach direction selected -> ask user, no auto-preempt.
    """

    def __init__(
        self,
        emergency_confidence_threshold: float = 0.40,
        signal_controller: Optional[SignalController] = None,
    ):
        self.emergency_threshold = emergency_confidence_threshold
        self.controller = signal_controller or SignalController()

    def evaluate(
        self,
        detection: DetectionResult,
        approach_direction: Optional[str] = None,
        current_green_direction: str = "East",
    ) -> PreemptionDecision:
        """
        Evaluate detection results and produce a preemption decision.

        Args:
            detection: Structured detection result from inference.
            approach_direction: User-selected camera approach direction.
            current_green_direction: Which direction currently has green.

        Returns:
            PreemptionDecision with full recommendation and signal sequence.
        """
        # Case 1: No emergency vehicle detected
        if not detection.emergency_detected or detection.emergency_count == 0:
            normal_snapshot = self.controller.get_normal_cycle_snapshot()
            return PreemptionDecision(
                should_preempt=False,
                confidence_level="none",
                emergency_type=None,
                emergency_confidence=0.0,
                approach_direction=approach_direction,
                explanation=(
                    "No emergency vehicle detected in the image. "
                    "Normal signal operation continues."
                ),
                signal_sequence=[normal_snapshot],
                is_uncertain=False,
            )

        # Find the highest-confidence emergency detection
        best_emergency = max(
            detection.emergency_vehicles, key=lambda d: d["confidence"]
        )
        best_confidence = best_emergency["confidence"]
        emergency_class = best_emergency["class"]

        # Case 2: No approach direction selected
        if not approach_direction or approach_direction not in DIRECTIONS:
            return PreemptionDecision(
                should_preempt=False,
                confidence_level=self._confidence_level(best_confidence),
                emergency_type=emergency_class,
                emergency_confidence=best_confidence,
                approach_direction=None,
                explanation=(
                    f"A {emergency_class} was detected with "
                    f"{best_confidence*100:.1f}% confidence, but no valid "
                    f"approach direction has been selected. Please select the "
                    f"camera approach direction (North/South/East/West) to "
                    f"enable signal preemption."
                ),
                signal_sequence=[],
                is_uncertain=True,
            )

        # Case 3: Low confidence — uncertain
        if best_confidence < self.emergency_threshold:
            normal_snapshot = self.controller.get_normal_cycle_snapshot()
            return PreemptionDecision(
                should_preempt=False,
                confidence_level="low",
                emergency_type=emergency_class,
                emergency_confidence=best_confidence,
                approach_direction=approach_direction,
                explanation=(
                    f"A possible {emergency_class} was detected with only "
                    f"{best_confidence*100:.1f}% confidence (below the "
                    f"{self.emergency_threshold*100:.0f}% threshold). "
                    f"Automatic preemption is NOT recommended. Manual "
                    f"verification is required."
                ),
                signal_sequence=[normal_snapshot],
                is_uncertain=True,
            )

        # Case 4: High-confidence emergency with valid approach -> PREEMPT
        signal_sequence = self.controller.generate_emergency_preemption(
            emergency_direction=approach_direction,
            current_green_direction=current_green_direction,
        )
        # Add recovery sequence
        recovery = self.controller.generate_post_emergency_recovery(
            emergency_direction=approach_direction
        )
        full_sequence = signal_sequence + recovery

        # Safety check
        assert self.controller.validate_no_conflicting_greens(full_sequence), \
            "SAFETY VIOLATION: Conflicting green signals detected!"

        conf_level = self._confidence_level(best_confidence)

        return PreemptionDecision(
            should_preempt=True,
            confidence_level=conf_level,
            emergency_type=emergency_class,
            emergency_confidence=best_confidence,
            approach_direction=approach_direction,
            explanation=(
                f"A {emergency_class} was detected with "
                f"{best_confidence*100:.1f}% confidence. The selected camera "
                f"approach is {approach_direction}. The system recommends "
                f"simulated emergency preemption for {approach_direction} "
                f"after the required yellow and all-red clearance phases.\n\n"
                f"NOTE: This is a simulated recommendation. The approach "
                f"direction is user-selected and does not represent actual "
                f"vehicle trajectory analysis from a single still image."
            ),
            signal_sequence=full_sequence,
            is_uncertain=False,
        )

    def _confidence_level(self, confidence: float) -> str:
        if confidence >= 0.80:
            return "high"
        elif confidence >= self.emergency_threshold:
            return "medium"
        else:
            return "low"
