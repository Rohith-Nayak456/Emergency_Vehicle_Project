"""
Unit tests for the decision engine module.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from src.decision_engine import DecisionEngine, DetectionResult, PreemptionDecision
from src.signal_controller import SignalState, IntersectionPhase


def make_detection(
    emergency: bool = False,
    emergency_class: str = "ambulance",
    emergency_conf: float = 0.94,
    total_vehicles: int = 5,
) -> DetectionResult:
    """Helper to create test DetectionResult objects."""
    emergency_vehicles = []
    if emergency:
        emergency_vehicles = [{
            "class": emergency_class,
            "class_id": 6 if emergency_class == "ambulance" else 7,
            "confidence": emergency_conf,
            "bbox": [120, 80, 350, 260],
        }]

    return DetectionResult(
        emergency_detected=emergency,
        emergency_vehicles=emergency_vehicles,
        all_detections=[{"class": "car", "class_id": 0, "confidence": 0.8, "bbox": [0, 0, 100, 100]}] * total_vehicles,
        vehicle_counts={"car": total_vehicles},
        total_vehicles=total_vehicles,
        emergency_count=len(emergency_vehicles),
        max_emergency_confidence=emergency_conf if emergency else 0.0,
    )


class TestDecisionEngine:
    """Tests for the DecisionEngine."""

    def setup_method(self):
        self.engine = DecisionEngine(emergency_confidence_threshold=0.40)

    # ── No emergency tests ──────────────────────────────────────────

    def test_no_emergency_no_preemption(self):
        """No emergency vehicle -> no preemption."""
        det = make_detection(emergency=False)
        decision = self.engine.evaluate(det, approach_direction="North")
        assert not decision.should_preempt
        assert decision.confidence_level == "none"
        assert not decision.is_uncertain

    def test_no_emergency_normal_signal(self):
        """No emergency -> signal sequence shows normal operation."""
        det = make_detection(emergency=False)
        decision = self.engine.evaluate(det, approach_direction="North")
        assert len(decision.signal_sequence) >= 1
        assert decision.signal_sequence[0].phase == IntersectionPhase.NORMAL_GREEN

    # ── High confidence emergency tests ─────────────────────────────

    def test_ambulance_high_confidence_preemption(self):
        """Ambulance with 94% confidence from North -> preempt North."""
        det = make_detection(emergency=True, emergency_class="ambulance", emergency_conf=0.94)
        decision = self.engine.evaluate(det, approach_direction="North")
        assert decision.should_preempt
        assert decision.emergency_type == "ambulance"
        assert decision.approach_direction == "North"
        assert not decision.is_uncertain

    def test_police_car_high_confidence_preemption(self):
        """Police car with 85% confidence from South -> preempt South."""
        det = make_detection(emergency=True, emergency_class="police_car", emergency_conf=0.85)
        decision = self.engine.evaluate(det, approach_direction="South")
        assert decision.should_preempt
        assert decision.emergency_type == "police_car"
        assert decision.approach_direction == "South"

    def test_preemption_signal_sequence_has_clearance(self):
        """Preemption sequence must include yellow and all-red clearance."""
        det = make_detection(emergency=True, emergency_conf=0.90)
        decision = self.engine.evaluate(
            det, approach_direction="North", current_green_direction="East"
        )
        phases = [s.phase for s in decision.signal_sequence]
        assert IntersectionPhase.NORMAL_YELLOW in phases
        assert IntersectionPhase.ALL_RED_CLEARANCE in phases
        assert IntersectionPhase.EMERGENCY_GREEN in phases

    def test_preemption_no_conflicting_greens(self):
        """Preemption signal sequence must never have conflicting greens."""
        det = make_detection(emergency=True, emergency_conf=0.90)
        for direction in ["North", "South", "East", "West"]:
            decision = self.engine.evaluate(det, approach_direction=direction)
            for snap in decision.signal_sequence:
                assert not snap.has_conflicting_greens(), \
                    f"Conflicting greens in direction={direction}"

    # ── Low confidence tests ────────────────────────────────────────

    def test_low_confidence_no_auto_preemption(self):
        """Emergency below threshold -> no automatic preemption."""
        det = make_detection(emergency=True, emergency_conf=0.25)
        decision = self.engine.evaluate(det, approach_direction="North")
        assert not decision.should_preempt
        assert decision.confidence_level == "low"
        assert decision.is_uncertain

    def test_borderline_confidence(self):
        """Emergency exactly at threshold -> should preempt."""
        det = make_detection(emergency=True, emergency_conf=0.40)
        decision = self.engine.evaluate(det, approach_direction="North")
        assert decision.should_preempt

    # ── Missing approach tests ──────────────────────────────────────

    def test_no_approach_no_preemption(self):
        """Emergency detected but no approach selected -> no preemption."""
        det = make_detection(emergency=True, emergency_conf=0.90)
        decision = self.engine.evaluate(det, approach_direction=None)
        assert not decision.should_preempt
        assert decision.is_uncertain
        assert "approach direction" in decision.explanation.lower()

    def test_invalid_approach_no_preemption(self):
        """Emergency detected but invalid approach -> no preemption."""
        det = make_detection(emergency=True, emergency_conf=0.90)
        decision = self.engine.evaluate(det, approach_direction="Northeast")
        assert not decision.should_preempt
        assert decision.is_uncertain

    # ── Explanation tests ───────────────────────────────────────────

    def test_explanation_mentions_vehicle_type(self):
        """Explanation should mention the detected emergency vehicle type."""
        det = make_detection(emergency=True, emergency_class="ambulance", emergency_conf=0.90)
        decision = self.engine.evaluate(det, approach_direction="North")
        assert "ambulance" in decision.explanation.lower()

    def test_explanation_mentions_confidence(self):
        """Explanation should include the confidence score."""
        det = make_detection(emergency=True, emergency_conf=0.90)
        decision = self.engine.evaluate(det, approach_direction="North")
        assert "90" in decision.explanation

    def test_explanation_mentions_simulation(self):
        """Explanation should mention this is a simulation."""
        det = make_detection(emergency=True, emergency_conf=0.90)
        decision = self.engine.evaluate(det, approach_direction="North")
        assert "simulated" in decision.explanation.lower() or "simulation" in decision.explanation.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
