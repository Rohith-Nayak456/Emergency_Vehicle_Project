"""
Unit tests for the signal controller module.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from src.signal_controller import (
    SignalController,
    SignalState,
    IntersectionPhase,
    DIRECTIONS,
    SignalSnapshot,
)


class TestSignalController:
    """Tests for the SignalController state machine."""

    def setup_method(self):
        self.controller = SignalController()

    # ── Normal cycle tests ──────────────────────────────────────────

    def test_normal_cycle_snapshot_has_one_green(self):
        """Normal snapshot should have exactly one green signal."""
        snap = self.controller.get_normal_cycle_snapshot()
        greens = sum(
            1 for d in DIRECTIONS
            if snap.get_signal(d) == SignalState.GREEN
        )
        assert greens == 1

    def test_normal_cycle_no_conflicting_greens(self):
        """Full normal cycle should never have two greens at once."""
        cycle = self.controller.generate_normal_cycle()
        assert self.controller.validate_no_conflicting_greens(cycle)

    def test_normal_cycle_has_yellow_between_greens(self):
        """Each green phase must be followed by a yellow phase."""
        cycle = self.controller.generate_normal_cycle()
        for i, snap in enumerate(cycle):
            if snap.phase == IntersectionPhase.NORMAL_GREEN and i + 1 < len(cycle):
                assert cycle[i + 1].phase == IntersectionPhase.NORMAL_YELLOW

    def test_normal_cycle_has_all_red_after_yellow(self):
        """Each yellow phase must be followed by all-red clearance."""
        cycle = self.controller.generate_normal_cycle()
        for i, snap in enumerate(cycle):
            if snap.phase == IntersectionPhase.NORMAL_YELLOW and i + 1 < len(cycle):
                assert cycle[i + 1].phase == IntersectionPhase.ALL_RED_CLEARANCE

    def test_normal_cycle_covers_all_directions(self):
        """Normal cycle must give green to all four directions."""
        cycle = self.controller.generate_normal_cycle()
        green_dirs = {
            snap.active_direction
            for snap in cycle
            if snap.phase == IntersectionPhase.NORMAL_GREEN
        }
        assert green_dirs == set(DIRECTIONS)

    # ── Emergency preemption tests ──────────────────────────────────

    def test_emergency_preemption_north(self):
        """Emergency from North: North should get green after clearance."""
        seq = self.controller.generate_emergency_preemption(
            emergency_direction="North",
            current_green_direction="East",
        )
        # Must have yellow, all-red, then emergency green
        phases = [s.phase for s in seq]
        assert IntersectionPhase.NORMAL_YELLOW in phases
        assert IntersectionPhase.ALL_RED_CLEARANCE in phases
        assert IntersectionPhase.EMERGENCY_GREEN in phases

        # Emergency green must be for North
        emg_green = [s for s in seq if s.phase == IntersectionPhase.EMERGENCY_GREEN]
        assert len(emg_green) == 1
        assert emg_green[0].north == SignalState.GREEN

    def test_emergency_preemption_south(self):
        """Emergency from South: South should get green."""
        seq = self.controller.generate_emergency_preemption(
            emergency_direction="South",
            current_green_direction="North",
        )
        emg_green = [s for s in seq if s.phase == IntersectionPhase.EMERGENCY_GREEN]
        assert len(emg_green) == 1
        assert emg_green[0].south == SignalState.GREEN

    def test_emergency_when_already_green(self):
        """If emergency direction already has green, just extend."""
        seq = self.controller.generate_emergency_preemption(
            emergency_direction="East",
            current_green_direction="East",
        )
        assert len(seq) == 1
        assert seq[0].phase == IntersectionPhase.EMERGENCY_GREEN
        assert seq[0].east == SignalState.GREEN

    def test_no_conflicting_greens_during_preemption(self):
        """Emergency preemption must never have two greens simultaneously."""
        for direction in DIRECTIONS:
            for current in DIRECTIONS:
                seq = self.controller.generate_emergency_preemption(
                    emergency_direction=direction,
                    current_green_direction=current,
                )
                assert self.controller.validate_no_conflicting_greens(seq), \
                    f"Conflict: emergency={direction}, current={current}"

    def test_other_directions_red_during_emergency_green(self):
        """During emergency green, all other directions must be red."""
        for direction in DIRECTIONS:
            seq = self.controller.generate_emergency_preemption(
                emergency_direction=direction,
                current_green_direction="East" if direction != "East" else "North",
            )
            emg_green = [s for s in seq if s.phase == IntersectionPhase.EMERGENCY_GREEN]
            for snap in emg_green:
                for d in DIRECTIONS:
                    if d != direction:
                        assert snap.get_signal(d) == SignalState.RED, \
                            f"{d} should be RED when {direction} has emergency green"

    def test_yellow_before_switching_away(self):
        """Current green must go through yellow before being switched."""
        seq = self.controller.generate_emergency_preemption(
            emergency_direction="North",
            current_green_direction="South",
        )
        # First snapshot should be yellow for South
        assert seq[0].phase == IntersectionPhase.NORMAL_YELLOW
        assert seq[0].south == SignalState.YELLOW

    def test_invalid_direction_raises(self):
        """Invalid direction should raise ValueError."""
        with pytest.raises(ValueError):
            self.controller.generate_emergency_preemption(
                emergency_direction="Northeast",
                current_green_direction="East",
            )

    # ── Recovery tests ──────────────────────────────────────────────

    def test_post_emergency_recovery(self):
        """After emergency, should transition through yellow/all-red to normal."""
        recovery = self.controller.generate_post_emergency_recovery("North")
        phases = [s.phase for s in recovery]
        assert IntersectionPhase.EMERGENCY_YELLOW in phases
        assert IntersectionPhase.ALL_RED_POST_EMERGENCY in phases
        assert IntersectionPhase.NORMAL_GREEN in phases

    def test_recovery_no_conflicting_greens(self):
        """Recovery sequence must not have conflicting greens."""
        for direction in DIRECTIONS:
            recovery = self.controller.generate_post_emergency_recovery(direction)
            assert self.controller.validate_no_conflicting_greens(recovery)

    # ── Full sequence tests ─────────────────────────────────────────

    def test_full_preemption_and_recovery(self):
        """Full preemption + recovery should be conflict-free."""
        for emg_dir in DIRECTIONS:
            for cur_dir in DIRECTIONS:
                preempt = self.controller.generate_emergency_preemption(
                    emergency_direction=emg_dir,
                    current_green_direction=cur_dir,
                )
                recovery = self.controller.generate_post_emergency_recovery(emg_dir)
                full = preempt + recovery
                assert self.controller.validate_no_conflicting_greens(full)


class TestSignalSnapshot:
    """Tests for the SignalSnapshot data class."""

    def test_conflicting_greens_detection(self):
        """Snapshot with two greens should be flagged as conflicting."""
        snap = SignalSnapshot(
            north=SignalState.GREEN,
            south=SignalState.GREEN,
            east=SignalState.RED,
            west=SignalState.RED,
            phase=IntersectionPhase.NORMAL_GREEN,
            active_direction="North",
            description="test",
            duration=30,
        )
        assert snap.has_conflicting_greens()

    def test_no_conflict_single_green(self):
        """Snapshot with one green should not be flagged."""
        snap = SignalSnapshot(
            north=SignalState.GREEN,
            south=SignalState.RED,
            east=SignalState.RED,
            west=SignalState.RED,
            phase=IntersectionPhase.NORMAL_GREEN,
            active_direction="North",
            description="test",
            duration=30,
        )
        assert not snap.has_conflicting_greens()

    def test_as_dict(self):
        """Snapshot should serialize to dict correctly."""
        snap = SignalSnapshot(
            north=SignalState.GREEN,
            south=SignalState.RED,
            east=SignalState.RED,
            west=SignalState.RED,
            phase=IntersectionPhase.EMERGENCY_GREEN,
            active_direction="North",
            description="test",
            duration=45,
        )
        d = snap.as_dict()
        assert d["north"] == "GREEN"
        assert d["south"] == "RED"
        assert d["phase"] == "EMERGENCY_GREEN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
