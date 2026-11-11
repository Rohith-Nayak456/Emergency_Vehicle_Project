"""
Four-way intersection signal controller with emergency preemption.

This module implements a state machine for a simulated four-way
traffic intersection. It supports normal cycling and emergency
preemption with proper yellow and all-red clearance phases.

IMPORTANT: This is a simulation for academic demonstration purposes.
It does NOT control real traffic signals.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import time


class SignalState(Enum):
    """Possible states for a single direction's signal."""
    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


class IntersectionPhase(Enum):
    """Phase of the overall intersection state machine."""
    NORMAL_GREEN = "NORMAL_GREEN"
    NORMAL_YELLOW = "NORMAL_YELLOW"
    ALL_RED_CLEARANCE = "ALL_RED_CLEARANCE"
    EMERGENCY_GREEN = "EMERGENCY_GREEN"
    EMERGENCY_YELLOW = "EMERGENCY_YELLOW"
    ALL_RED_POST_EMERGENCY = "ALL_RED_POST_EMERGENCY"


DIRECTIONS = ["North", "South", "East", "West"]

# Default phase durations (seconds) — configurable
DEFAULT_DURATIONS = {
    "normal_green": 30,
    "normal_yellow": 5,
    "all_red_clearance": 3,
    "emergency_green": 45,
    "emergency_yellow": 5,
}


@dataclass
class SignalSnapshot:
    """Immutable snapshot of all four signals at a point in time."""
    north: SignalState
    south: SignalState
    east: SignalState
    west: SignalState
    phase: IntersectionPhase
    active_direction: Optional[str]
    description: str
    duration: float  # seconds this state lasts

    def as_dict(self) -> dict:
        return {
            "north": self.north.value,
            "south": self.south.value,
            "east": self.east.value,
            "west": self.west.value,
            "phase": self.phase.value,
            "active_direction": self.active_direction,
            "description": self.description,
            "duration": self.duration,
        }

    def has_conflicting_greens(self) -> bool:
        """Check safety: no two directions should be green simultaneously."""
        greens = sum(1 for s in [self.north, self.south, self.east, self.west]
                     if s == SignalState.GREEN)
        return greens > 1

    def get_signal(self, direction: str) -> SignalState:
        """Get signal state for a direction."""
        return getattr(self, direction.lower())


class SignalController:
    """
    Four-way intersection signal controller with emergency preemption.

    State machine transitions:

    Normal cycle:
        NORMAL_GREEN(dir) -> NORMAL_YELLOW(dir) -> ALL_RED -> NORMAL_GREEN(next_dir)

    Emergency preemption:
        current_phase -> NORMAL_YELLOW(current_green) -> ALL_RED_CLEARANCE
        -> EMERGENCY_GREEN(emergency_dir) -> EMERGENCY_YELLOW(emergency_dir)
        -> ALL_RED_POST_EMERGENCY -> NORMAL_GREEN(next_dir)
    """

    def __init__(self, durations: Optional[dict] = None):
        self.durations = {**DEFAULT_DURATIONS, **(durations or {})}
        self._current_green_index = 0  # Index into DIRECTIONS
        self._phase = IntersectionPhase.NORMAL_GREEN

    @property
    def current_green_direction(self) -> str:
        return DIRECTIONS[self._current_green_index]

    def _next_direction_index(self, current: int) -> int:
        return (current + 1) % len(DIRECTIONS)

    def _make_snapshot(
        self,
        green_dir: Optional[str],
        yellow_dir: Optional[str],
        phase: IntersectionPhase,
        description: str,
        duration: float,
    ) -> SignalSnapshot:
        """Create a signal snapshot. All directions default to RED."""
        signals = {d.lower(): SignalState.RED for d in DIRECTIONS}
        if green_dir:
            signals[green_dir.lower()] = SignalState.GREEN
        if yellow_dir:
            signals[yellow_dir.lower()] = SignalState.YELLOW

        active = green_dir or yellow_dir

        return SignalSnapshot(
            north=signals["north"],
            south=signals["south"],
            east=signals["east"],
            west=signals["west"],
            phase=phase,
            active_direction=active,
            description=description,
            duration=duration,
        )

    def get_normal_cycle_snapshot(self) -> SignalSnapshot:
        """Get the current normal-cycle snapshot."""
        green_dir = self.current_green_direction
        return self._make_snapshot(
            green_dir=green_dir,
            yellow_dir=None,
            phase=IntersectionPhase.NORMAL_GREEN,
            description=f"Normal operation: {green_dir} has green signal.",
            duration=self.durations["normal_green"],
        )

    def generate_normal_cycle(self) -> list[SignalSnapshot]:
        """Generate one full normal cycle through all four directions."""
        snapshots = []
        for i in range(len(DIRECTIONS)):
            direction = DIRECTIONS[i]
            # Green phase
            snapshots.append(self._make_snapshot(
                green_dir=direction,
                yellow_dir=None,
                phase=IntersectionPhase.NORMAL_GREEN,
                description=f"Normal operation: {direction} has green signal.",
                duration=self.durations["normal_green"],
            ))
            # Yellow phase
            snapshots.append(self._make_snapshot(
                green_dir=None,
                yellow_dir=direction,
                phase=IntersectionPhase.NORMAL_YELLOW,
                description=f"{direction} transitioning to red (yellow clearance).",
                duration=self.durations["normal_yellow"],
            ))
            # All-red
            snapshots.append(self._make_snapshot(
                green_dir=None,
                yellow_dir=None,
                phase=IntersectionPhase.ALL_RED_CLEARANCE,
                description="All-red clearance phase.",
                duration=self.durations["all_red_clearance"],
            ))
        return snapshots

    def generate_emergency_preemption(
        self,
        emergency_direction: str,
        current_green_direction: Optional[str] = None,
    ) -> list[SignalSnapshot]:
        """
        Generate the full sequence of signal transitions for emergency
        preemption from the given direction.

        Args:
            emergency_direction: The approach direction of the emergency vehicle.
            current_green_direction: Which direction currently has green.
                                     If None, uses the controller's internal state.

        Returns:
            List of SignalSnapshot objects representing the transition sequence.

        Raises:
            ValueError: If emergency_direction is not valid.
        """
        if emergency_direction not in DIRECTIONS:
            raise ValueError(
                f"Invalid direction '{emergency_direction}'. "
                f"Must be one of {DIRECTIONS}."
            )

        current_green = current_green_direction or self.current_green_direction
        if current_green not in DIRECTIONS:
            raise ValueError(f"Invalid current green direction: {current_green}")

        snapshots = []

        # Step 1: If emergency direction already has green, just extend it
        if current_green == emergency_direction:
            snapshots.append(self._make_snapshot(
                green_dir=emergency_direction,
                yellow_dir=None,
                phase=IntersectionPhase.EMERGENCY_GREEN,
                description=(
                    f"Emergency vehicle approaching from {emergency_direction}. "
                    f"{emergency_direction} already has green — extending for "
                    f"emergency priority."
                ),
                duration=self.durations["emergency_green"],
            ))
            return snapshots

        # Step 2: Current green direction -> Yellow
        snapshots.append(self._make_snapshot(
            green_dir=None,
            yellow_dir=current_green,
            phase=IntersectionPhase.NORMAL_YELLOW,
            description=(
                f"Emergency preemption initiated. "
                f"{current_green} transitioning to red (yellow clearance)."
            ),
            duration=self.durations["normal_yellow"],
        ))

        # Step 3: All-red clearance
        snapshots.append(self._make_snapshot(
            green_dir=None,
            yellow_dir=None,
            phase=IntersectionPhase.ALL_RED_CLEARANCE,
            description=(
                "All-red clearance phase. All directions red before "
                "activating emergency green."
            ),
            duration=self.durations["all_red_clearance"],
        ))

        # Step 4: Emergency green
        snapshots.append(self._make_snapshot(
            green_dir=emergency_direction,
            yellow_dir=None,
            phase=IntersectionPhase.EMERGENCY_GREEN,
            description=(
                f"EMERGENCY GREEN: {emergency_direction} receives priority green. "
                f"All other directions remain red."
            ),
            duration=self.durations["emergency_green"],
        ))

        return snapshots

    def generate_post_emergency_recovery(
        self, emergency_direction: str
    ) -> list[SignalSnapshot]:
        """
        Generate the recovery sequence after emergency preemption ends.
        Transitions from emergency green back to normal cycle.
        """
        if emergency_direction not in DIRECTIONS:
            raise ValueError(f"Invalid direction: {emergency_direction}")

        snapshots = []

        # Emergency yellow
        snapshots.append(self._make_snapshot(
            green_dir=None,
            yellow_dir=emergency_direction,
            phase=IntersectionPhase.EMERGENCY_YELLOW,
            description=(
                f"Emergency phase ending. {emergency_direction} "
                f"transitioning to red (yellow clearance)."
            ),
            duration=self.durations["emergency_yellow"],
        ))

        # All-red post-emergency
        snapshots.append(self._make_snapshot(
            green_dir=None,
            yellow_dir=None,
            phase=IntersectionPhase.ALL_RED_POST_EMERGENCY,
            description="All-red clearance after emergency phase. Returning to normal operation.",
            duration=self.durations["all_red_clearance"],
        ))

        # Resume normal (next direction in cycle after emergency direction)
        emergency_idx = DIRECTIONS.index(emergency_direction)
        next_idx = self._next_direction_index(emergency_idx)
        next_dir = DIRECTIONS[next_idx]

        snapshots.append(self._make_snapshot(
            green_dir=next_dir,
            yellow_dir=None,
            phase=IntersectionPhase.NORMAL_GREEN,
            description=f"Normal operation resumed. {next_dir} has green signal.",
            duration=self.durations["normal_green"],
        ))

        return snapshots

    def validate_no_conflicting_greens(
        self, snapshots: list[SignalSnapshot]
    ) -> bool:
        """Verify that no snapshot has two directions green simultaneously."""
        return all(not s.has_conflicting_greens() for s in snapshots)
