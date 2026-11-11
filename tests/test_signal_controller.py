"""
Unit tests for Four-Way Traffic Signal Controller (Teammate 3 - Phase 5).
Covers normal cycles, safe transitions, emergency preemption, clearance,
conflicting emergency handling, and module-level APIs.
"""

import pytest

from src.signal_controller import (
    DEFAULT_APPROACHES,
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


@pytest.fixture
def controller():
    """Create a controller with easy test timings: 10s green, 3s yellow, 2s all-red, 15s emergency green."""
    return TrafficSignalController(
        green_duration=10.0,
        yellow_duration=3.0,
        all_red_duration=2.0,
        emergency_green_duration=15.0,
        initial_approach="NORTH",
    )


# -----------------------------------------------------------------------------
# Test 1: Normal traffic cycle
# -----------------------------------------------------------------------------
def test_normal_traffic_cycle(controller):
    """Verify normal traffic progression: GREEN -> YELLOW -> ALL_RED -> next GREEN."""
    # Start at NORTH GREEN
    state = controller.get_signal_state()
    assert state == {"NORTH": "GREEN", "EAST": "RED", "SOUTH": "RED", "WEST": "RED"}
    assert controller.phase == SignalPhase.GREEN
    assert controller.current_approach == "NORTH"

    # Advance 10s: NORTH GREEN completes and switches to YELLOW
    controller.advance_signal(10.0)
    assert controller.phase == SignalPhase.YELLOW
    assert controller.get_signal_state() == {
        "NORTH": "YELLOW",
        "EAST": "RED",
        "SOUTH": "RED",
        "WEST": "RED",
    }

    # Advance 3s: NORTH YELLOW completes and switches to ALL_RED
    controller.advance_signal(3.0)
    assert controller.phase == SignalPhase.ALL_RED
    assert controller.get_signal_state() == {
        "NORTH": "RED",
        "EAST": "RED",
        "SOUTH": "RED",
        "WEST": "RED",
    }

    # Advance 2s: ALL_RED completes and switches to next approach GREEN (EAST)
    controller.advance_signal(2.0)
    assert controller.phase == SignalPhase.GREEN
    assert controller.current_approach == "EAST"
    assert controller.get_signal_state() == {
        "NORTH": "RED",
        "EAST": "GREEN",
        "SOUTH": "RED",
        "WEST": "RED",
    }


# -----------------------------------------------------------------------------
# Test 2: All four directions receive green in sequence
# -----------------------------------------------------------------------------
def test_all_four_directions_cycle(controller):
    """Verify round-robin cycle over all four approaches: NORTH -> EAST -> SOUTH -> WEST -> NORTH."""
    cycle_order = ["NORTH", "EAST", "SOUTH", "WEST"]
    full_phase_time = 10.0 + 3.0 + 2.0  # 15s per approach

    for expected_approach in cycle_order:
        assert controller.current_approach == expected_approach
        assert controller.phase == SignalPhase.GREEN
        assert controller.get_signal_state()[expected_approach] == "GREEN"
        controller.advance_signal(full_phase_time)

    # After full cycle, return to NORTH GREEN
    assert controller.current_approach == "NORTH"
    assert controller.phase == SignalPhase.GREEN
    assert controller.get_signal_state()["NORTH"] == "GREEN"


# -----------------------------------------------------------------------------
# Test 11: GREEN-to-YELLOW transition
# -----------------------------------------------------------------------------
def test_green_to_yellow_transition(controller):
    """Ensure GREEN never switches directly to RED or another GREEN; must transition to YELLOW."""
    # Under normal operation
    controller.advance_signal(9.9)
    assert controller.phase == SignalPhase.GREEN
    controller.advance_signal(0.2)
    assert controller.phase == SignalPhase.YELLOW
    assert controller.get_signal_state()["NORTH"] == "YELLOW"

    # Under preemption while GREEN
    ctrl2 = TrafficSignalController()
    assert ctrl2.phase == SignalPhase.GREEN
    # Emergency requested for SOUTH
    res = ctrl2.request_emergency_preemption("SOUTH", "ambulance", 0.95)
    assert res["status"] == "transitioning"
    assert ctrl2.phase == SignalPhase.YELLOW
    assert ctrl2.get_signal_state()["NORTH"] == "YELLOW"


# -----------------------------------------------------------------------------
# Test 12: YELLOW-to-all-RED transition
# -----------------------------------------------------------------------------
def test_yellow_to_all_red_transition(controller):
    """Ensure YELLOW transition always leads to ALL_RED clearance before any new GREEN."""
    controller.advance_signal(10.0)  # Enter YELLOW
    assert controller.phase == SignalPhase.YELLOW
    assert controller.get_signal_state()["NORTH"] == "YELLOW"

    controller.advance_signal(3.0)  # Complete YELLOW
    assert controller.phase == SignalPhase.ALL_RED
    # All signals must be RED
    state = controller.get_signal_state()
    for app in DEFAULT_APPROACHES:
        assert state[app] == "RED"


# -----------------------------------------------------------------------------
# Test 13: All-RED-to-requested-GREEN transition
# -----------------------------------------------------------------------------
def test_all_red_to_requested_green_transition(controller):
    """Ensure that after all-red clearance, the requested emergency approach turns GREEN."""
    # NORTH is GREEN, preemption requested for WEST
    controller.request_emergency_preemption("WEST", "ambulance", 0.95)
    assert controller.phase == SignalPhase.YELLOW

    # Advance through yellow
    controller.advance_signal(3.0)
    assert controller.phase == SignalPhase.ALL_RED

    # Advance through all-red clearance
    controller.advance_signal(2.0)
    assert controller.phase == SignalPhase.GREEN
    assert controller.current_approach == "WEST"
    assert controller.mode == ControllerMode.EMERGENCY
    assert controller.get_signal_state()["WEST"] == "GREEN"
    assert controller.get_signal_state()["NORTH"] == "RED"
    assert controller.get_signal_state()["EAST"] == "RED"
    assert controller.get_signal_state()["SOUTH"] == "RED"


# -----------------------------------------------------------------------------
# Test 14: No conflicting GREEN signals (Invariance across 1000 steps)
# -----------------------------------------------------------------------------
def test_no_conflicting_green_signals(controller):
    """Exhaustive check: At NO point should multiple directions be GREEN, or GREEN and YELLOW simultaneously."""
    import random

    rng = random.Random(42)

    for step in range(500):
        # Randomly trigger emergency preemption
        if step % 25 == 0:
            target = rng.choice(list(DEFAULT_APPROACHES))
            etype = rng.choice(["ambulance", "police_car"])
            controller.request_emergency_preemption(target, etype, 0.9)

        # Advance random delta between 0.1s and 4.0s
        dt = rng.uniform(0.1, 4.0)
        state = controller.advance_signal(dt)

        # Invariant 1: At most one approach can be GREEN
        green_count = sum(1 for color in state.values() if color == SignalColor.GREEN.value)
        assert green_count <= 1, f"Conflicting GREEN signals detected at step {step}: {state}"

        # Invariant 2: At most one approach can be YELLOW
        yellow_count = sum(1 for color in state.values() if color == SignalColor.YELLOW.value)
        assert yellow_count <= 1, f"Multiple YELLOW signals detected at step {step}: {state}"

        # Invariant 3: GREEN and YELLOW cannot be active simultaneously on different approaches
        if green_count > 0 and yellow_count > 0:
            pytest.fail(f"GREEN and YELLOW active simultaneously at step {step}: {state}")


# -----------------------------------------------------------------------------
# Test 15: Emergency clearance (timeout and explicit reset)
# -----------------------------------------------------------------------------
def test_emergency_clearance_by_timeout(controller):
    """Emergency GREEN expires after configured duration, transitions to YELLOW, ALL_RED, then normal."""
    controller.request_emergency_preemption("SOUTH", "ambulance", 0.95)
    # Transition to SOUTH GREEN: 3s yellow + 2s all-red = 5s
    controller.advance_signal(5.0)
    assert controller.current_approach == "SOUTH"
    assert controller.phase == SignalPhase.GREEN
    assert controller.mode == ControllerMode.EMERGENCY

    # Advance 15s (emergency green duration): SOUTH becomes YELLOW
    controller.advance_signal(15.0)
    assert controller.phase == SignalPhase.YELLOW
    assert controller.get_signal_state()["SOUTH"] == "YELLOW"

    # Advance 3s yellow + 2s all-red = 5s
    controller.advance_signal(5.0)
    # Resumes normal operation
    assert controller.phase == SignalPhase.GREEN
    assert controller.mode == ControllerMode.NORMAL


def test_emergency_clearance_by_explicit_reset(controller):
    """Explicit clear_emergency() safely transitions away from emergency green via YELLOW and ALL_RED."""
    controller.request_emergency_preemption("SOUTH", "ambulance", 0.95)
    controller.advance_signal(5.0)
    assert controller.current_approach == "SOUTH"
    assert controller.phase == SignalPhase.GREEN
    assert controller.mode == ControllerMode.EMERGENCY

    # Explicit clearance called mid-green
    clear_res = controller.clear_emergency()
    assert clear_res["status"] == "clearing"
    # Current approach transitions to YELLOW first (safety rule)
    assert controller.phase == SignalPhase.YELLOW
    assert controller.get_signal_state()["SOUTH"] == "YELLOW"

    # Complete yellow and all-red
    controller.advance_signal(3.0 + 2.0)
    assert controller.phase == SignalPhase.GREEN
    assert controller.mode == ControllerMode.NORMAL


# -----------------------------------------------------------------------------
# Test 16: Multiple emergency requests (FIFO and conflict resolution)
# -----------------------------------------------------------------------------
def test_multiple_emergency_requests(controller):
    """Handle multiple emergency requests deterministically without conflicting GREENs."""
    # 1. NORTH is GREEN. Emergency for SOUTH arrives.
    controller.request_emergency_preemption("SOUTH", "ambulance", 0.95)
    # Transitions: 3s yellow + 2s all-red = 5s -> SOUTH GREEN
    controller.advance_signal(5.0)
    assert controller.current_approach == "SOUTH"
    assert controller.phase == SignalPhase.GREEN
    assert controller.mode == ControllerMode.EMERGENCY

    # 2. While SOUTH is active, WEST and EAST emergencies arrive
    res_west = controller.request_emergency_preemption("WEST", "police_car", 0.92)
    assert res_west["status"] == "queued"
    assert len(controller.emergency_queue) == 1

    res_east = controller.request_emergency_preemption("EAST", "ambulance", 0.89)
    assert res_east["status"] == "queued"
    assert len(controller.emergency_queue) == 2

    # SOUTH remains GREEN (not cut off abruptly)
    assert controller.current_approach == "SOUTH"
    assert controller.get_signal_state()["SOUTH"] == "GREEN"

    # 3. SOUTH completes its emergency green (15s) -> transitions through 3s yellow + 2s all-red = 5s
    controller.advance_signal(15.0 + 5.0)
    # WEST (next in FIFO) must receive GREEN
    assert controller.current_approach == "WEST"
    assert controller.phase == SignalPhase.GREEN
    assert controller.mode == ControllerMode.EMERGENCY
    assert controller.get_signal_state()["WEST"] == "GREEN"
    assert len(controller.emergency_queue) == 1  # EAST remaining

    # 4. WEST completes its emergency green -> transitions to EAST
    controller.advance_signal(15.0 + 5.0)
    assert controller.current_approach == "EAST"
    assert controller.phase == SignalPhase.GREEN
    assert controller.mode == ControllerMode.EMERGENCY
    assert controller.get_signal_state()["EAST"] == "GREEN"
    assert len(controller.emergency_queue) == 0


def test_duplicate_emergency_request_for_same_approach(controller):
    """If preemption is requested for the currently active emergency approach, extend the green."""
    controller.request_emergency_preemption("SOUTH", "ambulance", 0.95)
    controller.advance_signal(5.0)  # Enter SOUTH GREEN
    assert controller.current_approach == "SOUTH"

    controller.advance_signal(10.0)  # 10s elapsed out of 15s
    assert controller.time_in_phase == 10.0

    # Repeat detection for SOUTH (e.g. video stream frame)
    res = controller.request_emergency_preemption("SOUTH", "ambulance", 0.97)
    assert res["status"] == "active"
    # Timer refreshed
    assert controller.time_in_phase == 0.0


# -----------------------------------------------------------------------------
# Test 17: Return to normal operation
# -----------------------------------------------------------------------------
def test_return_to_normal_operation_after_emergency(controller):
    """Verify that after all emergency preemptions conclude, normal cycling resumes seamlessly."""
    # Preempt for EAST
    controller.request_emergency_preemption("EAST", "ambulance", 0.95)
    controller.advance_signal(5.0)  # 3s yellow + 2s all-red
    assert controller.current_approach == "EAST"
    assert controller.mode == ControllerMode.EMERGENCY

    # Complete emergency green (15s) + transition (5s)
    controller.advance_signal(20.0)
    # Mode is now NORMAL
    assert controller.mode == ControllerMode.NORMAL
    assert controller.phase == SignalPhase.GREEN

    # Controller continues cycling through normal approaches
    approach_1 = controller.current_approach
    controller.advance_signal(15.0)  # Full phase
    approach_2 = controller.current_approach
    assert approach_1 != approach_2
    assert controller.mode == ControllerMode.NORMAL


# -----------------------------------------------------------------------------
# Module-level API and Functional Interface Tests
# -----------------------------------------------------------------------------
def test_module_level_interface():
    """Verify the preferred functional API required by Requirement 8."""
    reset_controller(green_duration=8.0, yellow_duration=2.0, all_red_duration=1.0)

    # Initial state
    state = get_signal_state()
    assert state == {"NORTH": "GREEN", "EAST": "RED", "SOUTH": "RED", "WEST": "RED"}

    # Clock advance
    new_state = advance_signal(8.0)
    assert new_state["NORTH"] == "YELLOW"

    # Preemption request
    preempt_res = request_emergency_preemption("SOUTH", "ambulance", 0.95)
    assert preempt_res["status"] in ("transitioning", "queued", "active")

    # Advance to SOUTH GREEN
    advance_signal(2.0 + 1.0)
    assert get_signal_state()["SOUTH"] == "GREEN"

    # Clear emergency
    clear_res = clear_emergency()
    assert clear_res["status"] in ("clearing", "cleared")


def test_process_and_apply_decision_pipeline():
    """Verify the convenience pipeline: inference_result -> decision -> signal update."""
    reset_controller()

    sample_inference = {
        "camera": "EAST",
        "emergency_detected": True,
        "emergency_type": "ambulance",
        "confidence": 0.96,
        "detections": [],
        "annotated_image_path": "output/annotated.jpg",
        "status": "success",
    }

    result = process_and_apply_decision(sample_inference)
    assert result["decision"]["preemption_requested"] is True
    assert result["preemption"]["status"] == "transitioning"
    assert result["signal_state"]["NORTH"] == "YELLOW"


def test_detailed_telemetry(controller):
    """Verify detailed telemetry dictionary for frontend widgets."""
    telemetry = controller.get_detailed_state()
    assert "signals" in telemetry
    assert "current_phase" in telemetry
    assert "mode" in telemetry
    assert "time_remaining" in telemetry
    assert "total_time_elapsed" in telemetry
