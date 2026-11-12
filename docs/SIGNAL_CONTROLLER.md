# Emergency Decision Engine and Four-Way Traffic Signal Controller

**Phase 5 Implementation — Teammate 3**  
Project: *Emergency Vehicle Detection and Adaptive Signal Preemption for Heterogeneous Traffic Conditions*

---

## 1. Overview

This module provides two core systems:
1. **Emergency Decision Engine (`src.decision_engine`)**: Evaluates YOLO inference results received from Teammate 2 against safety, class, camera, and confidence criteria to decide whether signal preemption is justified.
2. **Four-Way Signal Controller (`src.signal_controller`)**: Simulates a 4-way intersection (NORTH, EAST, SOUTH, WEST) with round-robin normal cycling, safe multi-phase transitions, deterministic clock advancement, and adaptive emergency vehicle preemption.

---

## 2. Shared Data Contract

### Inference Input (from Teammate 2)

```json
{
  "camera": "NORTH",
  "emergency_detected": true,
  "emergency_type": "ambulance",
  "confidence": 0.94,
  "detections": [],
  "annotated_image_path": "output/annotated_sample.jpg",
  "status": "success"
}
```

### Decision Output

```json
{
  "preemption_requested": true,
  "approach": "NORTH",
  "emergency_type": "ambulance",
  "confidence": 0.94,
  "reason": "Valid ambulance detection above threshold (0.94 >= 0.50)"
}
```

If validation fails or confidence is below the configurable threshold:

```json
{
  "preemption_requested": false,
  "approach": "NORTH",
  "emergency_type": null,
  "confidence": 0.35,
  "reason": "Confidence 0.35 is below threshold 0.50"
}
```

---

## 3. Decision Engine Validation Rules

The function `process_emergency_decision(inference_result, confidence_threshold=0.50)` enforces 7 strict fail-safe validation steps:

1. **Payload Structure**: Verifies the input is a valid dictionary.
2. **Required Fields**: Ensures `status`, `camera`, `emergency_detected`, `emergency_type`, and `confidence` are present.
3. **Inference Status**: Must equal `"success"`.
4. **Camera Direction**: Strips and upper-cases direction, verifying membership in `{"NORTH", "SOUTH", "EAST", "WEST"}`.
5. **Emergency Flag**: `emergency_detected` must evaluate to `True`.
6. **Class Validation**: `emergency_type` must be an authorized emergency class (`"ambulance"` or `"police_car"`).
7. **Confidence Threshold**: Confidence must be a valid number $\ge \text{threshold}$ (default `0.50`, configurable).

---

## 4. Signal Controller Architecture

### Approaches and States
- **Approaches**: `NORTH`, `EAST`, `SOUTH`, `WEST`
- **Signal Colors**: `RED`, `YELLOW`, `GREEN`
- **Controller Modes**: `NORMAL`, `EMERGENCY`
- **Internal Phases**: `GREEN`, `YELLOW`, `ALL_RED`

### Default Configurable Timings
- `green_duration`: 10.0 seconds
- `yellow_duration`: 3.0 seconds
- `all_red_duration`: 2.0 seconds
- `emergency_green_duration`: 15.0 seconds

### Safe Transition Sequence
Conflicting GREEN signals are strictly prohibited. Signals never switch directly from GREEN to GREEN. Every approach change executes the following mandatory sequence:

$$\text{Active Approach GREEN} \xrightarrow{\text{yellow\_duration}} \text{Active Approach YELLOW} \xrightarrow{\text{all\_red\_duration}} \text{ALL RED CLEARANCE} \xrightarrow{} \text{Next Approach GREEN}$$

---

## 5. Multiple Emergency Requests Policy

When an emergency preemption is requested while another preemption is already active or transitioning:

1. **Active Emergency Retention**: The currently active emergency GREEN is **never cut off abruptly**. Drivers in the intersection are protected from collision.
2. **FIFO Queue**: Conflicting emergency requests for different approaches are added to a First-In-First-Out (`emergency_queue`).
3. **Deduplication**: If repeated frames detect an emergency vehicle on the same approach:
   - If that approach is currently active, its green window is refreshed to full duration.
   - If that approach is queued, its confidence score is updated without duplicate queue entries.
4. **Safe Sequential Preemption**: Upon completion or clearance of the active emergency, the controller executes standard safety clearance (YELLOW $\to$ ALL_RED) before servicing the next queued approach.
5. **Conflict Elimination**: Only one approach can be GREEN or YELLOW at any point in simulation time.

---

## 6. Emergency Clearance

Emergency GREEN is non-permanent and bounded by:
- **Timer Expiry**: Automatically concludes after `emergency_green_duration` seconds.
- **Manual / Frontend Reset**: Explicit call to `clear_emergency()`.

Upon clearance:
1. Active approach enters `YELLOW` (for `yellow_duration`).
2. Intersection enters `ALL_RED` (for `all_red_duration`).
3. Controller resumes normal round-robin cycle from the next scheduled approach (preventing starvation).

---

## 7. Frontend API (Integration Guide for Teammate 4)

Teammate 4 (Streamlit frontend) can import and interact with the controller without dealing with internal state machines:

```python
from src import (
    get_signal_state,
    advance_signal,
    request_emergency_preemption,
    clear_emergency,
    process_and_apply_decision,
)

# 1. Get current signal display
signals = get_signal_state()
# => {"NORTH": "GREEN", "SOUTH": "RED", "EAST": "RED", "WEST": "RED"}

# 2. Advance simulation clock (e.g., in a Streamlit update loop)
signals = advance_signal(delta_time=1.0)

# 3. Automatically process inference output and preempt if valid
result = process_and_apply_decision(inference_output)

# 4. Clear emergency when vehicle clears intersection
clear_emergency()
```

---

## 8. Deterministic Testing & Commands

All 17 verification scenarios run deterministically using virtual simulation time advancement (no `time.sleep()`), executing in under 0.2 seconds.

### Test Command
```bash
python -m pytest -v
```

### Coverage
- Normal 4-way traffic cycle & all four directions
- Ambulance & police car detections
- Low-confidence rejection & custom thresholds
- Normal vehicle fail-safe rejection
- Invalid direction, class, status, & missing fields fail-safes
- Phase transitions: GREEN $\to$ YELLOW, YELLOW $\to$ ALL_RED, ALL_RED $\to$ GREEN
- Conflicting green signal invariance across hundreds of randomized ticks
- Emergency clearance via timeout and manual reset
- Multiple concurrent and duplicate emergency requests
- Safe return to normal traffic operation
