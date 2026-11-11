# Emergency_Vehicle_Project
# Emergency Vehicle Decision Engine & Four-Way Signal Controller — Phase 5

This package contains the Teammate 3 Emergency Decision Engine and Four-Way Traffic Signal Controller module.

## Scope and limitations

* Controls safety-critical signal state transitions and emergency preemption for a 4-way intersection (`NORTH`, `SOUTH`, `EAST`, `WEST`).
* Processes validated inference payload dictionaries containing emergency types (`ambulance` and `police_car`).
* Applies strict confidence threshold checks to filter out false or weak inference detections.
* Operates on a fail-safe rule: low-confidence or malformed inputs will never trigger a green signal preemption.
* Manages real-time queueing for concurrent or overlapping emergency requests.
* This module relies on validated inference dictionaries provided by Teammate 2's inference pipeline.

## Signal State Transitions

| State | Duration / Condition | Description |
| --- | --- | --- |
| `GREEN` | Normal Cycle / Active Emergency | Allows traffic clearance for designated direction. |
| `YELLOW` | Transition Phase | Mandatory clearance buffer preceding signal changes. |
| `ALL_RED` | Safety Phase | All directions held red before switching green direction. |

## Windows setup

Run from the project root:

py -3.14 -m venv .venv
..venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

Dependencies were import-checked with Python 3.14.3. Confirm compatibility with the team's agreed Python version before final integration.

## Usage & API Integration

### 1. Process Emergency Decision

from src.decision_engine import process_emergency_decision

# Validate raw inference output against decision policies

decision = process_emergency_decision(
inference_result={
"status": "success",
"emergency_detected": True,
"emergency_type": "ambulance",
"confidence": 0.92,
"camera": "NORTH"
},
min_confidence=0.85
)
print(decision)

### 2. Request Signal Preemption

from src.signal_controller import SignalController

controller = SignalController()

# Transition signals safely to grant priority passage

status = controller.request_emergency_preemption(
direction="NORTH",
emergency_type="ambulance"
)
print(status)

## Run tests

python -m pytest -v

Tests evaluate deterministic state transitions, threshold validation, edge cases, and safety fail-safes without requiring live camera feeds or external hardware dependencies.

## Output contract

The emergency decision engine returns a validated structure detailing whether preemption is granted, the prioritized camera direction, decision confidence, and rationale. The signal controller guarantees safe sequence transitions (Green -> Yellow -> All-Red -> Green) and maintains a queue for active emergency clearances. Malformed payloads or unverified states immediately reject preemption requests.
