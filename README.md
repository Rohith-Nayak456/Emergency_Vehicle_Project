# Emergency_Vehicle_Project
# Emergency Vehicle Decision Engine & Four-Way Signal Controller — Phase 5

This package contains the Teammate 3 Emergency Decision Engine and Four-Way Traffic Signal Controller module.

## Scope and limitations
- Controls safety-critical signal state transitions and emergency preemption for a 4-way intersection (`NORTH`, `SOUTH`, `EAST`, `WEST`).
- Processes validated inference payload dictionaries containing emergency types (`ambulance` and `police_car`).
- Applies strict confidence threshold checks to filter out false or weak inference detections.
- Operates on a fail-safe rule: low-confidence or malformed inputs will never trigger a green signal preemption.
- Manages real-time queueing for concurrent or overlapping emergency requests.
- This module relies on validated inference dictionaries provided by Teammate 2's inference pipeline.

## Signal State Transitions
| State | Duration / Condition | Description |
| :--- | :--- | :--- |
| `GREEN` | Normal Cycle / Active Emergency | Allows traffic clearance for designated direction. |
| `YELLOW` | Transition Phase | Mandatory clearance buffer preceding signal changes. |
| `ALL_RED` | Safety Phase | All directions held red before switching green direction. |

## Windows setup
Run from the project root:
```cmd
py -3.14 -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
