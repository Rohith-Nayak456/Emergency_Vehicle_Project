# Emergency_Vehicle_Project
# Emergency Vehicle Detection — Phase 4 Inference

This package contains the Teammate 2 YOLO inference module.

## Scope and limitations

- Processes one uploaded still image per call.
- Camera direction is manually supplied: `NORTH`, `SOUTH`, `EAST`, or `WEST`.
- Emergency classes: `ambulance` and `police_car`.
- A still image cannot establish speed, motion direction, approach/departure, or ETA.
- This module does not decide signal priority or control traffic lights.
- Actual predictions require Teammate 1's trained `models/best.pt`.

## Class mapping

| ID | Class |
|---:|---|
| 0 | car |
| 1 | motorcycle |
| 2 | autorickshaw |
| 3 | truck |
| 4 | bus |
| 5 | bicycle |
| 6 | ambulance |
| 7 | police_car |

## Windows setup

Run from the project root:

```cmd
py -3.14 -m venv .venv
.\.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Dependencies were import-checked with Python 3.14.3. Confirm compatibility with
the team's agreed Python version before final integration.

## Add model weights

Place the trained model at `models/best.pt`. The module checks that its class
names and IDs match the mapping above.

## Run inference

```python
from src.inference import run_inference

result = run_inference("path/to/traffic.jpg", "NORTH")
print(result)
```

Annotated images are saved under `output/`; the original image is not overwritten.

## Run tests

```cmd
python -m pytest -q
```

Tests use mocked model outputs and do not require `best.pt`. They verify interface
and result handling, not real-world model accuracy.

## Output contract

The returned dictionary contains `camera`, `emergency_detected`, `emergency_type`,
`confidence`, `detections`, `annotated_image_path`, and `status`. If multiple
emergency vehicles are detected, the summary selects the highest-confidence one
while retaining all detections. If none are detected, the emergency summary is
`False`, `None`, and `0.0`. Inference errors raise exceptions and must not be
treated as a successful no-emergency result.
