# Emergency Vehicle Detection and Adaptive Signal Preemption
## Phase 4: Model Inference and Prediction Module

This module implements Phase 4 (Model Inference and Prediction) developed by **Teammate 2**. It provides an independent, robust inference architecture for single still-image processing, multi-class detection, emergency vehicle preemption flagging, and high-visibility annotated image generation.

---

## 1. Overview & Class Taxonomy

The YOLO object detection model is designed to detect 8 classes within heterogeneous traffic conditions:

| Class ID | Class Name | Category | Preemption Flag (`is_emergency`) |
| :---: | :---: | :---: | :---: |
| `0` | `car` | Standard Traffic | `False` |
| `1` | `motorcycle` | Standard Traffic | `False` |
| `2` | `autorickshaw` | Standard Traffic | `False` |
| `3` | `truck` | Standard Traffic | `False` |
| `4` | `bus` | Standard Traffic | `False` |
| `5` | `bicycle` | Standard Traffic | `False` |
| `6` | `ambulance` | **Emergency** | **`True`** |
| `7` | `police_car` | **Emergency** | **`True`** |

### Permitted Camera Directions
- `NORTH`
- `SOUTH`
- `EAST`
- `WEST`

---

## 2. Required Inference Interface

The primary interface is `run_inference(image_path, camera_direction)`:

```python
from inference import run_inference

result = run_inference(
    image_path="sample_traffic.jpg",
    camera_direction="NORTH"
)
```

### Standard Output Contract (`status == "success"`)

When an emergency vehicle is detected:
```json
{
  "camera": "NORTH",
  "emergency_detected": true,
  "emergency_type": "ambulance",
  "confidence": 0.94,
  "detections": [
    {
      "class_id": 6,
      "class_name": "ambulance",
      "confidence": 0.94,
      "bbox": [100, 120, 350, 280],
      "is_emergency": true
    },
    {
      "class_id": 0,
      "class_name": "car",
      "confidence": 0.88,
      "bbox": [370, 140, 510, 260],
      "is_emergency": false
    }
  ],
  "annotated_image_path": "output/annotated/annotated_sample_traffic_20261111_123456_a1b2c3.jpg",
  "status": "success"
}
```

When no emergency vehicle is detected:
```json
{
  "camera": "NORTH",
  "emergency_detected": false,
  "emergency_type": null,
  "confidence": 0.0,
  "detections": [
    {
      "class_id": 0,
      "class_name": "car",
      "confidence": 0.88,
      "bbox": [370, 140, 510, 260],
      "is_emergency": false
    }
  ],
  "annotated_image_path": "output/annotated/annotated_sample_traffic_20261111_123456_a1b2c3.jpg",
  "status": "success"
}
```

### Multiple Emergency Vehicles
If multiple emergency vehicles are detected in the same frame (e.g., both an ambulance and a police car), the engine selects the **highest-confidence** emergency detection for the summary fields (`emergency_type` and `confidence`), while retaining all individual detections in the `detections` list.

### Documented Error Contract (`status == "error"`)
If image reading fails, camera direction is invalid, or weights cannot be loaded, inference does NOT silently report a normal vehicle result:
```json
{
  "camera": "INVALID_DIR",
  "emergency_detected": false,
  "emergency_type": null,
  "confidence": 0.0,
  "detections": [],
  "annotated_image_path": null,
  "status": "error",
  "error_message": "Invalid camera direction 'INVALID_DIR'. Allowed values: NORTH, SOUTH, EAST, WEST"
}
```

---

## 3. Architecture & Key Features

- **Model Loading & Caching**: Models are loaded lazily and cached via `YOLOInferenceEngine` singleton (`get_inference_engine()`), eliminating redundant disk and weight loading per image.
- **CPU Inference Support**: Fully supports CPU execution (`device="cpu"`) without assuming NVIDIA GPU availability.
- **Configurable Relative Paths**: Default weights location is `models/best.pt` relative to the project root, configurable via function arguments or the `YOLO_MODEL_PATH` environment variable.
- **Image Validation**: Rigorous checks for missing files, corrupt image data, and unsupported extensions before executing model inference.
- **Distinct Visual Annotations**: Generates visual bounding boxes with distinct crimson/red styling and emergency banners for `ambulance` and `police_car`, while rendering standard green boxes for normal vehicles. Never overwrites the source image.
- **Streamlit Integration Ready**: Completely decoupled from Streamlit UI code. Callable cleanly from Streamlit frontend without side-effects or circular dependencies.

---

## 4. Project Structure

```
Emergency_Vehicle_Project/
├── .gitignore
├── requirements.txt
├── README.md
├── inference.py              # Root-level entry point function for teammates & Streamlit
├── inference/
│   ├── __init__.py           # Package exports (run_inference, YOLOInferenceEngine, etc.)
│   ├── config.py             # Class mappings, emergency IDs, camera directions, defaults
│   ├── detector.py           # Model lifecycle, prediction engine, summary extraction
│   ├── annotator.py          # Visual annotation generator with distinct styling
│   ├── validator.py          # Input validation for camera directions and image files
│   └── exceptions.py         # Standardized exception hierarchy
├── models/
│   └── .gitkeep              # Placeholder directory for Teammate 1's best.pt
└── tests/
    ├── __init__.py
    ├── conftest.py           # Shared test fixtures & controlled mock predictions
    ├── test_mock_inference.py # 12 required mock & interface unit tests
    └── test_real_inference.py # Actual YOLO tests (executes when best.pt is provided)
```

---

## 5. Setup & Installation

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 6. Running Tests

### A. Run Mock & Interface Unit Tests
These 12 tests verify camera validation, output dictionary structure, vehicle classes, multiple emergencies, error contracts, and image generation using controlled mock prediction objects:
```bash
pytest tests/test_mock_inference.py -v
```

### B. Run Actual YOLO Inference Tests
When Teammate 1 supplies `models/best.pt`:
```bash
pytest tests/test_real_inference.py -v
```
*(Note: Automatically skipped if `models/best.pt` is not present yet).*

---

## 7. Known Limitations

- **Still Image Prototype**: The prototype processes one still image at a time. It cannot calculate vehicle speed, movement direction, ETA, or verify whether an emergency vehicle is approaching or moving away from the intersection.
- **Camera Direction**: Direction is supplied by the user/sensor interface and cannot be inferred purely from pixel content.
- **Live Video**: This phase does not claim live RTSP or video stream support.
