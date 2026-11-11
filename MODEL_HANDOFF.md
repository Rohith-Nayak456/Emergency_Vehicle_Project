# Model Handoff — Teammate 1 → All Teammates

## Model Details

| Property | Value |
|---|---|
| Architecture | YOLO11n (Ultralytics) |
| File | `models/best.pt` |
| File size | ~5.3 MB |
| Parameters | ~2.6M |
| Training framework | Ultralytics 8.4.x |
| Training hardware | Google Colab T4 GPU |
| Inference hardware | CPU (Intel i5-1135G7 tested) |

## Class Mapping (DO NOT CHANGE)

```python
CLASS_NAMES = {
    0: "car",
    1: "motorcycle",
    2: "autorickshaw",
    3: "truck",
    4: "bus",
    5: "bicycle",
    6: "ambulance",      # EMERGENCY
    7: "police_car",     # EMERGENCY
}
EMERGENCY_CLASS_IDS = {6, 7}
```

## How to Load the Model

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")
print(model.names)  # Should print the 8-class dict above
```

## How to Run Inference

```python
results = model.predict(
    source="path/to/image.png",
    imgsz=640,
    conf=0.25,       # Detection confidence threshold
    verbose=False,
)

# Access detections
for box in results[0].boxes:
    class_id = int(box.cls)
    class_name = model.names[class_id]
    confidence = float(box.conf)
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    is_emergency = class_id in {6, 7}
```

## Agreed Inference Interface (Teammate 2)

```python
def run_inference(image_path: str, camera_direction: str) -> dict:
    """
    Returns:
    {
        "camera": "NORTH",
        "emergency_detected": True,
        "emergency_type": "ambulance",
        "confidence": 0.94,
        "detections": [
            {
                "class_id": 6,
                "class_name": "ambulance",
                "confidence": 0.94,
                "bbox": [100, 120, 350, 280],
                "is_emergency": True
            }
        ],
        "annotated_image_path": "path/to/annotated.jpg",
        "status": "success"
    }
    """
```

When no emergency is detected:
- `emergency_detected = False`
- `emergency_type = None`
- `confidence = 0.0`

If multiple emergency vehicles are detected, use the highest-confidence one for the summary fields.

## Model Performance

### Test Set (1,491 images)

| Metric | Value |
|---|---|
| mAP@0.5 | 60.90% |
| mAP@0.5:0.95 | 48.67% |
| Ambulance mAP@0.5 | 35.4% |
| Police car mAP@0.5 | 17.3% |

### Emergency Class Limitations

- **Ambulance recall: 31.2%** — model misses ~69% of ambulances
- **Police car recall: 12.5%** — model misses ~88% of police cars
- Caused by limited unique training scenes (94 ambulance, 40 police car)
- Do NOT claim high emergency detection reliability in the frontend or documentation

## Confidence Threshold Guidance

| Use Case | Recommended Threshold |
|---|---|
| General detection display | 0.25 |
| Emergency preemption decision | 0.40+ |

The decision engine should apply a higher threshold for emergency preemption to reduce false positives. Uncertain detections should NOT automatically trigger signal preemption.

## Dependencies

```
ultralytics>=8.0.0
torch>=2.0.0
opencv-python>=4.0.0
numpy
Pillow
```

See `requirements.txt` for the full list.

## Notes for Each Teammate

**Teammate 2 (Inference):** Use `YOLO("models/best.pt")` directly. Implement `run_inference()` per the interface above. The model returns bounding boxes in xyxy format.

**Teammate 3 (Decision Engine + Signal Controller):** Only class IDs 6 and 7 are emergency. Apply the emergency confidence threshold (0.40) before triggering preemption.

**Teammate 4 (Frontend):** Load model path from `config/settings.py` → `BEST_MODEL_PATH`. Display confidence scores. Highlight emergency vehicles in red.

**Teammate 5 (Testing + Docs):** Use the actual test metrics above. Do not fabricate better numbers. Acknowledge the emergency detection limitations.
