# Emergency Vehicle Detection and Adaptive Signal Preemption

Detection of vehicles at a four-way intersection using a YOLO object detection model, with special focus on identifying emergency vehicles (ambulance, police car) and triggering adaptive traffic signal preemption.

## Project Overview

This system processes uploaded traffic images to:
1. Detect eight vehicle classes using a trained YOLO11n model
2. Identify emergency vehicles (ambulance, police car)
3. Evaluate whether emergency signal preemption is warranted
4. Simulate safe four-way traffic signal transitions
5. Display results through an interactive Streamlit dashboard

**Important:** This is an academic prototype. The system uses uploaded still images, not live feeds. Traffic signals are simulated, not connected to real hardware. Vehicle speed, direction of travel and ETA are not estimated from a single image.

## Dataset

| Split | Images | Labels |
|---|---:|---:|
| Train | 6,836 | 6,836 |
| Validation | 1,412 | 1,412 |
| Test | 1,491 | 1,491 |
| **Total** | **9,739** | **9,739** |

**8 classes:** car, motorcycle, autorickshaw, truck, bus, bicycle, ambulance, police_car

The dataset is not included in this repository due to its size (~4 GB). It is available separately on Google Drive.

## Model

**Architecture:** YOLO11n (Ultralytics) — 2.6M parameters, lightweight for CPU inference

**Location:** `models/best.pt`

### Test Set Performance

| Metric | Value |
|---|---|
| Precision | 62.47% |
| Recall | 55.89% |
| mAP@0.5 | 60.90% |
| mAP@0.5:0.95 | 48.67% |

### Emergency Class Performance (Test Set)

| Class | Precision | Recall | mAP@0.5 |
|---|---|---|---|
| Ambulance | 40.9% | 31.2% | 35.4% |
| Police car | 39.7% | 12.5% | 17.3% |

> **Note:** Emergency class performance is limited by the small number of unique emergency scenes in the dataset (94 ambulance, 40 police car originals). See `reports/TEST_EVALUATION.md` for full details.

## Installation

```bash
# Clone the repository
git clone https://github.com/Rohith-Nayak456/Emergency_Vehicle_Project.git
cd Emergency_Vehicle_Project

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.10+ | PyTorch 2.x | Ultralytics 8.x | Streamlit 1.x

## Quick Start

```bash
# Verify model loads correctly
python -c "from ultralytics import YOLO; m = YOLO('models/best.pt'); print(m.names)"

# Run the Streamlit dashboard
streamlit run app.py

# Run tests
python -m pytest tests/ -v
```

## Project Structure

```
Emergency_Vehicle_Project/
├── app.py                     # Streamlit frontend (Teammate 4)
├── requirements.txt
├── PROJECT_STATUS.md
├── config/
│   ├── data.yaml              # Dataset configuration
│   └── settings.py            # Project-wide constants
├── models/
│   └── best.pt                # Trained YOLO11n weights
├── notebooks/
│   └── model_training.ipynb   # Google Colab training notebook
├── reports/
│   └── TEST_EVALUATION.md     # Model evaluation report
├── results/
│   └── dataset_report/        # Dataset validation reports
├── src/
│   ├── dataset_validation.py  # Dataset quality inspector
│   ├── decision_engine.py     # Emergency preemption logic (Teammate 3)
│   ├── inference.py           # YOLO inference module (Teammate 2)
│   ├── resplit_dataset.py     # Dataset splitting script
│   ├── signal_controller.py   # Four-way signal state machine (Teammate 3)
│   ├── train.py               # Local training script
│   ├── utils.py               # Shared utilities
│   └── verify_split.py        # Split verification script
└── tests/
    ├── test_decision_engine.py
    ├── test_inference.py
    └── test_signal_controller.py
```

## Team Responsibilities

| Teammate | Responsibility |
|---|---|
| 1 (Main) | Dataset, YOLO training, model delivery, integration, GitHub |
| 2 | Inference module (`src/inference.py`) |
| 3 | Decision engine + signal controller |
| 4 | Streamlit frontend (`app.py`) |
| 5 | Testing, documentation, demo preparation |

## Class Mapping

```python
{0: "car", 1: "motorcycle", 2: "autorickshaw", 3: "truck",
 4: "bus", 5: "bicycle", 6: "ambulance", 7: "police_car"}
```

Emergency classes: `ambulance` (ID 6), `police_car` (ID 7)

## Inference Interface

```python
from ultralytics import YOLO

model = YOLO("models/best.pt")
results = model.predict(source="path/to/image.png", imgsz=640, conf=0.25)
```

See `MODEL_HANDOFF.md` for the full integration interface specification.

## Known Limitations

1. Emergency class detection accuracy is limited due to few unique training scenes
2. Police car recall is particularly low (12.5% on test set)
3. Single-image analysis — no tracking, speed or trajectory estimation
4. Simulated traffic signals — not connected to real infrastructure
5. Sensitive to image quality and occlusion

## License

Academic project — not licensed for production use.
