# Emergency Vehicle Detection and Adaptive Signal Preemption

## 1. Project Overview

This project focuses on detecting emergency vehicles in heterogeneous traffic conditions and providing priority at traffic junctions through an adaptive traffic signal control system.

The system uses computer vision to identify vehicles and an emergency decision mechanism to request signal priority for emergency vehicles.

## 2. Problem Statement

Emergency vehicles such as ambulances and police vehicles can experience delays at busy traffic junctions. Conventional traffic signals operate using fixed timing and may not provide immediate priority to emergency vehicles.

This project aims to develop a prototype that detects emergency vehicles and supports adaptive traffic signal preemption.

## 3. Objectives

- Detect different vehicle categories from traffic images.
- Identify emergency vehicles such as ambulances and police cars.
- Evaluate emergency detection confidence.
- Determine the direction requiring signal priority.
- Simulate safe traffic signal transitions.
- Display detection and signal information through a frontend.

## 4. Proposed System

The proposed system consists of the following modules:

1. Vehicle Detection
2. Emergency Decision Engine
3. Traffic Signal Controller
4. Frontend Dashboard
5. Testing and Documentation

## 5. Vehicle Categories

| Class ID | Vehicle |
|---|---|
| 0 | Car |
| 1 | Motorcycle |
| 2 | Autorickshaw |
| 3 | Truck |
| 4 | Bus |
| 5 | Bicycle |
| 6 | Ambulance |
| 7 | Police Car |

## 6. System Workflow

Traffic Image → Camera Direction Selection → Vehicle Detection → Emergency Decision → Traffic Signal Controller → Frontend Display

## 7. Technologies

- Python
- YOLO
- OpenCV
- Streamlit
- Pytest

Final dependencies and versions should be confirmed from the integrated project requirements.

## 8. Project Structure

The following is the proposed project structure:

```text
Emergency_Vehicle_Project/
├── src/
├── tests/
├── models/
├── docs/
│   ├── TEST_PLAN.md
│   ├── DEMO_GUIDE.md
│   └── INTEGRATION_ISSUES.md
├── app.py
├── requirements.txt
└── README.md
