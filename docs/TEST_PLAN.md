\# TEST PLAN

\## Emergency Vehicle Detection and Adaptive Signal Preemption



\### 1. Introduction

This document describes the testing plan for the Emergency Vehicle Detection and Adaptive Signal Preemption system. The purpose is to verify vehicle detection, emergency classification, decision-making, traffic signal control, and frontend functionality.



\### 2. Testing Objectives

\- Verify detection of all supported vehicle classes.

\- Verify identification of ambulances and police cars.

\- Check emergency decision-making based on confidence thresholds.

\- Verify safe traffic signal preemption.

\- Check frontend functionality and error handling.

\- Identify integration issues between project modules.



\### 3. Testing Categories



1\. Unit Testing

2\. Integration Testing

3\. Frontend Testing

4\. Model Performance Testing

5\. System Testing



\### 4. Vehicle Detection Test Cases



| ID | Test Case | Expected Result |

|---|---|---|

| VD01 | Detect car | Car detected |

| VD02 | Detect motorcycle | Motorcycle detected |

| VD03 | Detect autorickshaw | Autorickshaw detected |

| VD04 | Detect truck | Truck detected |

| VD05 | Detect bus | Bus detected |

| VD06 | Detect bicycle | Bicycle detected |

| VD07 | Detect ambulance | Ambulance detected |

| VD08 | Detect police car | Police car detected |

| VD09 | Image without vehicles | No vehicle detections |

| VD10 | Image with multiple vehicles | Multiple detections displayed |



\### 5. Emergency Decision Test Cases



| ID | Test Case | Expected Result |

|---|---|---|

| ED01 | Ambulance above confidence threshold | Emergency identified |

| ED02 | Police car above confidence threshold | Emergency identified |

| ED03 | Emergency vehicle below threshold | No emergency preemption |

| ED04 | Ordinary vehicles only | Normal signal operation |

| ED05 | Invalid camera direction | Error handled safely |

| ED06 | Invalid emergency class | Request rejected |

| ED07 | Failed inference | Error handled safely |

| ED08 | Missing input fields | Validation error displayed |



\### 6. Traffic Signal Test Cases



| ID | Test Case | Expected Result |

|---|---|---|

| TS01 | Normal signal operation | Normal cycle maintained |

| TS02 | Emergency from NORTH | NORTH receives priority |

| TS03 | Emergency from SOUTH | SOUTH receives priority |

| TS04 | Emergency from EAST | EAST receives priority |

| TS05 | Emergency from WEST | WEST receives priority |

| TS06 | GREEN to YELLOW transition | Safe transition |

| TS07 | YELLOW to ALL-RED transition | All signals turn red |

| TS08 | ALL-RED to requested GREEN | Requested direction gets green |

| TS09 | Conflicting green signals | No conflicting green |

| TS10 | Emergency clearance | Normal operation resumes |

| TS11 | Multiple emergency requests | Requests handled safely |



\### 7. Frontend Test Cases



| ID | Test Case | Expected Result |

|---|---|---|

| FE01 | Launch application | Dashboard opens |

| FE02 | Upload image | Image displayed |

| FE03 | Select camera direction | Direction selected |

| FE04 | Run detection | Detection results displayed |

| FE05 | Display annotated image | Image with detection boxes shown |

| FE06 | Display emergency status | Correct status shown |

| FE07 | Display signal state | Signal state shown |

| FE08 | Reset application | Inputs and results reset |

| FE09 | Invalid image upload | Error message displayed |



\### 8. Integration Testing



The complete workflow will be tested after the project modules are integrated:



Image Upload → Camera Direction Selection → Vehicle Detection → Emergency Decision → Signal Controller → Frontend Display.



\### 9. Testing Tools

\- Python

\- Pytest

\- Streamlit

\- OpenCV

\- YOLO

\- Mock objects for isolated unit tests



\### 10. Important Notes

\- Mock-based tests do not prove actual YOLO model performance.

\- Actual model testing requires the trained model file and suitable test images.

\- Integration tests require compatible modules from all teammates.

\- Signal controller testing must verify that conflicting directions never receive GREEN simultaneously.



\### 11. Test Results

Test results will be recorded after executing the tests. Tests that cannot be performed due to missing modules or model files will be marked as pending.
