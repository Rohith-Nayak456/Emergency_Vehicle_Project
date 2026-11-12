\# PROJECT DEMONSTRATION GUIDE



\## Emergency Vehicle Detection and Adaptive Signal Preemption for Heterogeneous Traffic Conditions



\### 1. Introduction

This demonstration explains how the system detects emergency vehicles and gives priority to their direction by controlling the traffic signal.



\### 2. Demonstration Objectives

\- Demonstrate vehicle detection using YOLO.

\- Identify ambulances and police vehicles.

\- Show emergency decision-making.

\- Demonstrate adaptive traffic signal preemption.

\- Display detection results through the frontend.



\### 3. System Requirements

\- Python installed

\- Required project dependencies

\- Trained YOLO model file

\- Project source code

\- Sample traffic images

\- Streamlit application, if used by the final integrated project



\### 4. Demonstration Procedure



\*\*Step 1: Start the Application\*\*



Open the project folder and launch the application using the command specified in the final project setup.



\*\*Step 2: Upload a Traffic Image\*\*



Select a sample traffic image containing different types of vehicles.



\*\*Step 3: Select Camera Direction\*\*



Choose the direction from which the traffic image was captured:

\- NORTH

\- SOUTH

\- EAST

\- WEST



\*\*Step 4: Run Vehicle Detection\*\*



Run the detection process and display the detected vehicles with their bounding boxes and confidence scores.



\*\*Step 5: Demonstrate Normal Traffic\*\*



Upload an image containing only ordinary vehicles. The system should maintain normal traffic signal operation.



\*\*Step 6: Demonstrate Emergency Vehicle Detection\*\*



Upload an image containing an ambulance. Show the detected emergency vehicle and its confidence score.



\*\*Step 7: Demonstrate Signal Preemption\*\*



Show how the decision engine identifies the emergency and requests priority for the selected direction.



\*\*Step 8: Demonstrate Signal Safety\*\*



Explain the safe transition sequence:



GREEN → YELLOW → ALL-RED → REQUESTED GREEN



Verify that conflicting directions are not given GREEN simultaneously.



\*\*Step 9: Demonstrate Police Vehicle Detection\*\*



If a suitable test image is available, demonstrate police vehicle detection and emergency priority.



\### 5. Expected Demonstration Output

\- Uploaded traffic image

\- Detected vehicle classes

\- Bounding boxes and confidence scores

\- Emergency vehicle status

\- Selected camera direction

\- Traffic signal state



\### 6. Limitations

\- The system processes still images rather than continuous live video.

\- Camera direction is selected manually.

\- Vehicle speed, movement direction and estimated arrival time are not calculated.

\- Traffic signal operation is simulated.

\- No real traffic signal hardware is controlled.

\- Actual performance depends on the trained model and test images.



\### 7. Future Improvements

\- Live CCTV video processing

\- Automatic camera direction identification

\- Vehicle tracking and speed estimation

\- Estimated arrival time calculation

\- Integration with real traffic signal hardware

\- Improved emergency vehicle detection accuracy



\### 8. Conclusion

The demonstration illustrates how computer vision and adaptive signal control can support emergency vehicle prioritization at traffic junctions. The system is a prototype and requires further testing before real-world deployment.
