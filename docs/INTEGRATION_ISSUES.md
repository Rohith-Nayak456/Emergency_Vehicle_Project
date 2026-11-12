\# INTEGRATION ISSUES REPORT



\## Emergency Vehicle Detection and Adaptive Signal Preemption



\### 1. Purpose

This document records potential integration problems identified during project branch inspection. These issues must be verified when the modules are combined.



\### 2. Identified Issues



| ID | Issue | Description | Action Required |

|---|---|---|---|

| INT01 | Different inference interfaces | Different branches use different inference function names and output formats. | Agree on one common interface. |

| INT02 | Decision engine mismatch | Decision engine implementations use different function names and input formats. | Standardize the input and output structure. |

| INT03 | Signal controller mismatch | Controller implementations use different class and function names. | Select one controller implementation. |

| INT04 | Confidence threshold mismatch | Different project documents mention different emergency confidence thresholds. | Agree on one threshold. |

| INT05 | Model availability | Model file availability and training status need confirmation. | Verify the final trained model file. |

| INT06 | Frontend mock mode | The frontend contains simulated detection logic rather than confirmed real YOLO inference. | Connect and test the actual inference module. |

| INT07 | Signal safety | The mock frontend logic may show conflicting GREEN signals. | Enforce and test safe signal transitions. |

| INT08 | Documentation mismatch | Some API descriptions may not match the actual implementation. | Update documentation after integration. |



\### 3. Integration Workflow



The expected integrated workflow is:



1\. Upload traffic image.

2\. Select camera direction.

3\. Send image to vehicle detection module.

4\. Receive vehicle classes and confidence scores.

5\. Send detection results to the emergency decision engine.

6\. Send valid emergency requests to the signal controller.

7\. Display detection and signal results in the frontend.



\### 4. Verification Checklist



\- \[ ] All required modules are available.

\- \[ ] Module imports work correctly.

\- \[ ] Dependencies are installed.

\- \[ ] Model path is correct.

\- \[ ] Inference output matches the decision engine input.

\- \[ ] Decision engine output matches controller input.

\- \[ ] Frontend displays actual detection results.

\- \[ ] Signal transitions follow the safe sequence.

\- \[ ] No conflicting GREEN signals occur.

\- \[ ] Error handling works correctly.



\### 5. Important Note



The issues listed above are identified from branch and documentation inspection. They are not all confirmed runtime failures. Each issue must be verified against the final integrated code.



\### 6. Conclusion



Successful integration requires consistent module interfaces, a verified trained model, correct dependency configuration, and safe traffic signal transitions. All identified issues should be reviewed with the responsible teammates before final system testing.
