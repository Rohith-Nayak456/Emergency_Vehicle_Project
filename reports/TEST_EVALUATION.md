# Test Evaluation Report — YOLO11n Emergency Vehicle Detection

**Model:** YOLO11n (Ultralytics)
**Evaluation date:** 2026-11-11
**Evaluated on:** Test split (1,491 images — never used during training or model selection)

---

## Overall Test Results

| Metric | Value |
|---|---|
| Precision | 62.47% |
| Recall | 55.89% |
| mAP@0.5 | 60.90% |
| mAP@0.5:0.95 | 48.67% |

## Validation Results (used for model selection)

| Metric | Value |
|---|---|
| Precision | ~72.3% |
| Recall | ~70.2% |
| mAP@0.5 | ~75.6% |
| mAP@0.5:0.95 | ~61.8% |

The gap between validation and test performance is expected given the small dataset and limited emergency scene diversity.

## Per-Class Test Performance

| Class ID | Class | Precision | Recall | mAP@0.5 |
|---:|---|---|---|---|
| 0 | car | — | — | — |
| 1 | motorcycle | — | — | — |
| 2 | autorickshaw | — | — | — |
| 3 | truck | — | — | — |
| 4 | bus | — | — | — |
| 5 | bicycle | — | — | — |
| 6 | **ambulance** | **40.9%** | **31.2%** | **35.4%** |
| 7 | **police_car** | **39.7%** | **12.5%** | **17.3%** |

> Per-class metrics for common vehicle classes (0-5) are available in the Colab training output. The emergency classes are highlighted here because they are the most critical and the weakest.

## Emergency Class Analysis

### Ambulance (Class 6)
- **35.4% mAP@0.5** — below the overall average
- 31.2% recall means ~69% of ambulances are missed
- The dataset contains only **94 unique ambulance scenes** (expanded to 561 via augmentation)
- Limited visual diversity in training data

### Police Car (Class 7)
- **17.3% mAP@0.5** — significantly below average
- 12.5% recall means ~88% of police cars are missed
- The dataset contains only **40 unique police car scenes** (expanded to 513 via augmentation)
- Very limited visual diversity — the model has not learned generalizable police car features

## Dataset Composition

| Split | Total | Ambulance Imgs | Police Car Imgs |
|---|---:|---:|---:|
| Train | 6,836 | 433 | 364 |
| Val | 1,412 | 48 | 54 |
| Test | 1,491 | 80 | 95 |

**Unique original scenes:** ambulance=94, police_car=40
**Augmented copies are visually similar to originals and do not add real diversity.**

## Overfitting Assessment

- Validation mAP@0.5 (~75.6%) is notably higher than test mAP@0.5 (60.90%)
- This gap is consistent with mild overfitting, particularly on the emergency classes where augmented copies dominate
- Early stopping (patience=20) was used but the limited emergency data means generalization is inherently constrained

## Limitations

1. **Insufficient emergency training data.** The model has seen very few distinct ambulance and police car appearances.
2. **Augmentation does not equal diversity.** Most emergency images are colour/brightness variants of the same scenes.
3. **Police car detection is unreliable.** With only 40 original scenes and 12.5% test recall, the model frequently fails to detect police vehicles.
4. **Not production-ready.** These results reflect an academic prototype with limited data.

## Recommendations for Improvement

1. Collect additional diverse emergency vehicle images (100+ unique scenes per class)
2. Include varied ambulance and police car liveries, sizes, angles and lighting
3. Consider targeted augmentation strategies (e.g., copy-paste augmentation)
4. Experiment with class-weighted loss functions
5. Fine-tune on emergency-focused mini-batches after initial training
