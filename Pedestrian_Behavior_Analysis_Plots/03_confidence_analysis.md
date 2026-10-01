# Model Confidence Analysis

Uncertainty quantification and confidence score analysis for pedestrian behavior prediction.

## 1. Confidence Score Distributions

### ACTION Attribute

| Metric | Value |
|--------|-------|
| Mean Confidence | 0.944 |
| Std Confidence | 0.112 |
| Min Confidence | 0.500 |
| Max Confidence | 1.000 |
| Samples | 224,999 |

### LOOK Attribute

| Metric | Value |
|--------|-------|
| Mean Confidence | 0.916 |
| Std Confidence | 0.129 |
| Min Confidence | 0.500 |
| Max Confidence | 1.000 |
| Samples | 224,999 |

### CROSS Attribute

| Metric | Value |
|--------|-------|
| Mean Confidence | 0.912 |
| Std Confidence | 0.129 |
| Min Confidence | 0.500 |
| Max Confidence | 1.000 |
| Samples | 224,999 |

### OCCLUSION Attribute

| Metric | Value |
|--------|-------|
| Mean Confidence | 0.863 |
| Std Confidence | 0.162 |
| Min Confidence | 0.335 |
| Max Confidence | 1.000 |
| Samples | 224,999 |

## 2. Confidence-Based Filtering Recommendations

- **ACTION**: Optimal threshold >= 0.30 -> 0.897 accuracy at 100.0% coverage.
- **LOOK**: Optimal threshold >= 0.30 -> 0.846 accuracy at 100.0% coverage.
- **CROSS**: Optimal threshold >= 0.30 -> 0.860 accuracy at 100.0% coverage.
- **OCCLUSION**: Optimal threshold >= 0.50 -> 0.698 accuracy at 97.0% coverage.
