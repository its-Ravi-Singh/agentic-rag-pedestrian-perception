# Detailed Classification Metrics

Per-class precision, recall, and F1-score for each pedestrian attribute.

## ACTION Attribute

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| standing | 0.577 | 0.808 | 0.673 | 13102 |
| walking | 0.969 | 0.910 | 0.939 | 86332 |

### Summary Metrics

| Metric | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Macro Average | 0.773 | 0.859 | 0.806 |
| Weighted Average | 0.917 | 0.897 | 0.904 |
| **Overall Accuracy** | | | **0.897** |

---

## LOOK Attribute

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| looking | 0.574 | 0.794 | 0.666 | 19204 |
| not_looking | 0.946 | 0.859 | 0.900 | 80230 |

### Summary Metrics

| Metric | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Macro Average | 0.760 | 0.826 | 0.783 |
| Weighted Average | 0.874 | 0.846 | 0.855 |
| **Overall Accuracy** | | | **0.846** |

---

## CROSS Attribute

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| crossing | 0.858 | 0.914 | 0.885 | 58810 |
| not_crossing | 0.863 | 0.781 | 0.820 | 40624 |

### Summary Metrics

| Metric | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Macro Average | 0.860 | 0.847 | 0.852 |
| Weighted Average | 0.860 | 0.860 | 0.858 |
| **Overall Accuracy** | | | **0.860** |

---

## OCCLUSION Attribute

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|----------|
| full_occlusion | 0.102 | 0.566 | 0.172 | 4693 |
| no_occlusion | 0.982 | 0.698 | 0.816 | 205386 |
| partial_occlusion | 0.157 | 0.576 | 0.246 | 14170 |

### Summary Metrics

| Metric | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Macro Average | 0.413 | 0.613 | 0.412 |
| Weighted Average | 0.911 | 0.688 | 0.767 |
| **Overall Accuracy** | | | **0.688** |

---

