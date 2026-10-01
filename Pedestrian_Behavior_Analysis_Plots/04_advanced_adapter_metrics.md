# Advanced Adapter Performance Metrics

Analysis of adapter-specific performance, focusing on parameter and computational efficiency.

## 1. Parameter Efficiency Comparison

| Attribute | Adapter Type | Est. Params (M) | Accuracy | Efficiency Score* |
|-----------|--------------|-----------------|----------|-------------------|
| ACTION | LOHA | 0.89 | 0.897 | 1.01 |
| LOOK | LOHA | 0.89 | 0.846 | 0.95 |
| CROSS | ADALORA | 0.44 | 0.860 | 1.94 |
| OCCLUSION | ADALORA | 0.44 | 0.688 | 1.55 |

*Efficiency Score = Accuracy / Million Parameters (higher is better)

## 2. Computational Complexity Analysis

### GFLOPs per Adapter

| Adapter | Mean GFLOPs | Samples |
|---------|-------------|----------|
| ACTION | 21.765 | 225,197 |
| LOOK | 21.765 | 225,197 |
| CROSS | 17.669 | 225,197 |
| OCCLUSION | 17.669 | 225,197 |

## 3. Scalability Analysis

### Batch Processing Efficiency

| Batch Size | Avg Time (ms) | Time per Sample (ms) | Efficiency Gain* | Samples |
|------------|---------------|----------------------|------------------|----------|
| 1 | 117.7 | 117.7 | 1.00× | 14,810 |
| 2 | 136.4 | 68.2 | 1.73× | 22,957 |
| 3 | 155.4 | 51.8 | 2.27× | 26,138 |
| 4 | 171.9 | 43.0 | 2.74× | 29,215 |
| 5 | 186.8 | 37.4 | 3.15× | 26,769 |
| 6 | 203.2 | 33.9 | 3.48× | 23,532 |
| 7 | 229.0 | 32.7 | 3.60× | 20,432 |
| 8 | 243.8 | 30.5 | 3.86× | 13,765 |
| 9 | 257.3 | 28.6 | 4.12× | 10,908 |
| 10 | 285.1 | 28.5 | 4.13× | 7,725 |
| 11 | 293.8 | 26.7 | 4.41× | 6,182 |
| 12 | 302.7 | 25.2 | 4.67× | 5,607 |
| 13 | 353.2 | 27.2 | 4.33× | 4,609 |
| 14 | 352.9 | 25.2 | 4.67× | 3,504 |
| 15 | 365.1 | 24.3 | 4.84× | 2,864 |
| 16 | 396.0 | 24.7 | 4.76× | 2,209 |
| 17 | 411.1 | 24.2 | 4.87× | 1,287 |
| 18 | 426.4 | 23.7 | 4.97× | 627 |
| 19 | 458.6 | 24.1 | 4.88× | 536 |
| 20 | 466.9 | 23.3 | 5.04× | 414 |
| 21 | 468.6 | 22.3 | 5.28× | 249 |
| 22 | 484.9 | 22.0 | 5.34× | 252 |
| 23 | 493.7 | 21.5 | 5.48× | 249 |
| 24 | 514.4 | 21.4 | 5.49× | 173 |
| 25 | 595.5 | 23.8 | 4.94× | 136 |
| 26 | 634.6 | 24.4 | 4.82× | 16 |
| 28 | 634.2 | 22.6 | 5.20× | 32 |

*Efficiency Gain = Baseline Time per Sample / Current Time per Sample

## 4. Adapter Type Comparison Summary

### Performance vs Efficiency Trade-offs

| Rank | Attribute | Adapter | Accuracy | Parameters | Efficiency |
|------|-----------|---------|----------|------------|------------|
| 1 | CROSS | ADALORA | 0.860 | 0.44M | 1.94 |
| 2 | OCCLUSION | ADALORA | 0.688 | 0.44M | 1.55 |
| 3 | ACTION | LOHA | 0.897 | 0.89M | 1.01 |
| 4 | LOOK | LOHA | 0.846 | 0.89M | 0.95 |

## 4. Recommendations

- (Best) Best Accuracy: **ACTION** (LOHA) at 0.897
- (Most Efficient) Most Efficient: **CROSS** (ADALORA) at 1.94 acc/M_params
