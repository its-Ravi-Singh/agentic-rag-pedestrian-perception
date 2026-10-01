# Comprehensive Performance Analysis

Detailed computational efficiency and model performance metrics.

## 1. Computational Efficiency Metrics

### Latency & Throughput

| Metric | Value | Unit |
|--------|-------|------|
| Total Samples Processed | 225,197 | samples |
| Mean Latency | 208.17 | ms/sample |
| Median Latency | 189.92 | ms/sample |
| 95th Percentile Latency | 357.21 | ms/sample |
| 99th Percentile Latency | 433.51 | ms/sample |
| Mean Throughput | 4.8 | samples/sec |
| Mean FPS | 5.3 | frames/sec |
| Median FPS | 5.3 | frames/sec |

## 2. Scalability Analysis

### Performance vs Pedestrian Count

| Pedestrians | Avg Latency (ms) | Std Latency | Avg FPS | Std FPS | Sample Count |
|-------------|------------------|-------------|---------|---------|---------------|
| 1 | 117.72 | 9.21 | 8.5 | 0.5 | 14,810 |
| 2 | 136.36 | 8.95 | 7.4 | 0.4 | 22,957 |
| 3 | 155.41 | 8.39 | 6.5 | 0.3 | 26,138 |
| 4 | 171.91 | 8.96 | 5.8 | 0.3 | 29,215 |
| 5 | 186.80 | 11.03 | 5.4 | 0.3 | 26,769 |
| 6 | 203.23 | 10.93 | 4.9 | 0.2 | 23,532 |
| 7 | 228.97 | 12.32 | 4.4 | 0.2 | 20,432 |
| 8 | 243.81 | 13.57 | 4.1 | 0.2 | 13,765 |
| 9 | 257.28 | 12.86 | 3.9 | 0.2 | 10,908 |
| 10 | 285.09 | 14.00 | 3.5 | 0.1 | 7,725 |
| 11 | 293.81 | 13.37 | 3.4 | 0.1 | 6,182 |
| 12 | 302.73 | 13.09 | 3.3 | 0.1 | 5,607 |
| 13 | 353.22 | 11.54 | 2.8 | 0.1 | 4,609 |
| 14 | 352.85 | 11.87 | 2.8 | 0.1 | 3,504 |
| 15 | 365.11 | 11.56 | 2.7 | 0.1 | 2,864 |
| 16 | 395.96 | 11.25 | 2.5 | 0.1 | 2,209 |
| 17 | 411.07 | 8.46 | 2.4 | 0.1 | 1,287 |
| 18 | 426.43 | 11.72 | 2.4 | 0.1 | 627 |
| 19 | 458.63 | 14.10 | 2.2 | 0.1 | 536 |
| 20 | 466.85 | 10.36 | 2.1 | 0.1 | 414 |
| 21 | 468.55 | 9.56 | 2.1 | 0.0 | 249 |
| 22 | 484.88 | 14.56 | 2.1 | 0.1 | 252 |
| 23 | 493.69 | 12.81 | 2.0 | 0.1 | 249 |
| 24 | 514.39 | 15.83 | 1.9 | 0.1 | 173 |
| 25 | 595.45 | 10.97 | 1.7 | 0.0 | 136 |
| 26 | 634.65 | 0.00 | 1.6 | 0.0 | 16 |
| 28 | 634.16 | 8.37 | 1.6 | 0.0 | 32 |

## 3. Accuracy-Efficiency Trade-offs

### Attribute Performance Summary

| Attribute | Accuracy | Avg Latency (ms) | Efficiency Score* |
|-----------|----------|------------------|-------------------|
| ACTION | 0.897 | 226.85 | 3.95 |
| LOOK | 0.846 | 226.85 | 3.73 |
| CROSS | 0.860 | 226.85 | 3.79 |
| OCCLUSION | 0.688 | 226.85 | 3.03 |

*Efficiency Score = (Accuracy / Latency) × 1000 (higher is better)

## 4. Inference Stability Analysis

### Stability Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Latency CV* | 0.343 | Moderate |
| FPS CV* | 0.299 | Moderate |
| Latency Range | 99.8 - 642.9 ms | - |
| FPS Range | 1.6 - 10.0 | - |

*CV = Coefficient of Variation (std/mean). Lower values indicate more stable performance.

## 5. Performance Optimization Recommendations

- (High Alert) High Latency Detected (>100ms): Consider model quantization, pruning, or batch processing strategies.
- (Stat) Best Performing Attribute: ACTION (0.897 accuracy)
- (Stat) Needs Improvement: OCCLUSION (0.688 accuracy)

