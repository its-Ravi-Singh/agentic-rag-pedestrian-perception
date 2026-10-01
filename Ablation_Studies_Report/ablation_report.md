# Ablation Study: Adaptive vs. All Adapters

This report compares the performance of the system when firing all adapters on every frame versus using an adaptive, rule-based approach.

| Metric | ALL Mode | ADAPTIVE Mode | Change |
|---|---|---|---|
| **Avg Inference Time** | 110.75 ms | 78.63 ms | -29.0% ✅ |
| **Avg FPS** | 9.03 | 12.72 | +40.8% ✅ |
| **Avg GFLOPs** | 102.03 | 72.96 | -28.5% ✅ |
| **Avg Adapters Fired** | 5.33 | 3.80 | -28.7% ✅ |
| **Avg Time / Active Adapter** | 20.76 ms | 20.67 ms | -0.5% ✅ |

### Adapter Utilization (% of frames)

| Adapter | ALL Mode | ADAPTIVE Mode |
|---|---|---|
| ACTION | 100.0% | 79.6% |
| LOOK | 100.0% | 79.6% |
| CROSS | 100.0% | 79.6% |
| OCCLUSION | 100.0% | 55.2% |
| WEATHER | 33.3% | 4.2% |
| TIME_OF_DAY | 33.3% | 4.2% |
| PED_DENSITY | 33.3% | 33.3% |
| PRESENCE | 33.3% | 44.8% |
