# Adapter vs YOLO Performance Comparison

Comprehensive analysis of pedestrian density classification performance.

**Note**: YOLO counts are based on 'pedestrian' and 'ped' classes only, not the broader 'people' class, ensuring fair comparison with adapter classifications.

## Overall Performance

- **Classification Agreement**: 0.538 (53.8%)
- **Average Adapter Time**: 34.40 ms
- **Average YOLO Time**: 10.85 ms
- **Speed Advantage**: 0.32x (YOLO faster)

## Density Change Analysis

- **Videos analyzed**: 346
- **Average video duration**: 7.7 seconds
- **Average density changes per video**:
  - Adapter: 22.5
  - YOLO: 27.3
