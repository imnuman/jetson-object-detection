# Benchmark Log - Jetson Object Detection

## Overview

Performance benchmarks for YOLOv8 inference on various edge platforms.

## Test Configuration

- Model: YOLOv8n (nano variant)
- Input Size: 640x640
- Test Images: 1000 (COCO val subset)
- Precision: FP16 (TensorRT)

## Results (2025-01-11)

| Platform | FPS | Latency | Memory | TensorRT |
|----------|-----|---------|--------|----------|
| Jetson Orin Nano | 45 | 22ms | 512MB | 8.6.1 |
| Jetson Nano 4GB | 18 | 55ms | 890MB | 8.2.1 |
| Desktop RTX 3060 | 142 | 7ms | 1024MB | 8.6.1 |

## Optimization Notes

1. FP16 precision gives 2x speedup with minimal accuracy loss
2. TensorRT engine takes ~30s to build on first run
3. Memory usage stable over long inference runs

## Reproducing Benchmarks

```bash
# Export to TensorRT
python export_tensorrt.py --model yolov8n.pt --fp16

# Run benchmark
python benchmark.py --engine models/yolov8n.engine --images data/test
```
