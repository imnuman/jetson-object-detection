# Jetson Object Detection Pipeline

High-performance object detection pipeline for NVIDIA Jetson platforms with TensorRT optimization.

![Jetson Detection Demo](assets/demo.gif)

## Features

- **TensorRT Acceleration**: 3-5x faster inference vs PyTorch
- **Multi-Model Support**: YOLOv8, YOLOv5, SSD, EfficientDet
- **Camera Streaming**: USB, CSI, RTSP, GStreamer pipelines
- **Low Latency**: <30ms end-to-end on Jetson Orin Nano
- **Easy Deployment**: Docker containers and JetPack integration

## Performance Benchmarks

| Model | Jetson Orin Nano | Jetson Nano 4GB | Input Size |
|-------|------------------|-----------------|------------|
| YOLOv8n | 65 FPS | 28 FPS | 640x640 |
| YOLOv8s | 42 FPS | 15 FPS | 640x640 |
| YOLOv5n | 70 FPS | 32 FPS | 640x640 |
| SSD-MobileNetV2 | 85 FPS | 45 FPS | 300x300 |

## Installation

### Prerequisites

- NVIDIA Jetson (Orin Nano, Orin NX, Xavier NX, Nano)
- JetPack 5.x (for Orin) or JetPack 4.6 (for older devices)
- Python 3.8+

### Quick Install

```bash
# Clone repository
git clone https://github.com/imnuman/jetson-object-detection.git
cd jetson-object-detection

# Install dependencies
pip install -r requirements.txt

# Build TensorRT engines (first time only)
python scripts/build_engines.py --model yolov8n
```

### Docker Installation

```bash
# Pull pre-built image
docker pull imnuman/jetson-detection:latest

# Or build locally
docker build -t jetson-detection .

# Run with GPU access
docker run --runtime nvidia -it jetson-detection
```

## Usage

### Basic Detection

```python
from jetson_detection import JetsonDetector

# Initialize with TensorRT engine
detector = JetsonDetector(
    engine='models/yolov8n.engine',
    labels='configs/coco.txt'
)

# Detect from image
results = detector.detect('image.jpg')

# Detect from camera
detector.detect_camera(camera_id=0, show=True)
```

### Command Line

```bash
# Detect in image
python detect.py --source image.jpg --engine models/yolov8n.engine

# USB camera
python detect.py --source /dev/video0 --show

# CSI camera (Raspberry Pi Camera v2)
python detect.py --source csi://0 --show

# RTSP stream
python detect.py --source rtsp://192.168.1.100:554/stream
```

### GStreamer Pipeline

```python
from jetson_detection import GStreamerCamera

# CSI camera with ISP processing
camera = GStreamerCamera(
    sensor_id=0,
    width=1920,
    height=1080,
    fps=30,
    flip_method=0  # 0=none, 2=180 rotation
)

while True:
    frame = camera.read()
    results = detector.detect(frame)
```

## Model Conversion

### PyTorch to TensorRT

```bash
# Convert YOLOv8
python scripts/convert_model.py \
    --weights yolov8n.pt \
    --format engine \
    --fp16 \
    --workspace 4

# Convert with INT8 quantization
python scripts/convert_model.py \
    --weights yolov8n.pt \
    --format engine \
    --int8 \
    --calib-data calibration_images/
```

### Supported Export Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| TensorRT | .engine | Fastest inference on Jetson |
| ONNX | .onnx | Cross-platform deployment |
| TorchScript | .torchscript | PyTorch ecosystem |

## API Reference

### JetsonDetector

```python
class JetsonDetector:
    def __init__(
        self,
        engine: str,           # TensorRT engine path
        labels: str,           # Class labels file
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        max_detections: int = 100
    ):
        ...

    def detect(
        self,
        source: Union[str, np.ndarray],
        show: bool = False,
        save: bool = False
    ) -> List[Detection]:
        """Detect objects in image."""
        ...

    def detect_camera(
        self,
        camera_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        show: bool = True,
        callback: Callable = None
    ):
        """Real-time detection from camera."""
        ...

    def benchmark(self, iterations: int = 100) -> Dict:
        """Benchmark inference performance."""
        ...
```

### Detection Result

```python
@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    area: int
```

## Project Structure

```
jetson-object-detection/
├── configs/
│   ├── coco.txt           # COCO class labels
│   ├── custom.txt         # Custom class labels
│   └── camera.yaml        # Camera configurations
├── models/
│   ├── yolov8n.engine     # TensorRT engine
│   └── yolov8n.onnx       # ONNX model
├── scripts/
│   ├── build_engines.py   # Build TensorRT engines
│   ├── convert_model.py   # Model conversion
│   ├── benchmark.py       # Performance benchmark
│   └── calibrate.py       # INT8 calibration
├── jetson_detection/
│   ├── __init__.py
│   ├── detector.py        # Main detector class
│   ├── engine.py          # TensorRT engine wrapper
│   ├── camera.py          # Camera interfaces
│   └── utils.py           # Utilities
├── detect.py              # CLI tool
├── Dockerfile             # Docker container
├── requirements.txt
└── README.md
```

## Camera Support

### Supported Cameras

| Camera Type | Interface | Example |
|-------------|-----------|---------|
| USB Webcam | V4L2 | Logitech C920 |
| CSI Camera | MIPI CSI-2 | RPi Camera V2, IMX219 |
| IP Camera | RTSP/HTTP | Hikvision, Dahua |
| Industrial | GigE Vision | FLIR Blackfly |

### CSI Camera Configuration

```yaml
# configs/camera.yaml
csi_camera:
  sensor_id: 0
  width: 1920
  height: 1080
  fps: 30
  flip_method: 0
  exposure: auto
  gain: auto
```

## Optimization Tips

### Memory Optimization

```bash
# Increase swap (for Jetson Nano 4GB)
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Set max performance mode
sudo nvpmodel -m 0
sudo jetson_clocks
```

### INT8 Quantization

INT8 provides 2x speedup with minimal accuracy loss:

```bash
# Prepare calibration dataset (100-500 images)
python scripts/calibrate.py --images calibration/ --output calib.cache

# Build INT8 engine
python scripts/build_engines.py --model yolov8n --int8 --calib calib.cache
```

## Troubleshooting

### Common Issues

**TensorRT build fails**
```bash
# Ensure JetPack is properly installed
sudo apt-get install nvidia-tensorrt
```

**CUDA out of memory**
```bash
# Use smaller batch size or model
python detect.py --batch 1 --engine yolov8n.engine
```

**Camera not detected**
```bash
# Check V4L2 devices
v4l2-ctl --list-devices

# For CSI cameras
ls /dev/video*
```

## Contributing

Contributions welcome! Focus areas:
- Additional model support
- Camera drivers
- Performance optimizations
- Documentation

## License

MIT License - see [LICENSE](LICENSE)

## Acknowledgments

- [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [JetsonHacks](https://jetsonhacks.com/)

## Contact

- **Author**: Al Numan
- **Email**: admin@numanab.com
- **Project**: [ProjectX](https://projectxbd.com)

---

*Optimized object detection for edge AI applications on NVIDIA Jetson.*
