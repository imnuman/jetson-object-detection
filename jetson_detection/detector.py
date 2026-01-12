"""
Jetson Object Detection
=======================
High-performance object detection for NVIDIA Jetson with TensorRT.

Author: Al Numan
Project: ProjectX
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Union, Callable, Optional, Tuple
from dataclasses import dataclass
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single object detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2

    @property
    def center(self) -> Tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    def to_dict(self) -> Dict:
        return {
            'class_id': self.class_id,
            'class_name': self.class_name,
            'confidence': round(self.confidence, 3),
            'bbox': list(self.bbox),
            'center': list(self.center),
            'area': self.area
        }


class JetsonDetector:
    """
    High-performance object detector for NVIDIA Jetson.

    Uses TensorRT for optimized inference.

    Example:
        detector = JetsonDetector('models/yolov8n.engine')
        results = detector.detect('image.jpg')
        detector.detect_camera(0, show=True)
    """

    def __init__(
        self,
        engine: str,
        labels: str = None,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        max_detections: int = 100,
        input_size: Tuple[int, int] = (640, 640)
    ):
        """
        Initialize Jetson detector.

        Args:
            engine: Path to TensorRT engine (.engine)
            labels: Path to class labels file
            conf_threshold: Confidence threshold
            nms_threshold: NMS IoU threshold
            max_detections: Maximum detections per image
            input_size: Model input size (width, height)
        """
        self.engine_path = Path(engine)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.input_size = input_size

        # Load class labels
        self.labels = self._load_labels(labels)

        # Load TensorRT engine
        self.engine = None
        self.context = None
        self._load_engine()

        # Performance tracking
        self._inference_times = []

        logger.info(f"Detector initialized: {self.engine_path.name}")

    def _load_labels(self, labels_path: str) -> List[str]:
        """Load class labels from file."""
        if labels_path is None:
            # Default COCO labels
            return [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench',
                'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
                'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
                'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                'couch', 'potted plant', 'bed', 'dining table', 'toilet',
                'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
                'book', 'clock', 'vase', 'scissors', 'teddy bear',
                'hair drier', 'toothbrush'
            ]

        labels = []
        with open(labels_path, 'r') as f:
            for line in f:
                labels.append(line.strip())
        return labels

    def _load_engine(self):
        """Load TensorRT engine."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit

            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

            with open(self.engine_path, 'rb') as f:
                runtime = trt.Runtime(TRT_LOGGER)
                self.engine = runtime.deserialize_cuda_engine(f.read())

            self.context = self.engine.create_execution_context()

            # Allocate buffers
            self._allocate_buffers()

            logger.info("TensorRT engine loaded successfully")

        except ImportError:
            logger.warning("TensorRT not available, using ONNX fallback")
            self._load_onnx_fallback()

    def _load_onnx_fallback(self):
        """Load ONNX model as fallback."""
        try:
            import onnxruntime as ort

            onnx_path = self.engine_path.with_suffix('.onnx')
            if not onnx_path.exists():
                raise FileNotFoundError(f"Neither TensorRT nor ONNX found")

            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            self.onnx_session = ort.InferenceSession(
                str(onnx_path),
                providers=providers
            )
            logger.info("ONNX fallback loaded")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _allocate_buffers(self):
        """Allocate CUDA buffers for TensorRT."""
        import pycuda.driver as cuda

        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = np.prod(shape)
            dtype = np.float32

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference."""
        # Resize
        resized = cv2.resize(image, self.input_size)

        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # HWC to CHW
        transposed = normalized.transpose(2, 0, 1)

        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)

        return np.ascontiguousarray(batched)

    def _postprocess(
        self,
        outputs: np.ndarray,
        orig_shape: Tuple[int, int]
    ) -> List[Detection]:
        """Postprocess model outputs to detections."""
        detections = []

        # Parse YOLO output format
        # Shape: [1, num_detections, 6] where 6 = [x1, y1, x2, y2, conf, class]
        if len(outputs.shape) == 3:
            outputs = outputs[0]

        # Scale factors for original image
        scale_x = orig_shape[1] / self.input_size[0]
        scale_y = orig_shape[0] / self.input_size[1]

        for detection in outputs:
            if len(detection) < 6:
                continue

            confidence = detection[4]
            if confidence < self.conf_threshold:
                continue

            class_id = int(detection[5])
            x1 = int(detection[0] * scale_x)
            y1 = int(detection[1] * scale_y)
            x2 = int(detection[2] * scale_x)
            y2 = int(detection[3] * scale_y)

            # Clamp to image bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(orig_shape[1], x2)
            y2 = min(orig_shape[0], y2)

            class_name = self.labels[class_id] if class_id < len(self.labels) else 'unknown'

            detections.append(Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=float(confidence),
                bbox=(x1, y1, x2, y2)
            ))

        # Apply NMS
        detections = self._apply_nms(detections)

        return detections[:self.max_detections]

    def _apply_nms(self, detections: List[Detection]) -> List[Detection]:
        """Apply non-maximum suppression."""
        if not detections:
            return []

        boxes = np.array([d.bbox for d in detections])
        scores = np.array([d.confidence for d in detections])

        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.conf_threshold,
            self.nms_threshold
        )

        if len(indices) > 0:
            indices = indices.flatten()
            return [detections[i] for i in indices]

        return []

    def detect(
        self,
        source: Union[str, np.ndarray],
        show: bool = False,
        save: bool = False
    ) -> List[Detection]:
        """
        Detect objects in image.

        Args:
            source: Image path or numpy array
            show: Display result
            save: Save annotated image

        Returns:
            List of Detection objects
        """
        # Load image
        if isinstance(source, str):
            image = cv2.imread(source)
            if image is None:
                raise ValueError(f"Could not load image: {source}")
        else:
            image = source.copy()

        orig_shape = image.shape[:2]

        # Preprocess
        input_tensor = self._preprocess(image)

        # Run inference
        start_time = time.time()

        if hasattr(self, 'onnx_session'):
            # ONNX inference
            outputs = self.onnx_session.run(None, {'images': input_tensor})[0]
        else:
            # TensorRT inference
            outputs = self._trt_inference(input_tensor)

        inference_time = time.time() - start_time
        self._inference_times.append(inference_time)

        # Postprocess
        detections = self._postprocess(outputs, orig_shape)

        # Visualize
        if show or save:
            annotated = self._draw_detections(image, detections)
            if show:
                cv2.imshow('Detection', annotated)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            if save:
                cv2.imwrite('detection_result.jpg', annotated)

        return detections

    def _trt_inference(self, input_tensor: np.ndarray) -> np.ndarray:
        """Run TensorRT inference."""
        import pycuda.driver as cuda

        # Copy input to device
        np.copyto(self.inputs[0]['host'], input_tensor.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]['device'],
            self.inputs[0]['host'],
            self.stream
        )

        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings,
            stream_handle=self.stream.handle
        )

        # Copy output to host
        cuda.memcpy_dtoh_async(
            self.outputs[0]['host'],
            self.outputs[0]['device'],
            self.stream
        )

        self.stream.synchronize()

        return self.outputs[0]['host'].reshape(-1, 6)

    def _draw_detections(
        self,
        image: np.ndarray,
        detections: List[Detection]
    ) -> np.ndarray:
        """Draw detections on image."""
        annotated = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Color based on class
            color = self._get_color(det.class_id)

            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                color, -1
            )

            cv2.putText(
                annotated, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2
            )

        return annotated

    def _get_color(self, class_id: int) -> Tuple[int, int, int]:
        """Get color for class ID."""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128),
            (128, 128, 0), (128, 0, 128), (0, 128, 128)
        ]
        return colors[class_id % len(colors)]

    def detect_camera(
        self,
        camera_id: int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        show: bool = True,
        callback: Callable[[List[Detection]], None] = None
    ):
        """
        Real-time detection from camera.

        Args:
            camera_id: Camera device ID
            width: Frame width
            height: Frame height
            fps: Target FPS
            show: Display video
            callback: Optional callback for detections
        """
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)

        if not cap.isOpened():
            raise ValueError(f"Could not open camera: {camera_id}")

        logger.info(f"Camera opened: {width}x{height}@{fps}FPS")

        frame_count = 0
        fps_start = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                # Detect
                detections = self.detect(frame)

                # Callback
                if callback:
                    callback(detections)

                # Draw
                if show:
                    annotated = self._draw_detections(frame, detections)

                    # FPS display
                    frame_count += 1
                    if frame_count % 30 == 0:
                        elapsed = time.time() - fps_start
                        current_fps = frame_count / elapsed

                    fps_text = f"FPS: {self.get_fps():.1f}"
                    cv2.putText(
                        annotated, fps_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )

                    cv2.imshow('Jetson Detection', annotated)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            cap.release()
            cv2.destroyAllWindows()

    def get_fps(self) -> float:
        """Get average inference FPS."""
        if not self._inference_times:
            return 0.0
        avg_time = sum(self._inference_times[-30:]) / len(self._inference_times[-30:])
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def benchmark(self, iterations: int = 100) -> Dict:
        """
        Benchmark inference performance.

        Args:
            iterations: Number of iterations

        Returns:
            Performance metrics
        """
        # Create dummy input
        dummy = np.random.randint(0, 255, (*self.input_size, 3), dtype=np.uint8)

        times = []
        for _ in range(iterations):
            start = time.time()
            self.detect(dummy)
            times.append(time.time() - start)

        return {
            'iterations': iterations,
            'avg_latency_ms': np.mean(times) * 1000,
            'min_latency_ms': np.min(times) * 1000,
            'max_latency_ms': np.max(times) * 1000,
            'std_latency_ms': np.std(times) * 1000,
            'avg_fps': 1.0 / np.mean(times)
        }
