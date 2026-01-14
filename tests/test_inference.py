"""
Tests for Jetson Object Detection.
Run with: pytest tests/ -v
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestConfiguration:
    """Test configuration and setup."""

    def test_requirements_exists(self):
        """Check requirements.txt exists."""
        assert (Path(__file__).parent.parent / "requirements.txt").exists()

    def test_has_tensorrt_dependency(self):
        """Check TensorRT is in requirements."""
        with open(Path(__file__).parent.parent / "requirements.txt") as f:
            content = f.read().lower()
        assert "tensorrt" in content or "onnxruntime" in content

    def test_jetson_detection_module_exists(self):
        """Check main detection module exists."""
        module_dir = Path(__file__).parent.parent / "jetson_detection"
        assert module_dir.exists()
        assert len(list(module_dir.glob("*.py"))) > 0


class TestModels:
    """Test model configuration."""

    def test_models_directory_exists(self):
        """Check models directory exists."""
        assert (Path(__file__).parent.parent / "models").exists()


class TestBenchmarks:
    """Test benchmark results."""

    def test_benchmark_logs_exist(self):
        """Check benchmark logs directory exists."""
        assert (Path(__file__).parent.parent / "logs").exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
