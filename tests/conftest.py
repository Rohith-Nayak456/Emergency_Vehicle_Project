"""Shared pytest fixtures and mock generators for inference testing."""

from pathlib import Path
from typing import Any, List, Optional
import pytest
from PIL import Image

from inference.config import CLASS_MAPPING


class MockBox:
    """Mock representing an Ultralytics YOLO detected box."""

    def __init__(self, xyxy: List[float], conf: float, cls_id: int):
        self.xyxy = [xyxy]
        self.conf = [conf]
        self.cls = [cls_id]


class MockResult:
    """Mock representing an Ultralytics YOLO Results object."""

    def __init__(self, boxes: Optional[List[MockBox]] = None, names: Optional[dict] = None):
        self.boxes = boxes or []
        self.names = names or CLASS_MAPPING


class MockYOLOModel:
    """Mock representing the Ultralytics YOLO model for interface testing only.

    NOTE: Mock predictions must never be presented as actual model results.
    This mock is strictly for interface and regression testing before best.pt is provided.
    """

    def __init__(self, mock_boxes: Optional[List[MockBox]] = None):
        self.names = CLASS_MAPPING.copy()
        self.mock_boxes = mock_boxes or []
        self.predict_call_count = 0
        self.last_source = None

    def predict(self, source: Any, **kwargs: Any) -> List[MockResult]:
        self.predict_call_count += 1
        self.last_source = source
        return [MockResult(boxes=self.mock_boxes, names=self.names)]


@pytest.fixture
def sample_image_file(tmp_path: Path) -> Path:
    """Create a temporary valid RGB JPEG image."""
    img_path = tmp_path / "test_traffic.jpg"
    img = Image.new("RGB", (640, 480), color=(100, 150, 200))
    img.save(img_path, format="JPEG")
    return img_path


@pytest.fixture
def corrupt_image_file(tmp_path: Path) -> Path:
    """Create a corrupt image file."""
    corrupt_path = tmp_path / "corrupt.jpg"
    corrupt_path.write_bytes(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00INVALID_DATA_GARBAGE")
    return corrupt_path


@pytest.fixture
def unsupported_format_file(tmp_path: Path) -> Path:
    """Create a file with an unsupported image extension."""
    txt_path = tmp_path / "document.txt"
    txt_path.write_text("not an image")
    return txt_path


@pytest.fixture
def mock_normal_vehicles_model() -> MockYOLOModel:
    """YOLO model returning only normal traffic (car, bus, motorcycle)."""
    boxes = [
        MockBox(xyxy=[50.0, 60.0, 200.0, 180.0], conf=0.89, cls_id=0),   # car
        MockBox(xyxy=[220.0, 100.0, 380.0, 320.0], conf=0.91, cls_id=4), # bus
        MockBox(xyxy=[10.0, 20.0, 45.0, 90.0], conf=0.76, cls_id=1),     # motorcycle
    ]
    return MockYOLOModel(mock_boxes=boxes)


@pytest.fixture
def mock_ambulance_model() -> MockYOLOModel:
    """YOLO model returning normal traffic plus an ambulance."""
    boxes = [
        MockBox(xyxy=[100.0, 120.0, 350.0, 280.0], conf=0.94, cls_id=6), # ambulance
        MockBox(xyxy=[400.0, 150.0, 520.0, 260.0], conf=0.82, cls_id=0), # car
    ]
    return MockYOLOModel(mock_boxes=boxes)


@pytest.fixture
def mock_police_car_model() -> MockYOLOModel:
    """YOLO model returning a police car."""
    boxes = [
        MockBox(xyxy=[110.0, 130.0, 360.0, 290.0], conf=0.92, cls_id=7), # police_car
    ]
    return MockYOLOModel(mock_boxes=boxes)


@pytest.fixture
def mock_multiple_emergency_model() -> MockYOLOModel:
    """YOLO model returning both ambulance and police car with differing confidences."""
    boxes = [
        MockBox(xyxy=[50.0, 80.0, 250.0, 220.0], conf=0.85, cls_id=7),   # police_car (0.85)
        MockBox(xyxy=[300.0, 100.0, 580.0, 340.0], conf=0.96, cls_id=6), # ambulance (0.96)
        MockBox(xyxy=[10.0, 10.0, 40.0, 80.0], conf=0.70, cls_id=1),     # motorcycle
    ]
    return MockYOLOModel(mock_boxes=boxes)


@pytest.fixture
def mock_empty_detections_model() -> MockYOLOModel:
    """YOLO model returning zero detections."""
    return MockYOLOModel(mock_boxes=[])
