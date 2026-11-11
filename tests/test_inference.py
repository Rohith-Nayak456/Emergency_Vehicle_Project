from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src import inference


class FakeTensor:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value

    def tolist(self):
        return self.value


class FakeBox:
    def __init__(self, class_id, confidence, bbox):
        self.cls = FakeTensor(class_id)
        self.conf = FakeTensor(confidence)
        self.xyxy = [FakeTensor(bbox)]


class FakeModel:
    def __init__(self, boxes):
        self.boxes = boxes

    def predict(self, source, verbose=False):
        return [SimpleNamespace(boxes=self.boxes)]


@pytest.fixture
def sample_image(tmp_path):
    path = tmp_path / "traffic.jpg"
    assert cv2.imwrite(str(path), np.zeros((240, 320, 3), dtype=np.uint8))
    return path


def use_fake_model(monkeypatch, tmp_path, boxes):
    output_dir = tmp_path / "output"
    monkeypatch.setattr(inference, "OUTPUT_DIR", output_dir)
    monkeypatch.setattr(inference, "load_model", lambda: FakeModel(boxes))
    return output_dir


def test_invalid_camera(sample_image):
    with pytest.raises(ValueError, match="Use NORTH"):
        inference.run_inference(sample_image, "NORTHEAST")


def test_missing_image():
    with pytest.raises(FileNotFoundError, match="Input image not found"):
        inference.run_inference("missing.jpg", "NORTH")


def test_unreadable_image(tmp_path):
    path = tmp_path / "broken.jpg"
    path.write_text("not an image")
    with pytest.raises(ValueError, match="Unable to read image"):
        inference.run_inference(path, "NORTH")


def test_ambulance_detection(sample_image, tmp_path, monkeypatch):
    output_dir = use_fake_model(
        monkeypatch, tmp_path, [FakeBox(6, 0.94, [20, 30, 180, 150])]
    )
    result = inference.run_inference(sample_image, "north")
    assert result["camera"] == "NORTH"
    assert result["emergency_detected"] is True
    assert result["emergency_type"] == "ambulance"
    assert result["confidence"] == 0.94
    assert result["detections"][0]["is_emergency"] is True
    assert Path(result["annotated_image_path"]).is_file()
    assert Path(result["annotated_image_path"]).parent == output_dir


def test_police_car_detection(sample_image, tmp_path, monkeypatch):
    use_fake_model(
        monkeypatch, tmp_path, [FakeBox(7, 0.91, [25, 35, 190, 160])]
    )
    result = inference.run_inference(sample_image, "WEST")
    assert result["emergency_type"] == "police_car"
    assert result["confidence"] == 0.91


def test_normal_vehicle(sample_image, tmp_path, monkeypatch):
    use_fake_model(
        monkeypatch, tmp_path, [FakeBox(0, 0.88, [20, 30, 180, 150])]
    )
    result = inference.run_inference(sample_image, "SOUTH")
    assert result["emergency_detected"] is False
    assert result["emergency_type"] is None
    assert result["confidence"] == 0.0
    assert result["detections"][0]["class_name"] == "car"
    assert result["detections"][0]["is_emergency"] is False


def test_multiple_emergencies_choose_highest(sample_image, tmp_path, monkeypatch):
    use_fake_model(
        monkeypatch, tmp_path,
        [
            FakeBox(6, 0.84, [10, 20, 100, 120]),
            FakeBox(7, 0.96, [120, 30, 250, 170]),
        ],
    )
    result = inference.run_inference(sample_image, "EAST")
    assert result["emergency_type"] == "police_car"
    assert result["confidence"] == 0.96
    assert len(result["detections"]) == 2


def test_empty_detections(sample_image, tmp_path, monkeypatch):
    use_fake_model(monkeypatch, tmp_path, [])
    result = inference.run_inference(sample_image, "NORTH")
    assert result["emergency_detected"] is False
    assert result["emergency_type"] is None
    assert result["detections"] == []
    assert result["status"] == "success"


def test_missing_model_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "MODEL_PATH", tmp_path / "missing.pt")
    inference.load_model.cache_clear()
    with pytest.raises(FileNotFoundError, match="Model weights not found"):
        inference.load_model()
    inference.load_model.cache_clear()
