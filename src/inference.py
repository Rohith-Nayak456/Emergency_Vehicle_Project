
from pathlib import Path
from functools import lru_cache
from uuid import uuid4

import cv2
from ultralytics import YOLO


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "best.pt"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Camera directions
CAMERAS = {"NORTH", "SOUTH", "EAST", "WEST"}

# Dataset class mapping
CLASS_NAMES = {
    0: "car",
    1: "motorcycle",
    2: "autorickshaw",
    3: "truck",
    4: "bus",
    5: "bicycle",
    6: "ambulance",
    7: "police_car",
}

EMERGENCY_CLASSES = {"ambulance", "police_car"}


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the trained YOLO model."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {MODEL_PATH}"
        )

    model = YOLO(str(MODEL_PATH))

    names = model.names

    if isinstance(names, list):
        actual_names = {
            i: name for i, name in enumerate(names)
        }
    else:
        actual_names = {
            int(k): str(v) for k, v in names.items()
        }

    if actual_names != CLASS_NAMES:
        raise RuntimeError(
            "Model class names do not match project mapping"
        )

    return model


def run_inference(image_path, camera_direction):
    """
    Run YOLO inference on one uploaded image.

    Returns the agreed team inference dictionary.
    """

    # Validate camera
    if not isinstance(camera_direction, str):
        raise ValueError(
            "Camera direction must be a string"
        )

    camera_direction = camera_direction.strip().upper()

    if camera_direction not in CAMERAS:
        raise ValueError(
            "Invalid camera direction. Use NORTH, SOUTH, EAST or WEST"
        )

    # Validate image path
    image_path = Path(image_path)

    if image_path.suffix.lower() not in {
        ".jpg", ".jpeg", ".png"
    }:
        raise ValueError("Unsupported image format")

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input image not found: {image_path}"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"Unable to read image: {image_path}"
        )

    # Run YOLO
    model = load_model()

    results = model.predict(
        source=image,
        verbose=False
    )

    if not results:
        raise RuntimeError(
            "YOLO returned no inference results"
        )

    result = results[0]

    detections = []
    emergency_candidates = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls.item())

            if class_id not in CLASS_NAMES:
                raise RuntimeError(
                    f"Unexpected class ID: {class_id}"
                )

            class_name = CLASS_NAMES[class_id]
            confidence = float(box.conf.item())

            bbox = [
                int(v) for v in box.xyxy[0].tolist()
            ]

            is_emergency = (
                class_name in EMERGENCY_CLASSES
            )

            detection = {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": bbox,
                "is_emergency": is_emergency,
            }

            detections.append(detection)

            if is_emergency:
                emergency_candidates.append(detection)

    # Select highest-confidence emergency vehicle
    top_emergency = (
        max(
            emergency_candidates,
            key=lambda item: item["confidence"]
        )
        if emergency_candidates else None
    )

    # Annotate image
    annotated = image.copy()

    for detection in detections:

        x1, y1, x2, y2 = detection["bbox"]

        color = (
            (0, 0, 255)
            if detection["is_emergency"]
            else (0, 255, 0)
        )

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            color,
            2
        )

        label = (
            f'{detection["class_name"]} '
            f'{detection["confidence"]:.1%}'
        )

        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # Save annotated image
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_DIR / f"annotated_{uuid4().hex}.jpg"
    )

    if not cv2.imwrite(str(output_path), annotated):
        raise RuntimeError(
            "Failed to save annotated image"
        )

    # Return shared team contract
    return {
        "camera": camera_direction,
        "emergency_detected": top_emergency is not None,
        "emergency_type": (
            top_emergency["class_name"]
            if top_emergency else None
        ),
        "confidence": (
            top_emergency["confidence"]
            if top_emergency else 0.0
        ),
        "detections": detections,
        "annotated_image_path": str(output_path),
        "status": "success",
    }
