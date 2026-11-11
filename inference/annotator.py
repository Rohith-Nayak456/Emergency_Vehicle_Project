"""Image annotation utilities with distinct emergency vehicle highlighting."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from PIL import Image, ImageDraw, ImageFont

from inference.config import DEFAULT_OUTPUT_DIR, EMERGENCY_CLASSES

# Colors in RGB format
COLOR_EMERGENCY_BOX = (231, 76, 60)         # Crimson / Red
COLOR_EMERGENCY_TEXT_BG = (192, 57, 43)     # Darker Red
COLOR_NORMAL_BOX = (46, 204, 113)           # Fresh Green
COLOR_NORMAL_TEXT_BG = (39, 174, 96)        # Darker Green
COLOR_TEXT = (255, 255, 255)                # White


def _load_font(size: int = 14) -> ImageFont.ImageFont:
    """Attempt to load a reasonable font or fall back to default."""
    try:
        # Standard fonts on Windows
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size=size)
        except Exception:
            return ImageFont.load_default()


def annotate_image(
    image_path: Union[str, Path],
    detections: List[Dict[str, Any]],
    output_dir: Optional[Union[str, Path]] = None,
    output_filename: Optional[str] = None,
) -> str:
    """Annotate an image with bounding boxes, labels, and emergency highlights.

    Does NOT overwrite the original image.

    Args:
        image_path: Path to the original input image.
        detections: List of detection dictionaries, each containing:
                    - 'bbox': [x1, y1, x2, y2]
                    - 'class_name': str
                    - 'confidence': float
                    - 'is_emergency': bool
        output_dir: Directory where annotated image should be saved.
                    Defaults to DEFAULT_OUTPUT_DIR.
        output_filename: Optional specific filename. Defaults to a timestamped name.

    Returns:
        Absolute string path to the saved annotated image.
    """
    src_path = Path(image_path).resolve()
    target_dir = Path(output_dir or DEFAULT_OUTPUT_DIR).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    if output_filename:
        out_path = target_dir / output_filename
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:6]
        out_path = target_dir / f"annotated_{src_path.stem}_{timestamp}_{unique_id}{src_path.suffix}"

    with Image.open(src_path).convert("RGB") as img:
        draw = ImageDraw.Draw(img)
        font = _load_font(size=max(14, int(min(img.width, img.height) * 0.025)))

        for det in detections:
            bbox = det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]
            class_name = det.get("class_name", "unknown")
            conf = det.get("confidence", 0.0)
            is_emergency = det.get("is_emergency", class_name in EMERGENCY_CLASSES)

            # Choose visual styling based on emergency status
            if is_emergency:
                box_color = COLOR_EMERGENCY_BOX
                bg_color = COLOR_EMERGENCY_TEXT_BG
                border_width = max(3, int(min(img.width, img.height) * 0.006))
                label = f"EMERGENCY: {class_name.upper()} {conf:.0%}"
            else:
                box_color = COLOR_NORMAL_BOX
                bg_color = COLOR_NORMAL_TEXT_BG
                border_width = max(2, int(min(img.width, img.height) * 0.003))
                label = f"{class_name} {conf:.0%}"

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=border_width)

            # Draw text banner
            # Use textbbox if available (Pillow >= 8.0.0)
            try:
                text_bbox = draw.textbbox((x1, y1), label, font=font)
                text_w = text_bbox[2] - text_bbox[0]
                text_h = text_bbox[3] - text_bbox[1]
            except Exception:
                text_w, text_h = 100, 16

            pad = 4
            text_bg_y1 = max(0, y1 - text_h - (pad * 2))
            text_bg_y2 = y1 if text_bg_y1 < y1 else text_h + (pad * 2)
            text_bg_x2 = min(img.width, x1 + text_w + (pad * 2))

            draw.rectangle([x1, text_bg_y1, text_bg_x2, text_bg_y2], fill=bg_color)
            draw.text((x1 + pad, text_bg_y1 + pad), label, fill=COLOR_TEXT, font=font)

        img.save(out_path)

    return str(out_path.resolve())
