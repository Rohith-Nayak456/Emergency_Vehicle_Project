"""Input validation routines for camera directions and image files."""

from pathlib import Path
from typing import Union
from PIL import Image, UnidentifiedImageError

from inference.config import (
    SUPPORTED_IMAGE_EXTENSIONS,
    VALID_CAMERA_DIRECTIONS,
)
from inference.exceptions import (
    InvalidCameraDirectionError,
    InvalidImageError,
)


def validate_camera_direction(camera_direction: str) -> str:
    """Validate camera direction against the four permissible values.

    Args:
        camera_direction: Direction string (must be NORTH, SOUTH, EAST, or WEST).

    Returns:
        The validated uppercase camera direction.

    Raises:
        InvalidCameraDirectionError: If the direction is invalid.
    """
    if not isinstance(camera_direction, str):
        raise InvalidCameraDirectionError(
            f"Camera direction must be a string, got {type(camera_direction).__name__}"
        )

    direction = camera_direction.strip().upper()
    if direction not in VALID_CAMERA_DIRECTIONS:
        allowed = ", ".join(VALID_CAMERA_DIRECTIONS)
        raise InvalidCameraDirectionError(
            f"Invalid camera direction '{camera_direction}'. Allowed values: {allowed}"
        )
    return direction


def validate_image_file(image_path: Union[str, Path]) -> Path:
    """Validate that an image exists, has an allowed extension, and is not corrupt.

    Args:
        image_path: Path to the image file.

    Returns:
        Resolved Path object pointing to the valid image.

    Raises:
        InvalidImageError: If the file does not exist, has an unsupported
                           extension, or is corrupted.
    """
    if image_path is None:
        raise InvalidImageError("Image path cannot be None")

    path = Path(image_path).resolve()

    if not path.exists():
        raise InvalidImageError(f"Image file does not exist: {path}")

    if not path.is_file():
        raise InvalidImageError(f"Path is not a regular file: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
        allowed = ", ".join(SUPPORTED_IMAGE_EXTENSIONS)
        raise InvalidImageError(
            f"Unsupported image extension '{suffix}' for {path.name}. Allowed: {allowed}"
        )

    # Validate image file integrity
    try:
        with Image.open(path) as img:
            img.verify()
        # Verify only reads header; reopen and perform a light transpose to ensure full file readability
        with Image.open(path) as img:
            img.load()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError(
            f"Corrupt or unreadable image file '{path.name}': {exc}"
        ) from exc

    return path
