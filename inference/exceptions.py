"""Custom exceptions for the emergency vehicle inference module.

Provides structured error types conforming to the inference error contract.
"""


class InferenceBaseError(Exception):
    """Base exception for all inference errors."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ModelNotFoundError(InferenceBaseError):
    """Raised when the specified YOLO model weights file is missing."""
    pass


class ModelLoadError(InferenceBaseError):
    """Raised when loading the model into memory fails."""
    pass


class InvalidCameraDirectionError(InferenceBaseError):
    """Raised when an invalid camera direction is supplied."""
    pass


class InvalidImageError(InferenceBaseError):
    """Raised when the provided image is missing, corrupt, or unsupported."""
    pass


class InferenceExecutionError(InferenceBaseError):
    """Raised when inference execution fails during prediction."""
    pass
