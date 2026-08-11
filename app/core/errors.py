"""
Explicit Domain Exception Hierarchy.
Prevents low-level OpenCV/Paddle exceptions from leaking unhandled.
"""

class DomainError(Exception):
    """Base class for all domain errors."""
    pass

class InvalidImageError(DomainError):
    """Raised when input image is invalid or unreadable."""
    pass

class UnsupportedImageError(DomainError):
    """Raised when image format or color space is unsupported."""
    pass

class DocumentBoundaryError(DomainError):
    """Raised when document boundary detection fails or has low confidence."""
    pass

class PreprocessingError(DomainError):
    """Raised when a preprocessing stage fails."""
    pass

class OCRError(DomainError):
    """Raised when OCR engine fails to execute."""
    pass

class ConfigurationError(DomainError):
    """Raised when a document configuration is malformed or invalid."""
    pass

class ClassificationError(DomainError):
    """Raised when document classification fails or is ambiguous."""
    pass

class ExtractionError(DomainError):
    """Raised when field extraction fails."""
    pass

class ValidationError(DomainError):
    """Raised when validation execution fails."""
    pass
