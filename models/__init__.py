"""
Models Package - MVC Architecture
Chứa các model xử lý dữ liệu và business logic
"""

from .plate_detector import PlateDetectorModel
from .ocr_model import OCRModel
from .image_processor import ImageProcessorModel

__all__ = [
    'PlateDetectorModel',
    'OCRModel',
    'ImageProcessorModel'
]
