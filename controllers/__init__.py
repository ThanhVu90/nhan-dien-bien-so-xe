"""
Controllers Package - MVC Architecture
Chứa các controller điều khiển logic giữa Model và View
"""

from .plate_recognition_controller import PlateRecognitionController
from .image_controller import ImageController
from .video_controller import VideoController

__all__ = [
    'PlateRecognitionController',
    'ImageController',
    'VideoController'
]
