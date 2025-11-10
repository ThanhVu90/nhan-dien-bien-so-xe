"""
Plate Detector Model - MVC Architecture
Model chịu trách nhiệm phát hiện biển số xe sử dụng YOLO
"""

from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import os


class PlateDetectorModel:
    """Model phát hiện biển số xe"""
    
    def __init__(self, model_path: str = 'weights/best.pt'):
        """
        Khởi tạo YOLO detector model
        
        Args:
            model_path: Đường dẫn đến YOLO model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = YOLO(model_path)
        self.model_path = model_path
        print(f"✅ YOLO model loaded: {model_path}")
    
    def detect(self, image: np.ndarray, conf: float = 0.25) -> List[Dict]:
        """
        Phát hiện biển số trong ảnh
        
        Args:
            image: Ảnh đầu vào (numpy array)
            conf: Confidence threshold
            
        Returns:
            List of detections, mỗi detection có:
            {
                'bbox': (x1, y1, x2, y2),
                'confidence': float,
                'plate_image': np.ndarray
            }
        """
        results = self.model.predict(source=image, conf=conf, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                # Crop plate image
                plate_img = image[y1:y2, x1:x2]
                
                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'plate_image': plate_img.copy()
                }
                
                detections.append(detection)
        
        return detections
    
    def detect_from_file(self, image_path: str, conf: float = 0.25) -> List[Dict]:
        """
        Phát hiện biển số từ file ảnh
        
        Args:
            image_path: Đường dẫn file ảnh
            conf: Confidence threshold
            
        Returns:
            List of detections
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        return self.detect(image, conf)
    
    def get_model_info(self) -> Dict:
        """
        Lấy thông tin về model
        
        Returns:
            Dictionary chứa thông tin model
        """
        return {
            'model_path': self.model_path,
            'model_type': 'YOLOv8',
            'task': 'object_detection',
            'classes': ['LicensePlate']
        }
