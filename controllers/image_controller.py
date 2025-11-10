"""
Image Controller - MVC Architecture
Controller xử lý ảnh đơn và batch
"""

import cv2
import numpy as np
from typing import Dict, List
import os

from .plate_recognition_controller import PlateRecognitionController


class ImageController:
    """Controller cho xử lý ảnh"""
    
    def __init__(self, model_path: str = 'weights/best.pt'):
        """
        Khởi tạo controller
        
        Args:
            model_path: Đường dẫn model
        """
        self.controller = PlateRecognitionController(
            model_path=model_path,
            use_ocr=True,
            use_gpu=False
        )
    
    def process_image(self, image_path: str, conf: float = 0.25) -> List[Dict]:
        """
        Xử lý ảnh đơn (wrapper cho GUI)
        
        Args:
            image_path: Đường dẫn ảnh
            conf: Confidence threshold
            
        Returns:
            List of results with 'text' and 'confidence' keys
        """
        results = self.controller.process_image(
            image_path,
            conf=conf,
            preprocess=True
        )
        
        # Format results for GUI
        formatted_results = []
        for r in results:
            formatted_results.append({
                'text': r.get('plate_text', 'N/A'),
                'confidence': r.get('ocr_confidence', 0.0),
                'bbox': r.get('bbox', [0, 0, 0, 0]),
                'detection_confidence': r.get('detection_confidence', 0.0)
            })
        
        return formatted_results
    
    def detect_single_image(self, 
                           image_path: str,
                           conf: float = 0.25,
                           save_result: bool = True,
                           output_dir: str = 'results/images') -> List[Dict]:
        """
        Nhận diện ảnh đơn
        
        Args:
            image_path: Đường dẫn ảnh
            conf: Confidence threshold
            save_result: Lưu kết quả
            output_dir: Thư mục output
            
        Returns:
            List of results
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Process
        results = self.controller.process_image(
            image_path,
            conf=conf,
            preprocess=True
        )
        
        print(f"\n✅ Detected {len(results)} plate(s)")
        
        # Save if needed
        if save_result and results:
            self._save_results(image_path, results, output_dir)
        
        return results
    
    def detect_folder(self,
                     folder_path: str,
                     conf: float = 0.25,
                     save_results: bool = True,
                     output_dir: str = 'results/images') -> Dict:
        """
        Nhận diện tất cả ảnh trong folder
        
        Args:
            folder_path: Đường dẫn folder
            conf: Confidence threshold
            save_results: Lưu kết quả
            output_dir: Thư mục output
            
        Returns:
            Statistics
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Process
        stats = self.controller.process_folder(
            folder_path,
            conf=conf,
            output_dir=output_dir if save_results else None,
            preprocess=True
        )
        
        return stats
    
    def _save_results(self, 
                     image_path: str,
                     results: List[Dict],
                     output_dir: str):
        """Lưu kết quả"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Read original image
        image = cv2.imread(image_path)
        if image is None:
            return
        
        result_image = image.copy()
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Draw results
        for idx, result in enumerate(results, 1):
            x1, y1, x2, y2 = result['bbox']
            text = result['plate_text']
            det_conf = result['detection_confidence']
            ocr_conf = result['ocr_confidence']
            
            # Draw box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw text
            label = f"{text}"
            conf_label = f"Det:{det_conf:.2f} OCR:{ocr_conf:.2f}"
            
            cv2.putText(result_image, label, (x1, y1-25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(result_image, conf_label, (x1, y1-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Save plate crop
            plate_path = os.path.join(output_dir, f"{image_name}_plate_{idx}.jpg")
            cv2.imwrite(plate_path, result['plate_image'])
        
        # Save result image
        result_path = os.path.join(output_dir, f"{image_name}_result.jpg")
        cv2.imwrite(result_path, result_image)
        
        print(f"✅ Results saved to: {output_dir}")
