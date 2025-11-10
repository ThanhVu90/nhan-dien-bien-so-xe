"""
Plate Recognition Controller - MVC Architecture
Controller chính điều khiển luồng nhận diện biển số
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path

from models import PlateDetectorModel, OCRModel, ImageProcessorModel


class PlateRecognitionController:
    """Controller chính cho hệ thống nhận diện biển số"""
    
    def __init__(self, 
                 model_path: str = 'weights/best.pt',
                 use_ocr: bool = True,
                 use_gpu: bool = False):
        """
        Khởi tạo controller
        
        Args:
            model_path: Đường dẫn YOLO model
            use_ocr: Sử dụng OCR
            use_gpu: Sử dụng GPU
        """
        # Initialize models
        self.detector = PlateDetectorModel(model_path)
        self.image_processor = ImageProcessorModel()
        
        self.use_ocr = use_ocr
        if use_ocr:
            self.ocr = OCRModel(lang='en', use_gpu=use_gpu)
        else:
            self.ocr = None
        
        print(f"✅ PlateRecognitionController initialized")
    
    def process_image(self, 
                     image: np.ndarray | str,
                     conf: float = 0.25,
                     preprocess: bool = True,
                     fast_mode: bool = True) -> List[Dict]:
        """
        Xử lý ảnh: phát hiện và nhận dạng biển số
        
        Args:
            image: Ảnh numpy array hoặc đường dẫn file
            conf: Confidence threshold
            preprocess: Tiền xử lý ảnh
            fast_mode: Chế độ nhanh
            
        Returns:
            List of results:
            {
                'bbox': (x1, y1, x2, y2),
                'detection_confidence': float,
                'plate_text': str,
                'ocr_confidence': float,
                'plate_image': np.ndarray
            }
        """
        # Load image if path
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Cannot read image: {image}")
        
        # Preprocess if needed
        if preprocess:
            image, debug_info = self.image_processor.preprocess(
                image,
                deskew=True,
                enhance=True,
                denoise_img=True,
                sharpen_img=True,
                fast_mode=fast_mode
            )
        
        # Detect plates
        detections = self.detector.detect(image, conf)
        
        results = []
        
        # Process each plate
        for detection in detections:
            plate_img = detection['plate_image']
            
            # OCR if enabled
            if self.use_ocr and self.ocr and self.ocr.available:
                text, ocr_conf = self.ocr.recognize_multiple_attempts(plate_img)
            else:
                text, ocr_conf = "", 0.0
            
            result = {
                'bbox': detection['bbox'],
                'detection_confidence': detection['confidence'],
                'plate_text': text,
                'ocr_confidence': ocr_conf,
                'plate_image': plate_img
            }
            
            results.append(result)
        
        return results
    
    def process_folder(self,
                      folder_path: str,
                      conf: float = 0.25,
                      output_dir: str | None = None,
                      max_images: int | None = None,
                      preprocess: bool = True) -> Dict:
        """
        Xử lý tất cả ảnh trong folder
        
        Args:
            folder_path: Đường dẫn folder
            conf: Confidence threshold
            output_dir: Thư mục output (None = không lưu)
            max_images: Số ảnh tối đa
            preprocess: Tiền xử lý
            
        Returns:
            Statistics dict
        """
        # Get images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(folder_path).glob(f'*{ext}'))
            image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"\n📁 Processing {len(image_files)} images\n")
        
        all_results = []
        total_plates = 0
        
        for idx, img_path in enumerate(image_files, 1):
            print(f"[{idx}/{len(image_files)}] {img_path.name}")
            
            try:
                results = self.process_image(str(img_path), conf, preprocess)
                all_results.extend(results)
                total_plates += len(results)
                
                # Print results
                for i, result in enumerate(results, 1):
                    text = result['plate_text']
                    det_conf = result['detection_confidence']
                    ocr_conf = result['ocr_confidence']
                    print(f"  Plate {i}: '{text}' | Det: {det_conf:.2f} | OCR: {ocr_conf:.2f}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        stats = {
            'total_images': len(image_files),
            'total_plates': total_plates,
            'avg_plates_per_image': total_plates / len(image_files) if image_files else 0,
            'results': all_results
        }
        
        print(f"\n✅ Processing completed!")
        print(f"   Total plates: {total_plates}")
        
        return stats
    
    def get_info(self) -> Dict:
        """
        Lấy thông tin về controller và models
        
        Returns:
            Info dictionary
        """
        return {
            'detector': self.detector.get_model_info(),
            'ocr': self.ocr.get_model_info() if self.ocr else None,
            'use_ocr': self.use_ocr
        }
