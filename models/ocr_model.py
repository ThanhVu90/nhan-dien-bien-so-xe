"""
OCR Model - MVC Architecture
Model chịu trách nhiệm nhận dạng ký tự từ biển số
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
import re


class OCRModel:
    """Model nhận dạng ký tự OCR"""
    
    def __init__(self, lang: str = 'en', use_gpu: bool = False):
        """
        Khởi tạo PaddleOCR model
        
        Args:
            lang: Ngôn ngữ ('en', 'ch', 'vi')
            use_gpu: Sử dụng GPU
        """
        try:
            from paddleocr import PaddleOCR
            
            # Initialize PaddleOCR with compatible parameters
            # Different versions have different parameters
            self.ocr = PaddleOCR(lang=lang)
            
            self.available = True
            print(f"✅ PaddleOCR initialized (lang={lang})")
            
        except ImportError:
            self.available = False
            self.ocr = None
            print("❌ PaddleOCR not available - please install: pip install paddleocr")
        except Exception as e:
            self.available = False
            self.ocr = None
            print(f"❌ Error initializing PaddleOCR: {e}")
    
    def recognize(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Nhận dạng text từ ảnh biển số
        
        Args:
            image: Ảnh biển số đã crop
            
        Returns:
            (text, confidence)
        """
        if not self.available:
            return "", 0.0
        
        try:
            # OCR - use predict() for newer PaddleOCR versions
            results = self.ocr.predict(image)
            
            if not results or len(results) == 0:
                return "", 0.0
            
            # Extract text and confidence from new format
            result_dict = results[0]  # First result
            
            if 'rec_texts' in result_dict and 'rec_scores' in result_dict:
                texts = result_dict['rec_texts']
                scores = result_dict['rec_scores']
                
                if texts and scores:
                    # Combine all detected texts
                    combined_text = ' '.join(texts)
                    avg_confidence = np.mean(scores) if scores else 0.0
                    
                    # Post-process
                    combined_text = self.post_process(combined_text)
                    
                    return combined_text, float(avg_confidence)
            
            return "", 0.0
            
        except Exception as e:
            print(f"⚠️ OCR error: {e}")
            return "", 0.0
    
    def recognize_multiple_attempts(self, image: np.ndarray, max_attempts: int = 3) -> Tuple[str, float]:
        """
        Nhận dạng với nhiều lần thử
        
        Args:
            image: Ảnh biển số
            max_attempts: Số lần thử tối đa
            
        Returns:
            (best_text, best_confidence)
        """
        if not self.available:
            return "", 0.0
        
        results = []
        
        # Attempt 1: Original
        text1, conf1 = self.recognize(image)
        results.append((text1, conf1))
        
        if max_attempts > 1:
            # Attempt 2: Enhanced contrast
            enhanced = self._enhance_contrast(image)
            text2, conf2 = self.recognize(enhanced)
            results.append((text2, conf2))
        
        if max_attempts > 2:
            # Attempt 3: Sharpened
            sharpened = self._sharpen(image)
            text3, conf3 = self.recognize(sharpened)
            results.append((text3, conf3))
        
        # Select best result
        valid_results = [(t, c) for t, c in results if len(t) > 0]
        
        if not valid_results:
            return "", 0.0
        
        # Sort by confidence
        valid_results.sort(key=lambda x: x[1], reverse=True)
        
        return valid_results[0]
    
    def post_process(self, text: str) -> str:
        """
        Hậu xử lý text theo định dạng biển số Việt Nam
        
        Args:
            text: Text gốc
            
        Returns:
            Text đã xử lý
        """
        # Remove spaces, newlines
        text = text.replace(" ", "").replace("\n", "").replace("\t", "")
        text = text.upper()
        
        # Character corrections
        char_map = {
            'O': '0', 'Q': '0', 'I': '1', 'L': '1',
            'Z': '2', 'S': '5', 'B': '8', '|': '1',
            '/': '1', '\\': '1'
        }
        
        # Smart replacement by position
        result = ""
        alphanumeric_count = 0
        
        for char in text:
            if char in ['-', '.']:
                result += char
                continue
            
            if not char.isalnum():
                continue
            
            # Position 0-1: Digits (province code)
            if alphanumeric_count < 2:
                if char.isalpha() and char in char_map:
                    result += char_map[char]
                elif char.isdigit():
                    result += char
                else:
                    result += char_map.get(char, char)
            
            # Position 2: Letter (vehicle type)
            elif alphanumeric_count == 2:
                if char.isdigit():
                    digit_to_letter = {'0': 'O', '1': 'I', '3': 'B', '5': 'S', '8': 'B'}
                    result += digit_to_letter.get(char, char)
                else:
                    result += char
            
            # Position 3+: Digits
            else:
                if char.isalpha() and char in char_map:
                    result += char_map[char]
                elif char.isdigit():
                    result += char
                else:
                    result += char_map.get(char, char)
            
            alphanumeric_count += 1
        
        # Format: 99A-99999 or 99A-999.99
        result = self._format_vietnamese_plate(result)
        
        return result
    
    def _format_vietnamese_plate(self, text: str) -> str:
        """Format theo chuẩn biển số VN"""
        text = text.replace("-", "").replace(".", "")
        
        if len(text) < 3:
            return text
        
        if not (text[0].isdigit() and text[1].isdigit() and text[2].isalpha()):
            return text
        
        province = text[:2]
        vehicle_type = text[2]
        numbers = text[3:]
        
        if len(numbers) == 0:
            return f"{province}{vehicle_type}"
        
        formatted = f"{province}{vehicle_type}-{numbers}"
        
        if len(numbers) >= 5:
            formatted = f"{province}{vehicle_type}-{numbers[:-2]}.{numbers[-2:]}"
        
        return formatted
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Tăng cường contrast"""
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            return enhanced
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def _sharpen(self, image: np.ndarray) -> np.ndarray:
        """Làm nét ảnh"""
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return sharpened
    
    def get_model_info(self) -> Dict:
        """Lấy thông tin model"""
        return {
            'available': self.available,
            'engine': 'PaddleOCR',
            'version': '2.7.3'
        }
