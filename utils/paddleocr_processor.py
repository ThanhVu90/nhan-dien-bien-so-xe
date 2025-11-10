"""
PaddleOCR Processor for License Plate Recognition
Sử dụng PaddleOCR với tiền xử lý ảnh nâng cao
Version: 2.7.3 với PaddlePaddle 2.6.1
Author: AI Assistant
Date: November 7, 2025
"""

import cv2
import numpy as np
import os
from typing import List, Tuple, Dict, Optional
from .image_preprocessing import AdvancedImagePreprocessor


class PaddleOCRProcessor:
    """OCR Processor sử dụng PaddleOCR"""
    
    def __init__(self, 
                 lang: str = 'en',
                 use_gpu: bool = False,
                 use_angle_cls: bool = True):
        """
        Khởi tạo PaddleOCR
        
        Args:
            lang: Ngôn ngữ ('en', 'ch', 'vi')
            use_gpu: Sử dụng GPU
            use_angle_cls: Sử dụng angle classifier (xoay ảnh tự động)
        """
        try:
            from paddleocr import PaddleOCR
            
            # PaddleOCR 2.7.3 parameters
            self.ocr = PaddleOCR(
                use_angle_cls=use_angle_cls,
                lang=lang,
                use_gpu=use_gpu
            )
            
            self.preprocessor = AdvancedImagePreprocessor()
            self.available = True
            print(f"✅ PaddleOCR v2.7.3 initialized successfully (lang={lang}, gpu={use_gpu})")
            
        except ImportError as e:
            self.available = False
            self.ocr = None
            self.preprocessor = None
            print("❌ PaddleOCR not available. Install: pip install paddlepaddle paddleocr")
        except Exception as e:
            self.available = False
            self.ocr = None
            self.preprocessor = None
            print(f"❌ Error initializing PaddleOCR: {e}")
    
    def preprocess_plate_image(self, image: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý ảnh biển số trước khi OCR
        
        Args:
            image: Ảnh biển số đã crop
            
        Returns:
            Ảnh đã xử lý
        """
        if self.preprocessor is None:
            return image
        
        # Apply full preprocessing pipeline
        processed = self.preprocessor.preprocess_for_ocr(image)
        return processed
    
    def recognize_text(self, 
                      image: np.ndarray, 
                      preprocess: bool = True,
                      return_confidence: bool = True) -> Tuple[str, float]:
        """
        Nhận dạng text từ ảnh biển số
        Cải tiến: Xử lý biển số 2 tầng (đọc từ trên xuống dưới)
        
        Args:
            image: Ảnh biển số
            preprocess: Có tiền xử lý không
            return_confidence: Trả về confidence score
            
        Returns:
            (text, confidence)
        """
        if not self.available:
            return "", 0.0
        
        # Preprocess
        if preprocess:
            image = self.preprocess_plate_image(image)
        
        try:
            # OCR
            results = self.ocr.ocr(image, cls=True)
            
            if not results or not results[0]:
                return "", 0.0
            
            # Extract text and confidence
            # Sort by Y coordinate (top to bottom) for 2-layer plates
            text_items = []
            
            for line in results[0]:
                if line and len(line) >= 2:
                    bbox = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text = line[1][0]  # Text
                    conf = line[1][1]  # Confidence
                    
                    # Calculate Y position (average of all points)
                    y_pos = sum(point[1] for point in bbox) / len(bbox)
                    
                    text_items.append({
                        'text': text,
                        'confidence': conf,
                        'y_pos': y_pos
                    })
            
            # Sort by Y position (top to bottom)
            text_items.sort(key=lambda x: x['y_pos'])
            
            # Combine texts
            texts = [item['text'] for item in text_items]
            confidences = [item['confidence'] for item in text_items]
            
            # Join with space for 2-layer plates, then remove space in post-processing
            combined_text = ' '.join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Post-process
            combined_text = self.post_process_text(combined_text)
            
            if return_confidence:
                return combined_text, float(avg_confidence)
            else:
                return combined_text, float(avg_confidence)
                
        except Exception as e:
            print(f"⚠️ OCR error: {e}")
            return "", 0.0
    
    def recognize_with_multiple_attempts(self, 
                                         image: np.ndarray,
                                         max_attempts: int = 3) -> Tuple[str, float]:
        """
        Nhận dạng với nhiều lần thử và preprocessing khác nhau
        
        Args:
            image: Ảnh biển số
            max_attempts: Số lần thử tối đa
            
        Returns:
            (text tốt nhất, confidence)
        """
        if not self.available:
            return "", 0.0
        
        results = []
        
        # Attempt 1: Full preprocessing
        text1, conf1 = self.recognize_text(image, preprocess=True)
        results.append((text1, conf1, "full_preprocess"))
        
        if max_attempts > 1:
            # Attempt 2: Original image
            text2, conf2 = self.recognize_text(image, preprocess=False)
            results.append((text2, conf2, "original"))
        
        if max_attempts > 2:
            # Attempt 3: Only contrast enhancement
            enhanced = self.preprocessor.enhance_contrast(image)
            text3, conf3 = self.recognize_text(enhanced, preprocess=False)
            results.append((text3, conf3, "contrast_only"))
        
        # Select best result
        valid_results = [(t, c, m) for t, c, m in results if len(t) > 0]
        
        if not valid_results:
            return "", 0.0
        
        # Sort by confidence
        valid_results.sort(key=lambda x: x[1], reverse=True)
        
        best_text, best_conf, method = valid_results[0]
        
        return best_text, best_conf
    
    def post_process_text(self, text: str) -> str:
        """
        Hậu xử lý text theo định dạng biển số Việt Nam
        Cải tiến: Tránh nhầm lẫn ký tự, số và xử lý biển số 2 tầng
        
        Args:
            text: Text nhận dạng được
            
        Returns:
            Text đã xử lý
        """
        # Remove spaces and newlines (xử lý biển số 2 tầng)
        text = text.replace(" ", "").replace("\n", "").replace("\t", "")
        
        # Convert to uppercase
        text = text.upper()
        
        # Dictionary ánh xạ ký tự dễ nhầm lẫn
        # Dựa vào quy tắc biển số Việt Nam
        char_corrections = {
            # Số 0 vs chữ O
            'O': '0',   # O -> 0 (ưu tiên số)
            'Q': '0',   # Q -> 0
            
            # Số 1 vs chữ I, L
            'I': '1',   # I -> 1
            'L': '1',   # L -> 1 (trong một số font)
            
            # Số 2 vs chữ Z
            'Z': '2',   # Z -> 2
            
            # Số 5 vs chữ S
            'S': '5',   # S -> 5 (ở vị trí số)
            
            # Số 6 vs chữ G
            'G': '6',   # G -> 6 (hiếm gặp)
            
            # Số 8 vs chữ B
            'B': '8',   # B -> 8 (ở vị trí số)
            
            # Ký tự không hợp lệ trong biển số VN
            '|': '1',   # Đường thẳng -> 1
            '/': '1',   # Slash -> 1
            '\\': '1',  # Backslash -> 1
        }
        
        # Apply smart replacement dựa trên vị trí
        processed = self._smart_character_replacement(text, char_corrections)
        
        # Remove invalid characters (keep only valid plate characters)
        # Biển số VN: 0-9, A-Z (không có I, O, Q, W), dấu - và .
        valid_chars = "0123456789ABCDEFGHJKLMNPRSTUVXYZ-."
        processed = ''.join([c for c in processed if c in valid_chars])
        
        # Format Vietnamese plate
        processed = self.format_vietnamese_plate(processed)
        
        return processed
    
    def _smart_character_replacement(self, text: str, corrections: dict) -> str:
        """
        Thay thế ký tự thông minh dựa trên vị trí trong biển số
        
        Quy tắc biển số Việt Nam:
        - 2 ký tự đầu: Số (mã tỉnh) - VD: 29, 30, 51
        - Ký tự thứ 3: Chữ cái (loại xe) - VD: A, B, C, D, E, F, G, H, K, L
        - Các ký tự còn lại: Số (có thể có dấu - và .)
        
        Args:
            text: Text gốc
            corrections: Dictionary các ký tự cần sửa
            
        Returns:
            Text đã sửa
        """
        if len(text) < 3:
            return text
        
        result = ""
        
        for i, char in enumerate(text):
            # Bỏ qua ký tự đặc biệt - và .
            if char in ['-', '.']:
                result += char
                continue
            
            # Đếm số ký tự alphanumeric đã thêm (không tính - và .)
            alphanumeric_pos = len([c for c in result if c.isalnum()])
            
            # Vị trí 0-1: Phải là số (mã tỉnh)
            if alphanumeric_pos < 2:
                if char.isalpha() and char in corrections:
                    result += corrections[char]
                elif char.isdigit():
                    result += char
                elif char.isalpha():
                    # Cố gắng convert chữ sang số
                    if char in corrections:
                        result += corrections[char]
                    else:
                        result += char  # Giữ nguyên nếu không biết
                else:
                    result += char
            
            # Vị trí 2: Phải là chữ cái (loại xe)
            elif alphanumeric_pos == 2:
                if char.isdigit():
                    # Nếu là số, cố convert sang chữ gần giống
                    digit_to_letter = {
                        '0': 'O', '1': 'I', '3': 'B', '5': 'S',
                        '6': 'G', '8': 'B'
                    }
                    if char in digit_to_letter:
                        result += digit_to_letter[char]
                    else:
                        result += char
                elif char.isalpha():
                    # Đảm bảo không phải I, O, Q, W (không dùng trong biển số VN)
                    invalid_letters = {'I': 'L', 'O': 'D', 'Q': 'D', 'W': 'V'}
                    result += invalid_letters.get(char, char)
                else:
                    result += char
            
            # Vị trí 3+: Phải là số
            else:
                if char.isalpha() and char in corrections:
                    result += corrections[char]
                elif char.isdigit():
                    result += char
                elif char.isalpha():
                    # Convert chữ sang số
                    if char in corrections:
                        result += corrections[char]
                    else:
                        # Các chữ không có trong corrections
                        letter_to_digit = {
                            'A': '4', 'C': '0', 'D': '0', 'E': '3',
                            'F': '7', 'H': '4', 'J': '1', 'K': '1',
                            'M': '11', 'N': '1', 'P': '9', 'R': '8',
                            'T': '7', 'U': '0', 'V': '5', 'X': '8',
                            'Y': '7'
                        }
                        result += letter_to_digit.get(char, char)
                else:
                    result += char
        
        return result
    
    def _should_replace_to_digit(self, text: str, position: int) -> bool:
        """
        Kiểm tra xem ký tự tại vị trí này có nên thay thế thành số không
        DEPRECATED: Sử dụng _smart_character_replacement thay thế
        
        Args:
            text: Text đầy đủ
            position: Vị trí ký tự
            
        Returns:
            True nếu nên thay thế
        """
        # Count alphanumeric characters before this position
        alphanumeric_count = sum(1 for c in text[:position] if c.isalnum())
        
        # First 2 positions: always digits (province code)
        if alphanumeric_count < 2:
            return True
        # 3rd position: letter (vehicle type)
        elif alphanumeric_count == 2:
            return False
        # Rest: digits
        else:
            return True
    
    def format_vietnamese_plate(self, text: str) -> str:
        """
        Format text theo chuẩn biển số Việt Nam
        Cải tiến: Xử lý biển số 2 tầng và format thông minh hơn
        
        Args:
            text: Text chưa format
            
        Returns:
            Text đã format
        """
        # Remove existing formatting
        text = text.replace("-", "").replace(".", "")
        
        if len(text) < 3:
            return text
        
        # Validate format: 2 digits + 1 letter + digits
        if not (len(text) >= 3 and 
                text[0].isdigit() and 
                text[1].isdigit() and 
                text[2].isalpha()):
            # Invalid format, return as is
            return text
        
        # Extract components
        province = text[:2]        # 29, 30, 51, etc.
        vehicle_type = text[2]     # A, B, C, etc.
        numbers = text[3:]         # 12345, etc.
        
        # Format based on length
        # Standard format: 99A-99999 or 99A-999.99
        
        if len(numbers) == 0:
            return f"{province}{vehicle_type}"
        
        # Format with dash
        formatted = f"{province}{vehicle_type}-{numbers}"
        
        # Add dot before last 2 digits if appropriate
        # Format: 99A-999.99 (for 5 or 6 digit numbers)
        if len(numbers) >= 5:
            # Insert dot before last 2 digits
            formatted = f"{province}{vehicle_type}-{numbers[:-2]}.{numbers[-2:]}"
        
        return formatted
    
    def batch_recognize(self, images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """
        Nhận dạng batch nhiều ảnh
        
        Args:
            images: List các ảnh biển số
            
        Returns:
            List (text, confidence)
        """
        results = []
        for image in images:
            text, conf = self.recognize_with_multiple_attempts(image)
            results.append((text, conf))
        
        return results


def create_ocr_processor(use_gpu: bool = False, lang: str = 'en') -> Optional[PaddleOCRProcessor]:
    """
    Factory function để tạo OCR processor
    
    Args:
        use_gpu: Sử dụng GPU
        lang: Ngôn ngữ
        
    Returns:
        PaddleOCRProcessor instance hoặc None nếu không khởi tạo được
    """
    processor = PaddleOCRProcessor(lang=lang, use_gpu=use_gpu)
    if processor.available:
        return processor
    else:
        print("❌ Cannot initialize PaddleOCR! Please check installation.")
        return None


if __name__ == "__main__":
    # Test
    processor = create_ocr_processor(use_gpu=False)
    if processor:
        print("✅ OCR Processor created successfully")
        print(f"   - PaddleOCR v2.7.3 with PaddlePaddle v2.6.1")
    else:
        print("❌ Failed to create OCR processor")

