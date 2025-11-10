"""
Image Processor Model - MVC Architecture
Model xử lý ảnh: deskew, denoise, enhance, etc.
"""

import cv2
import numpy as np
from typing import Tuple, Dict
import math


class ImageProcessorModel:
    """Model xử lý ảnh nâng cao"""
    
    def __init__(self):
        """Khởi tạo image processor"""
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def deskew(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Tự động xoay ảnh để chỉnh góc nghiêng
        
        Args:
            image: Ảnh đầu vào
            
        Returns:
            (ảnh đã xoay, góc xoay)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Quick check
        if gray.shape[0] < 20 or gray.shape[1] < 20:
            return image, 0.0
        
        # Auto Canny
        v = np.median(gray)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(gray, lower, upper)
        
        # Hough Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50,
                                minLineLength=min(50, gray.shape[1]//2),
                                maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            return image, 0.0
        
        # Calculate angles (limit to 20 lines for speed)
        angles = []
        max_lines = min(20, len(lines))
        for i in range(max_lines):
            x1, y1, x2, y2 = lines[i][0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)
        
        angles = np.array(angles)
        angles = angles[np.abs(angles) < 45]
        
        if len(angles) == 0:
            return image, 0.0
        
        median_angle = np.median(angles)
        
        # Rotate
        if abs(median_angle) > 0.5:
            h, w = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
            return rotated, median_angle
        
        return image, 0.0
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Tăng cường độ tương phản
        
        Args:
            image: Ảnh đầu vào
            
        Returns:
            Ảnh đã tăng cường
        """
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            return enhanced
        else:
            return self.clahe.apply(image)
    
    def denoise(self, image: np.ndarray, fast_mode: bool = True) -> np.ndarray:
        """
        Giảm nhiễu
        
        Args:
            image: Ảnh đầu vào
            fast_mode: Chế độ nhanh
            
        Returns:
            Ảnh đã giảm nhiễu
        """
        h_value = 5 if fast_mode else 10
        
        if len(image.shape) == 3:
            denoised = cv2.fastNlMeansDenoisingColored(image, None, h_value, h_value, 7, 21)
        else:
            denoised = cv2.fastNlMeansDenoising(image, None, h_value, 7, 21)
        
        return denoised
    
    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Làm nét ảnh
        
        Args:
            image: Ảnh đầu vào
            
        Returns:
            Ảnh đã làm nét
        """
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return sharpened
    
    def preprocess(self, 
                   image: np.ndarray,
                   deskew: bool = True,
                   enhance: bool = True,
                   denoise_img: bool = True,
                   sharpen_img: bool = True,
                   fast_mode: bool = True) -> Tuple[np.ndarray, Dict]:
        """
        Pipeline xử lý ảnh đầy đủ
        
        Args:
            image: Ảnh đầu vào
            deskew: Xoay ảnh
            enhance: Tăng cường contrast
            denoise_img: Giảm nhiễu
            sharpen_img: Làm nét
            fast_mode: Chế độ nhanh
            
        Returns:
            (ảnh đã xử lý, debug info)
        """
        debug_info = {}
        processed = image.copy()
        
        # Deskew
        if deskew:
            processed, angle = self.deskew(processed)
            debug_info['rotation_angle'] = angle
        
        # Denoise
        if denoise_img:
            processed = self.denoise(processed, fast_mode)
            debug_info['denoised'] = True
        
        # Enhance contrast
        if enhance:
            processed = self.enhance_contrast(processed)
            debug_info['contrast_enhanced'] = True
        
        # Sharpen
        if sharpen_img:
            processed = self.sharpen(processed)
            debug_info['sharpened'] = True
        
        return processed, debug_info
