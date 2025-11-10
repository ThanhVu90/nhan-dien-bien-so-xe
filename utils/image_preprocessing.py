"""
Advanced Image Preprocessing for License Plate OCR
Xử lý ảnh nâng cao: nghiêng, ngược sáng, tối, mờ
Author: AI Assistant
Date: November 7, 2025
"""

import cv2
import numpy as np
from typing import Tuple, List
import math


class AdvancedImagePreprocessor:
    """Tiền xử lý ảnh nâng cao cho OCR biển số xe"""
    
    def __init__(self):
        """Khởi tạo preprocessor"""
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    def auto_canny(self, image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
        """
        Automatic Canny edge detection với threshold tự động
        
        Args:
            image: Ảnh grayscale
            sigma: Tham số điều chỉnh threshold
            
        Returns:
            Ảnh edges
        """
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        return edged
    
    def deskew_image(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Tự động xoay ảnh để chỉnh góc nghiêng (OPTIMIZED)
        
        Args:
            image: Ảnh đầu vào
            
        Returns:
            (ảnh đã xoay, góc xoay)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Quick check: if image is too small, skip
        if gray.shape[0] < 20 or gray.shape[1] < 20:
            return image, 0.0
        
        # Detect edges
        edges = self.auto_canny(gray)
        
        # Hough Line Transform để tìm góc nghiêng
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                                minLineLength=min(50, gray.shape[1]//2), 
                                maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            return image, 0.0
        
        # Tính góc - chỉ lấy tối đa 20 lines để tăng tốc
        angles = []
        max_lines = min(20, len(lines))
        for i in range(max_lines):
            x1, y1, x2, y2 = lines[i][0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angles.append(angle)
        
        # Lọc các góc outliers
        angles = np.array(angles)
        angles = angles[np.abs(angles) < 45]  # Chỉ lấy góc < 45 độ
        
        if len(angles) == 0:
            return image, 0.0
        
        median_angle = np.median(angles)
        
        # Xoay ảnh
        if abs(median_angle) > 0.5:  # Chỉ xoay nếu góc > 0.5 độ
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
        Tăng cường độ tương phản cho ảnh tối/ngược sáng
        
        Args:
            image: Ảnh đầu vào
            
        Returns:
            Ảnh đã tăng cường
        """
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            l = self.clahe.apply(l)
            
            # Merge and convert back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            return enhanced
        else:
            # Grayscale image
            return self.clahe.apply(image)
    
    def denoise_image(self, image: np.ndarray, fast_mode: bool = True) -> np.ndarray:
        """
        Giảm nhiễu cho ảnh mờ (OPTIMIZED)
        
        Args:
            image: Ảnh đầu vào
            fast_mode: Dùng mode nhanh (h=5) thay vì chính xác (h=10)
            
        Returns:
            Ảnh đã giảm nhiễu
        """
        h_value = 5 if fast_mode else 10
        
        if len(image.shape) == 3:
            # Color image - fastNlMeansDenoisingColored
            denoised = cv2.fastNlMeansDenoisingColored(image, None, h_value, h_value, 7, 21)
        else:
            # Grayscale - fastNlMeansDenoising
            denoised = cv2.fastNlMeansDenoising(image, None, h_value, 7, 21)
        
        return denoised
    
    def sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """
        Làm nét ảnh mờ
        
        Args:
            image: Ảnh đầu vào
            
        Returns:
            Ảnh đã làm nét
        """
        # Unsharp masking
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        return sharpened
    
    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """
        Adaptive threshold cho ảnh có độ sáng không đồng đều
        
        Args:
            image: Ảnh grayscale
            
        Returns:
            Ảnh binary
        """
        # Gaussian adaptive threshold
        binary = cv2.adaptiveThreshold(
            image, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )
        return binary
    
    def remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """
        Loại bỏ bóng và vùng tối không đều
        
        Args:
            image: Ảnh đầu vào
            
        Returns:
            Ảnh đã loại bỏ bóng
        """
        if len(image.shape) == 3:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Dilate để tạo background
        dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
        
        # Median blur
        bg = cv2.medianBlur(dilated, 21)
        
        # Subtract background
        diff = 255 - cv2.absdiff(gray, bg)
        
        # Normalize
        normalized = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)
        
        return normalized
    
    def morphological_operations(self, image: np.ndarray) -> np.ndarray:
        """
        Các phép toán hình thái học để làm sạch ảnh
        
        Args:
            image: Ảnh binary
            
        Returns:
            Ảnh đã làm sạch
        """
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Close gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def preprocess_full_pipeline(self, image: np.ndarray, 
                                 auto_rotate: bool = True,
                                 enhance_contrast: bool = True,
                                 denoise: bool = True,
                                 sharpen: bool = True,
                                 remove_shadow: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Pipeline đầy đủ xử lý ảnh biển số
        
        Args:
            image: Ảnh đầu vào
            auto_rotate: Tự động xoay chỉnh góc
            enhance_contrast: Tăng cường độ tương phản
            denoise: Giảm nhiễu
            sharpen: Làm nét
            remove_shadow: Loại bỏ bóng
            
        Returns:
            (ảnh đã xử lý, thông tin debug)
        """
        debug_info = {}
        processed = image.copy()
        
        # 1. Auto rotate (deskew)
        if auto_rotate:
            processed, angle = self.deskew_image(processed)
            debug_info['rotation_angle'] = angle
        
        # 2. Denoise
        if denoise:
            processed = self.denoise_image(processed)
            debug_info['denoised'] = True
        
        # 3. Remove shadows
        if remove_shadow:
            processed = self.remove_shadows(processed)
            debug_info['shadow_removed'] = True
        
        # 4. Enhance contrast
        if enhance_contrast:
            processed = self.enhance_contrast(processed)
            debug_info['contrast_enhanced'] = True
        
        # 5. Sharpen
        if sharpen:
            processed = self.sharpen_image(processed)
            debug_info['sharpened'] = True
        
        return processed, debug_info
    
    def preprocess_for_ocr(self, image: np.ndarray, 
                          target_size: Tuple[int, int] | None = None) -> np.ndarray:
        """
        Tiền xử lý tối ưu cho OCR
        
        Args:
            image: Ảnh biển số đã crop
            target_size: Kích thước mục tiêu (width, height)
            
        Returns:
            Ảnh đã xử lý sẵn sàng cho OCR
        """
        # Full pipeline
        processed, _ = self.preprocess_full_pipeline(
            image,
            auto_rotate=True,
            enhance_contrast=True,
            denoise=True,
            sharpen=True,
            remove_shadow=True
        )
        
        # Resize if needed
        if target_size:
            processed = cv2.resize(processed, target_size, 
                                  interpolation=cv2.INTER_CUBIC)
        
        return processed


def test_preprocessing(image_path: str):
    """Test preprocessing với một ảnh"""
    import matplotlib.pyplot as plt
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot load image: {image_path}")
        return
    
    # Preprocess
    preprocessor = AdvancedImagePreprocessor()
    processed, debug_info = preprocessor.preprocess_full_pipeline(image)
    
    # Display
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f'Processed\n{debug_info}')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Debug info: {debug_info}")


if __name__ == "__main__":
    # Test
    preprocessor = AdvancedImagePreprocessor()
    print("✅ Advanced Image Preprocessor initialized")
