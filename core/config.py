"""
Configuration - MVC Architecture
Cấu hình chung cho toàn bộ ứng dụng
"""

import os


class Config:
    """Class cấu hình"""
    
    # Model paths
    YOLO_MODEL_PATH = 'weights/best.pt'
    OCR_MODEL_PATH = 'ocr_models/'
    
    # Default settings
    DEFAULT_CONFIDENCE = 0.25
    USE_GPU = False
    OCR_LANGUAGE = 'en'
    
    # Processing settings
    PREPROCESS_IMAGES = True
    FAST_MODE = True
    PROCESS_EVERY_N_FRAMES = 1  # For video
    
    # Output paths
    OUTPUT_DIR = 'results'
    IMAGE_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'images')
    VIDEO_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'videos')
    
    # Display settings
    SHOW_REALTIME = True
    SAVE_RESULTS = True
    
    # Vietnamese plate format
    PLATE_FORMAT_REGEX = r'^\d{2}[A-Z]-\d{3,5}\.?\d{0,2}$'
    
    # Logging
    VERBOSE = True
    LOG_FILE = 'app.log'
    
    @classmethod
    def get_config(cls) -> dict:
        """
        Lấy tất cả config dạng dictionary
        
        Returns:
            Config dictionary
        """
        return {
            'yolo_model_path': cls.YOLO_MODEL_PATH,
            'default_confidence': cls.DEFAULT_CONFIDENCE,
            'use_gpu': cls.USE_GPU,
            'ocr_language': cls.OCR_LANGUAGE,
            'preprocess_images': cls.PREPROCESS_IMAGES,
            'fast_mode': cls.FAST_MODE,
            'output_dir': cls.OUTPUT_DIR
        }
    
    @classmethod
    def create_directories(cls):
        """Tạo các thư mục cần thiết"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.IMAGE_OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.VIDEO_OUTPUT_DIR, exist_ok=True)
