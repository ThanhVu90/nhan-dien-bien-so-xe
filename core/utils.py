"""
Utilities - MVC Architecture
Các hàm utility chung
"""

import os
import time
from pathlib import Path
from typing import List
import json


class Utils:
    """Class chứa các utility functions"""
    
    @staticmethod
    def get_image_files(folder_path: str, extensions: List[str] = None) -> List[str]:
        """
        Lấy danh sách file ảnh trong folder
        
        Args:
            folder_path: Đường dẫn folder
            extensions: List extensions (None = default)
            
        Returns:
            List đường dẫn ảnh
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        image_files = []
        for ext in extensions:
            image_files.extend(Path(folder_path).glob(f'*{ext}'))
            image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
        
        return [str(f) for f in image_files]
    
    @staticmethod
    def get_video_files(folder_path: str) -> List[str]:
        """
        Lấy danh sách file video trong folder
        
        Args:
            folder_path: Đường dẫn folder
            
        Returns:
            List đường dẫn video
        """
        extensions = ['.mp4', '.avi', '.mov', '.mkv']
        video_files = []
        
        for ext in extensions:
            video_files.extend(Path(folder_path).glob(f'*{ext}'))
            video_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
        
        return [str(f) for f in video_files]
    
    @staticmethod
    def format_time(seconds: float) -> str:
        """
        Format thời gian
        
        Args:
            seconds: Số giây
            
        Returns:
            String format đẹp
        """
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            secs = seconds % 60
            return f"{minutes}m{secs:.0f}s"
    
    @staticmethod
    def save_json(data: dict, file_path: str):
        """
        Lưu data dạng JSON
        
        Args:
            data: Dictionary data
            file_path: Đường dẫn file
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def load_json(file_path: str) -> dict:
        """
        Load data từ JSON
        
        Args:
            file_path: Đường dẫn file
            
        Returns:
            Dictionary data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def create_timestamp() -> str:
        """
        Tạo timestamp string
        
        Returns:
            Timestamp string
        """
        return time.strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def validate_plate_format(text: str) -> bool:
        """
        Kiểm tra format biển số VN
        
        Args:
            text: Text biển số
            
        Returns:
            True nếu hợp lệ
        """
        import re
        # Format: 99A-99999 or 99A-999.99
        pattern = r'^\d{2}[A-Z]-\d{3,5}\.?\d{0,2}$'
        return bool(re.match(pattern, text))
    
    @staticmethod
    def ensure_dir(directory: str):
        """
        Đảm bảo thư mục tồn tại
        
        Args:
            directory: Đường dẫn thư mục
        """
        os.makedirs(directory, exist_ok=True)
