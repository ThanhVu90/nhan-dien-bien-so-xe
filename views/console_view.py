"""
Console View - MVC Architecture
View hiển thị giao diện console/terminal
"""

from typing import Dict, List, Any


class ConsoleView:
    """View cho giao diện console"""
    
    def __init__(self):
        """Khởi tạo console view"""
        self.line_width = 70
    
    def show_header(self, title: str):
        """
        Hiển thị header
        
        Args:
            title: Tiêu đề
        """
        print("\n" + "="*self.line_width)
        print(title.center(self.line_width))
        print("="*self.line_width)
    
    def show_menu(self, options: List[str]):
        """
        Hiển thị menu lựa chọn
        
        Args:
            options: List các lựa chọn
        """
        print()
        for i, option in enumerate(options, 1):
            print(f"{i}. {option}")
        print("="*self.line_width)
    
    def get_input(self, prompt: str) -> str:
        """
        Lấy input từ user
        
        Args:
            prompt: Câu hỏi
            
        Returns:
            User input
        """
        return input(f"\n{prompt}: ").strip()
    
    def show_info(self, message: str):
        """
        Hiển thị thông tin
        
        Args:
            message: Thông điệp
        """
        print(f"ℹ️  {message}")
    
    def show_success(self, message: str):
        """
        Hiển thị thành công
        
        Args:
            message: Thông điệp
        """
        print(f"✅ {message}")
    
    def show_error(self, message: str):
        """
        Hiển thị lỗi
        
        Args:
            message: Thông điệp lỗi
        """
        print(f"❌ {message}")
    
    def show_warning(self, message: str):
        """
        Hiển thị cảnh báo
        
        Args:
            message: Thông điệp cảnh báo
        """
        print(f"⚠️  {message}")
    
    def show_results(self, results: List[Dict]):
        """
        Hiển thị kết quả nhận diện
        
        Args:
            results: List kết quả
        """
        print(f"\n📊 Results: {len(results)} plate(s) detected\n")
        
        for idx, result in enumerate(results, 1):
            text = result.get('plate_text', 'N/A')
            det_conf = result.get('detection_confidence', 0.0)
            ocr_conf = result.get('ocr_confidence', 0.0)
            
            print(f"  Plate {idx}:")
            print(f"    Text: {text}")
            print(f"    Detection confidence: {det_conf:.2f}")
            print(f"    OCR confidence: {ocr_conf:.2f}")
            print()
    
    def show_statistics(self, stats: Dict):
        """
        Hiển thị thống kê
        
        Args:
            stats: Dictionary thống kê
        """
        print(f"\n📈 Statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            elif isinstance(value, list):
                print(f"   {key}: {', '.join(map(str, value))}")
            else:
                print(f"   {key}: {value}")
        print()
    
    def show_progress(self, current: int, total: int, message: str = "Processing"):
        """
        Hiển thị progress
        
        Args:
            current: Số hiện tại
            total: Tổng số
            message: Thông điệp
        """
        percentage = (current / total * 100) if total > 0 else 0
        print(f"\r{message}: {current}/{total} ({percentage:.1f}%)", end='', flush=True)
        
        if current >= total:
            print()  # New line when done
    
    def clear_screen(self):
        """Xóa màn hình"""
        import os
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def wait_for_key(self, message: str = "Press Enter to continue"):
        """
        Đợi user nhấn phím
        
        Args:
            message: Thông điệp
        """
        input(f"\n{message}...")
