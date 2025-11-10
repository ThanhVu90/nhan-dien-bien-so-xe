"""
Main Application - MVC Architecture
License Plate Recognition System
Version: 3.0 MVC
Author: AI Assistant
Date: November 10, 2025
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from controllers import ImageController, VideoController
from views import ConsoleView
from core import Config


class Application:
    """Main Application với MVC Architecture"""
    
    def __init__(self):
        """Khởi tạo ứng dụng"""
        self.view = ConsoleView()
        self.config = Config()
        
        # Create necessary directories
        Config.create_directories()
    
    def run(self):
        """Chạy ứng dụng"""
        self.view.show_header("🚗 LICENSE PLATE RECOGNITION - MVC ARCHITECTURE")
        
        menu_options = [
            "🖼️  Nhận diện ảnh đơn",
            "📁 Nhận diện folder ảnh",
            "🎬 Nhận diện video",
            "📸 Nhận diện webcam",
            "⚙️  Cấu hình",
            "❌ Thoát"
        ]
        
        while True:
            self.view.show_menu(menu_options)
            choice = self.view.get_input("Chọn chức năng (1-6)")
            
            try:
                if choice == '1':
                    self._detect_single_image()
                elif choice == '2':
                    self._detect_folder()
                elif choice == '3':
                    self._detect_video()
                elif choice == '4':
                    self._detect_webcam()
                elif choice == '5':
                    self._show_config()
                elif choice == '6':
                    self.view.show_success("Goodbye!")
                    break
                else:
                    self.view.show_error("Lựa chọn không hợp lệ!")
            
            except KeyboardInterrupt:
                self.view.show_warning("\nĐã hủy thao tác")
            except Exception as e:
                self.view.show_error(f"Lỗi: {e}")
                import traceback
                traceback.print_exc()
    
    def _detect_single_image(self):
        """Nhận diện ảnh đơn"""
        self.view.show_header("NHẬN DIỆN ẢNH ĐƠN")
        
        image_path = self.view.get_input("Nhập đường dẫn ảnh")
        
        if not os.path.exists(image_path):
            self.view.show_error(f"File không tồn tại: {image_path}")
            return
        
        self.view.show_info("Đang khởi tạo controller...")
        controller = ImageController(Config.YOLO_MODEL_PATH)
        
        self.view.show_info("Đang xử lý ảnh...")
        results = controller.detect_single_image(
            image_path,
            conf=Config.DEFAULT_CONFIDENCE,
            save_result=Config.SAVE_RESULTS,
            output_dir=Config.IMAGE_OUTPUT_DIR
        )
        
        self.view.show_results(results)
        self.view.wait_for_key()
    
    def _detect_folder(self):
        """Nhận diện folder ảnh"""
        self.view.show_header("NHẬN DIỆN FOLDER ẢNH")
        
        folder_path = self.view.get_input("Nhập đường dẫn folder")
        
        if not os.path.exists(folder_path):
            self.view.show_error(f"Folder không tồn tại: {folder_path}")
            return
        
        self.view.show_info("Đang khởi tạo controller...")
        controller = ImageController(Config.YOLO_MODEL_PATH)
        
        self.view.show_info("Đang xử lý folder...")
        stats = controller.detect_folder(
            folder_path,
            conf=Config.DEFAULT_CONFIDENCE,
            save_results=Config.SAVE_RESULTS,
            output_dir=Config.IMAGE_OUTPUT_DIR
        )
        
        self.view.show_statistics({
            'Total images': stats['total_images'],
            'Total plates': stats['total_plates'],
            'Average plates/image': f"{stats['avg_plates_per_image']:.2f}"
        })
        self.view.wait_for_key()
    
    def _detect_video(self):
        """Nhận diện video"""
        self.view.show_header("NHẬN DIỆN VIDEO")
        
        video_path = self.view.get_input("Nhập đường dẫn video")
        
        if not os.path.exists(video_path):
            self.view.show_error(f"File không tồn tại: {video_path}")
            return
        
        # Ask for output
        save_output = self.view.get_input("Lưu video kết quả? (y/n)").lower() == 'y'
        output_path = None
        
        if save_output:
            output_path = os.path.join(
                Config.VIDEO_OUTPUT_DIR,
                f"output_{os.path.basename(video_path)}"
            )
        
        self.view.show_info("Đang khởi tạo controller...")
        controller = VideoController(Config.YOLO_MODEL_PATH)
        
        self.view.show_info("Đang xử lý video (nhấn 'q' để dừng)...")
        stats = controller.process_video(
            video_path,
            conf=Config.DEFAULT_CONFIDENCE,
            output_path=output_path,
            show=Config.SHOW_REALTIME,
            process_every_n_frames=Config.PROCESS_EVERY_N_FRAMES
        )
        
        self.view.show_statistics(stats)
        self.view.wait_for_key()
    
    def _detect_webcam(self):
        """Nhận diện webcam"""
        self.view.show_header("NHẬN DIỆN WEBCAM")
        
        camera_id = self.view.get_input("Nhập Camera ID (default=0)")
        camera_id = int(camera_id) if camera_id else 0
        
        self.view.show_info("Đang khởi tạo controller...")
        controller = VideoController(Config.YOLO_MODEL_PATH)
        
        self.view.show_info("Đang mở webcam (nhấn 'q' để dừng)...")
        controller.process_webcam(
            conf=Config.DEFAULT_CONFIDENCE,
            camera_id=camera_id
        )
        
        self.view.show_success("Webcam đã đóng")
        self.view.wait_for_key()
    
    def _show_config(self):
        """Hiển thị cấu hình"""
        self.view.show_header("CẤU HÌNH HỆ THỐNG")
        
        config = Config.get_config()
        self.view.show_statistics(config)
        self.view.wait_for_key()


def main():
    """Main function"""
    try:
        app = Application()
        app.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Chương trình đã dừng!")
    except Exception as e:
        print(f"\n❌ Lỗi nghiêm trọng: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
