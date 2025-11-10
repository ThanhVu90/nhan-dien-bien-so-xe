"""
License Plate Detection - Webcam Real-time Detection with OCR
Author: ThanhVu90
Updated: November 7, 2025 - Added PaddleOCR support
"""

from ultralytics import YOLO
import cv2
from utils import create_ocr_processor

class WebcamLicensePlateDetector:
    def __init__(self, model_path='models/best.pt', enable_ocr=True):
        """
        Khởi tạo detector
        
        Args:
            model_path: Đường dẫn đến file model (.pt)
            enable_ocr: Bật OCR (PaddleOCR)
        """
        self.model = YOLO(model_path)
        print(f"✅ Model loaded: {model_path}")
        
        self.enable_ocr = enable_ocr
        if enable_ocr:
            try:
                self.ocr_processor = create_ocr_processor(
                    use_gpu=False,
                    lang='en'
                )
                if self.ocr_processor:
                    print(f"✅ OCR enabled")
                else:
                    print(f"⚠️ OCR disabled: Failed to initialize")
                    self.enable_ocr = False
            except Exception as e:
                print(f"⚠️ OCR disabled: {e}")
                self.enable_ocr = False
                self.ocr_processor = None
        else:
            self.ocr_processor = None
    
    def detect_webcam(self, conf=0.25, camera_id=0):
        """
        Nhận diện biển số từ webcam
        
        Args:
            conf: Confidence threshold
            camera_id: ID của camera (0 = camera mặc định)
        """
        # Mở webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")
        
        window_name = 'License Plate Detection - Webcam'
        print("📹 Webcam opened. Press 'q' to quit or close the window to stop.")

        while True:
            # Đọc frame
            ret, frame = cap.read()
            
            if not ret:
                print("❌ Cannot read frame")
                break
            
            # Predict
            results = self.model.predict(
                source=frame,
                conf=conf,
                verbose=False
            )
            
            # Get detections and perform OCR
            annotated_frame = frame.copy()
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                    
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Draw box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # OCR if enabled
                    if self.enable_ocr and self.ocr_processor:
                        try:
                            plate_img = frame[y1:y2, x1:x2]
                            if plate_img.size > 0:
                                text, ocr_conf = self.ocr_processor.recognize_text(
                                    plate_img, 
                                    preprocess=True,
                                    return_confidence=True
                                )
                                
                                # Draw text
                                label = f"{text} ({ocr_conf:.2f})"
                                cv2.putText(annotated_frame, label, (x1, y1-10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        except Exception as e:
                            pass
            
            # Hiển thị
            cv2.imshow(window_name, annotated_frame)

            # Nhấn 'q' để thoát
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Nếu người dùng đóng cửa sổ (nhấn X), getWindowProperty sẽ trả về <1
            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("Window closed by user (X). Exiting...")
                    break
            except Exception:
                # một số backend không hỗ trợ getWindowProperty, bỏ qua nếu lỗi
                pass
        
        # Giải phóng
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Webcam closed")

def main():
    # Khởi tạo detector
    print("\n" + "="*60)
    print("LICENSE PLATE DETECTION - WEBCAM MODE WITH OCR")
    print("="*60)
    
    enable_ocr = input("Enable OCR? (y/n, default=y): ").strip().lower()
    enable_ocr = enable_ocr != 'n'
    
    detector = WebcamLicensePlateDetector('models/best.pt', enable_ocr=enable_ocr)
    
    print("Press 'q' to quit")
    print("="*60)
    
    # Bắt đầu detect
    detector.detect_webcam()

if __name__ == "__main__":
    main()