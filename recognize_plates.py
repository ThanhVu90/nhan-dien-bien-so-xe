"""
License Plate Detection and Recognition with PaddleOCR
Phát hiện và nhận dạng biển số xe với xử lý ảnh nâng cao
Author: AI Assistant
Date: November 7, 2025
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Dict
from utils import create_ocr_processor


class LicensePlateRecognitionSystem:
    """Hệ thống nhận dạng biển số hoàn chỉnh: YOLO + PaddleOCR"""
    
    def __init__(self, 
                 detection_model_path: str = 'models/best.pt',
                 use_paddle_ocr: bool = True,
                 use_gpu: bool = False,
                 ocr_lang: str = 'en'):
        """
        Khởi tạo hệ thống
        
        Args:
            detection_model_path: Đường dẫn YOLO model
            use_paddle_ocr: Sử dụng PaddleOCR (nếu False dùng EasyOCR)
            use_gpu: Sử dụng GPU
            ocr_lang: Ngôn ngữ OCR ('en', 'ch', 'vi')
        """
        # Load YOLO detection model
        if not os.path.exists(detection_model_path):
            raise FileNotFoundError(f"Detection model not found: {detection_model_path}")
        
        self.detection_model = YOLO(detection_model_path)
        print(f"✅ YOLO detection model loaded: {detection_model_path}")
        
        # Initialize OCR processor with PaddleOCR v2.7.3
        self.ocr_processor = create_ocr_processor(
            use_gpu=use_gpu,
            lang=ocr_lang
        )
        
        if self.ocr_processor is None:
            raise RuntimeError("Failed to initialize PaddleOCR processor")
        
        print(f"✅ OCR processor initialized")
    
    def detect_plates(self, image: np.ndarray, conf: float = 0.25) -> List[Dict]:
        """
        Phát hiện biển số trong ảnh
        
        Args:
            image: Ảnh đầu vào
            conf: Confidence threshold
            
        Returns:
            List các detection results
        """
        results = self.detection_model.predict(source=image, conf=conf, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                
                # Crop plate
                plate_img = image[y1:y2, x1:x2]
                
                detection = {
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'plate_image': plate_img
                }
                
                detections.append(detection)
        
        return detections
    
    def recognize_plate(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """
        Nhận dạng text từ ảnh biển số
        
        Args:
            plate_image: Ảnh biển số đã crop
            
        Returns:
            (text, confidence)
        """
        text, conf = self.ocr_processor.recognize_with_multiple_attempts(
            plate_image, 
            max_attempts=3
        )
        return text, conf
    
    def process_image(self, 
                     image_path: str,
                     conf: float = 0.25,
                     save_result: bool = True,
                     output_dir: str = 'results/paddleocr') -> List[Dict]:
        """
        Xử lý ảnh: Phát hiện và nhận dạng biển số
        
        Args:
            image_path: Đường dẫn ảnh
            conf: Confidence threshold
            save_result: Lưu kết quả
            output_dir: Thư mục lưu kết quả
            
        Returns:
            List kết quả
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        
        image_name = Path(image_path).stem
        
        # Detect plates
        detections = self.detect_plates(image, conf)
        
        print(f"\n🔍 Detected {len(detections)} license plate(s)")
        
        results = []
        
        # Recognize each plate
        for idx, detection in enumerate(detections, 1):
            plate_img = detection['plate_image']
            
            # OCR
            text, ocr_conf = self.recognize_plate(plate_img)
            
            result = {
                'text': text,
                'detection_confidence': detection['confidence'],
                'ocr_confidence': ocr_conf,
                'bbox': detection['bbox'],
                'plate_image': plate_img
            }
            
            results.append(result)
            
            print(f"  Plate {idx}: '{text}' | Det: {detection['confidence']:.2f} | OCR: {ocr_conf:.2f}")
        
        # Save results
        if save_result and results:
            os.makedirs(output_dir, exist_ok=True)
            
            # Draw results on image
            result_image = image.copy()
            for idx, result in enumerate(results, 1):
                x1, y1, x2, y2 = result['bbox']
                text = result['text']
                det_conf = result['detection_confidence']
                ocr_conf = result['ocr_confidence']
                
                # Draw box
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw text
                label = f"{text}"
                conf_label = f"Det:{det_conf:.2f} OCR:{ocr_conf:.2f}"
                
                cv2.putText(result_image, label, (x1, y1-25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(result_image, conf_label, (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Save individual plate
                plate_path = os.path.join(output_dir, f"{image_name}_plate_{idx}.jpg")
                cv2.imwrite(plate_path, result['plate_image'])
            
            # Save result image
            result_path = os.path.join(output_dir, f"{image_name}_result.jpg")
            cv2.imwrite(result_path, result_image)
            print(f"✅ Results saved to: {output_dir}")
        
        return results
    
    def process_folder(self, 
                      folder_path: str,
                      conf: float = 0.25,
                      output_dir: str = 'results/paddleocr',
                      max_images: int | None = None) -> Dict:
        """
        Xử lý tất cả ảnh trong folder
        
        Args:
            folder_path: Đường dẫn folder
            conf: Confidence threshold
            output_dir: Thư mục lưu kết quả
            max_images: Số ảnh tối đa (None = all)
            
        Returns:
            Thống kê kết quả
        """
        # Get all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(folder_path).glob(f'*{ext}'))
            image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"\n📁 Found {len(image_files)} images to process\n")
        
        all_results = []
        total_plates = 0
        
        for idx, img_path in enumerate(image_files, 1):
            print(f"{'='*70}")
            print(f"[{idx}/{len(image_files)}] Processing: {img_path.name}")
            print(f"{'='*70}")
            
            try:
                results = self.process_image(str(img_path), conf, True, output_dir)
                all_results.extend(results)
                total_plates += len(results)
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
        # Statistics
        stats = {
            'total_images': len(image_files),
            'total_plates': total_plates,
            'avg_plates_per_image': total_plates / len(image_files) if image_files else 0
        }
        
        print(f"\n{'='*70}")
        print("✅ PROCESSING COMPLETED")
        print(f"{'='*70}")
        print(f"📊 Statistics:")
        print(f"   Total images: {stats['total_images']}")
        print(f"   Total plates detected: {stats['total_plates']}")
        print(f"   Average plates/image: {stats['avg_plates_per_image']:.2f}")
        print(f"📁 Results saved to: {output_dir}")
        print(f"{'='*70}\n")
        
        return stats
    
    def process_video(self,
                     video_path: str,
                     conf: float = 0.25,
                     output_path: str = 'results/paddleocr/output.mp4',
                     show: bool = True,
                     process_every_n_frames: int = 1) -> Dict:
        """
        Xử lý video
        
        Args:
            video_path: Đường dẫn video
            conf: Confidence threshold
            output_path: Đường dẫn video output
            show: Hiển thị realtime
            process_every_n_frames: Xử lý mỗi N frames
            
        Returns:
            Thống kê
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_count = 0
        detected_texts = set()
        
        print(f"\n🎥 Processing video: {video_path}")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Process every {process_every_n_frames} frame(s)\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every N frames
                if frame_count % process_every_n_frames == 0:
                    # Detect and recognize
                    detections = self.detect_plates(frame, conf)
                    
                    for detection in detections:
                        x1, y1, x2, y2 = detection['bbox']
                        plate_img = detection['plate_image']
                        
                        # OCR
                        text, ocr_conf = self.recognize_plate(plate_img)
                        
                        if text:
                            detected_texts.add(text)
                        
                        # Draw
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        label = f"{text} ({ocr_conf:.2f})"
                        cv2.putText(frame, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    processed_count += 1
                
                # Write frame
                out.write(frame)
                
                # Show
                if show:
                    cv2.imshow('License Plate Recognition', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress
                if frame_count % (fps * 10) == 0:  # Every 10 seconds
                    print(f"   Processed {frame_count} frames...")
        
        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
        
        stats = {
            'total_frames': frame_count,
            'processed_frames': processed_count,
            'unique_plates': len(detected_texts),
            'plates': list(detected_texts)
        }
        
        print(f"\n✅ Video processed: {output_path}")
        print(f"📊 Statistics:")
        print(f"   Total frames: {stats['total_frames']}")
        print(f"   Processed frames: {stats['processed_frames']}")
        print(f"   Unique plates: {stats['unique_plates']}")
        print(f"   Plates: {', '.join(stats['plates'])}\n")
        
        return stats


def main():
    """Main function - CLI interface"""
    print("\n" + "="*70)
    print("LICENSE PLATE DETECTION & RECOGNITION WITH PADDLEOCR")
    print("="*70)
    print("1. Process single image")
    print("2. Process folder")
    print("3. Process video")
    print("="*70)
    
    choice = input("\nChoose mode (1/2/3): ").strip()
    
    # Initialize system
    print("\n🔧 Initializing system...")
    try:
        system = LicensePlateRecognitionSystem(
            detection_model_path='models/best.pt',
            use_paddle_ocr=True,
            use_gpu=False,
            ocr_lang='en'
        )
        print("✅ System initialized!\n")
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    if choice == '1':
        image_path = input("Enter image path: ").strip()
        system.process_image(image_path)
        
    elif choice == '2':
        folder_path = input("Enter folder path: ").strip()
        max_images_input = input("Max images (Enter = all): ").strip()
        max_images: int | None = int(max_images_input) if max_images_input else None
        system.process_folder(folder_path, max_images=max_images)
        
    elif choice == '3':
        video_path = input("Enter video path: ").strip()
        process_every = input("Process every N frames (default=1): ").strip()
        process_every = int(process_every) if process_every else 1
        system.process_video(video_path, process_every_n_frames=process_every)
        
    else:
        print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
