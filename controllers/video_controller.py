"""
Video Controller - MVC Architecture  
Controller xử lý video và webcam
"""

import cv2
import numpy as np
from typing import Dict
import os

from .plate_recognition_controller import PlateRecognitionController


class VideoController:
    """Controller cho xử lý video và webcam"""
    
    def __init__(self, model_path: str = 'weights/best.pt'):
        """
        Khởi tạo controller
        
        Args:
            model_path: Đường dẫn model
        """
        self.controller = PlateRecognitionController(
            model_path=model_path,
            use_ocr=True,
            use_gpu=False
        )
    
    def process_video(self,
                     video_path: str,
                     conf: float = 0.25,
                     output_path: str | None = None,
                     show: bool = True,
                     process_every_n_frames: int = 1) -> Dict:
        """
        Xử lý video
        
        Args:
            video_path: Đường dẫn video
            conf: Confidence threshold
            output_path: Đường dẫn output video (None = không lưu)
            show: Hiển thị realtime
            process_every_n_frames: Xử lý mỗi N frames
            
        Returns:
            Statistics
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Video writer
        out = None
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        processed_count = 0
        detected_texts = set()
        
        print(f"\n🎥 Processing video: {video_path}")
        print(f"   Resolution: {width}x{height} @ {fps}fps\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every N frames
                if frame_count % process_every_n_frames == 0:
                    results = self.controller.process_image(
                        frame,
                        conf=conf,
                        preprocess=False,  # Skip preprocessing for speed
                        fast_mode=True
                    )
                    
                    # Draw results
                    for result in results:
                        x1, y1, x2, y2 = result['bbox']
                        text = result['plate_text']
                        ocr_conf = result['ocr_confidence']
                        
                        if text:
                            detected_texts.add(text)
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{text} ({ocr_conf:.2f})"
                        cv2.putText(frame, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    processed_count += 1
                
                # Write frame
                if out:
                    out.write(frame)
                
                # Show
                if show:
                    cv2.imshow('Video Processing', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress
                if frame_count % (fps * 5) == 0:
                    print(f"   Processed {frame_count} frames...")
        
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
        
        stats = {
            'total_frames': frame_count,
            'processed_frames': processed_count,
            'unique_plates': len(detected_texts),
            'plates': list(detected_texts)
        }
        
        print(f"\n✅ Video processed!")
        print(f"   Total frames: {frame_count}")
        print(f"   Unique plates: {len(detected_texts)}")
        if detected_texts:
            print(f"   Plates: {', '.join(detected_texts)}")
        
        return stats
    
    def process_webcam(self, conf: float = 0.25, camera_id: int = 0):
        """
        Xử lý webcam realtime
        
        Args:
            conf: Confidence threshold
            camera_id: Camera ID
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")
        
        window_name = 'Webcam - License Plate Detection'
        print("\n📹 Webcam opened. Press 'q' to quit\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process
                results = self.controller.process_image(
                    frame,
                    conf=conf,
                    preprocess=False,
                    fast_mode=True
                )
                
                # Draw results
                for result in results:
                    x1, y1, x2, y2 = result['bbox']
                    text = result['plate_text']
                    ocr_conf = result['ocr_confidence']
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    if text:
                        label = f"{text} ({ocr_conf:.2f})"
                        cv2.putText(frame, label, (x1, y1-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show
                cv2.imshow(window_name, frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Check window closed
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except:
                    pass
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print("✅ Webcam closed")
