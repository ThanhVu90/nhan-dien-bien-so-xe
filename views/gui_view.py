"""
GUI View - MVC Architecture
View hiển thị giao diện đồ họa (Tkinter)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import os
from pathlib import Path
import threading


class GUIView:
    """View cho giao diện GUI với Tkinter"""
    
    def __init__(self, controller=None):
        """
        Khởi tạo GUI view
        
        Args:
            controller: Controller để xử lý logic
        """
        self.controller = controller
        self.root = tk.Tk()
        self.root.title("🚗 License Plate Recognition - MVC")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.current_image = None
        self.current_image_path = None
        self.results = []
        self.is_processing = False
        
        # Setup UI
        self._setup_ui()
        
    def _setup_ui(self):
        """Thiết lập giao diện"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🚗 LICENSE PLATE RECOGNITION",
            font=("Arial", 24, "bold"),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=20)
        
        # Main container
        main_container = tk.Frame(self.root, bg='#f0f0f0')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Controls
        left_panel = tk.Frame(main_container, bg='white', width=300, relief=tk.RIDGE, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        self._setup_controls(left_panel)
        
        # Right panel - Display
        right_panel = tk.Frame(main_container, bg='white', relief=tk.RIDGE, bd=2)
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self._setup_display(right_panel)
        
        # Status bar
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            bg='#ecf0f1',
            font=("Arial", 10)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def _setup_controls(self, parent):
        """Thiết lập panel điều khiển"""
        # Title
        controls_title = tk.Label(
            parent,
            text="📋 Controls",
            font=("Arial", 16, "bold"),
            bg='white'
        )
        controls_title.pack(pady=10)
        
        # Buttons
        btn_style = {
            'font': ('Arial', 11),
            'width': 25,
            'height': 2,
            'relief': tk.RAISED,
            'bd': 2
        }
        
        # Single image button
        self.btn_single = tk.Button(
            parent,
            text="🖼️  Nhận diện ảnh đơn",
            command=self.process_single_image,
            bg='#3498db',
            fg='white',
            **btn_style
        )
        self.btn_single.pack(pady=5, padx=10)
        
        # Folder button
        self.btn_folder = tk.Button(
            parent,
            text="📁 Nhận diện folder ảnh",
            command=self.process_folder,
            bg='#2ecc71',
            fg='white',
            **btn_style
        )
        self.btn_folder.pack(pady=5, padx=10)
        
        # Video button
        self.btn_video = tk.Button(
            parent,
            text="🎬 Nhận diện video",
            command=self.process_video,
            bg='#9b59b6',
            fg='white',
            **btn_style
        )
        self.btn_video.pack(pady=5, padx=10)
        
        # Webcam button
        self.btn_webcam = tk.Button(
            parent,
            text="📸 Nhận diện webcam",
            command=self.process_webcam,
            bg='#e74c3c',
            fg='white',
            **btn_style
        )
        self.btn_webcam.pack(pady=5, padx=10)
        
        # Clear button
        self.btn_clear = tk.Button(
            parent,
            text="🗑️  Xóa kết quả",
            command=self.clear_results,
            bg='#95a5a6',
            fg='white',
            **btn_style
        )
        self.btn_clear.pack(pady=5, padx=10)
        
        # Separator
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, pady=20, padx=10)
        
        # Settings section
        settings_title = tk.Label(
            parent,
            text="⚙️  Settings",
            font=("Arial", 14, "bold"),
            bg='white'
        )
        settings_title.pack(pady=5)
        
        # Confidence threshold
        conf_frame = tk.Frame(parent, bg='white')
        conf_frame.pack(pady=10, padx=10, fill=tk.X)
        
        tk.Label(conf_frame, text="Confidence:", bg='white', font=("Arial", 10)).pack(anchor=tk.W)
        self.conf_var = tk.DoubleVar(value=0.5)
        self.conf_scale = tk.Scale(
            conf_frame,
            from_=0.1,
            to=1.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.conf_var,
            bg='white',
            font=("Arial", 9)
        )
        self.conf_scale.pack(fill=tk.X)
        
        # Info section
        info_frame = tk.Frame(parent, bg='#ecf0f1', relief=tk.SUNKEN, bd=1)
        info_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(
            info_frame,
            text="ℹ️  Information",
            font=("Arial", 12, "bold"),
            bg='#ecf0f1'
        ).pack(pady=5)
        
        self.info_text = scrolledtext.ScrolledText(
            info_frame,
            height=10,
            wrap=tk.WORD,
            font=("Courier", 9),
            bg='white'
        )
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initial info
        self._show_info("Welcome to License Plate Recognition!\n\nSelect an option to start.")
        
    def _setup_display(self, parent):
        """Thiết lập panel hiển thị"""
        # Image display
        image_frame = tk.Frame(parent, bg='white')
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Image label with border
        image_border = tk.Frame(image_frame, bg='#bdc3c7', bd=2, relief=tk.SUNKEN)
        image_border.pack(fill=tk.BOTH, expand=True)
        
        self.image_label = tk.Label(
            image_border,
            text="No image loaded\n\nClick a button to start",
            font=("Arial", 14),
            bg='#ecf0f1',
            fg='#7f8c8d'
        )
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Results display
        results_frame = tk.Frame(parent, bg='white', height=200)
        results_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        results_frame.pack_propagate(False)
        
        tk.Label(
            results_frame,
            text="📊 Results",
            font=("Arial", 14, "bold"),
            bg='white'
        ).pack(anchor=tk.W, padx=5, pady=5)
        
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            height=8,
            wrap=tk.WORD,
            font=("Courier", 10),
            bg='#f9f9f9'
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=(0, 5))
        
    def _show_info(self, message):
        """Hiển thị thông tin"""
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, message)
        
    def _update_status(self, message):
        """Cập nhật status bar"""
        self.status_bar.config(text=message)
        self.root.update_idletasks()
        
    def _display_image(self, image_path):
        """Hiển thị ảnh"""
        try:
            # Load image
            image = Image.open(image_path)
            
            # Resize to fit display
            display_width = 800
            display_height = 500
            image.thumbnail((display_width, display_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # Update label
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep reference
            
            self.current_image_path = image_path
            
        except Exception as e:
            messagebox.showerror("Error", f"Cannot display image: {str(e)}")
            
    def _display_results(self, results):
        """Hiển thị kết quả"""
        self.results_text.delete(1.0, tk.END)
        
        if not results:
            self.results_text.insert(tk.END, "No plates detected.")
            return
            
        self.results_text.insert(tk.END, f"Found {len(results)} plate(s):\n\n")
        
        for i, result in enumerate(results, 1):
            text = result.get('text', 'N/A')
            conf = result.get('confidence', 0)
            self.results_text.insert(tk.END, f"{i}. {text} (Confidence: {conf:.2f})\n")
            
    def process_single_image(self):
        """Xử lý ảnh đơn"""
        if self.is_processing:
            messagebox.showwarning("Warning", "Processing in progress...")
            return
            
        # Select image
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        self._update_status("Processing image...")
        self._show_info(f"Processing: {os.path.basename(file_path)}\nPlease wait...")
        
        # Process in thread
        def process():
            try:
                self.is_processing = True
                
                if self.controller:
                    # Use controller
                    results = self.controller.process_image(file_path)
                    
                    # Display results
                    self.root.after(0, lambda: self._display_image(file_path))
                    self.root.after(0, lambda: self._display_results(results))
                    self.root.after(0, lambda: self._update_status(f"✅ Detected {len(results)} plate(s)"))
                    
                    # Show info
                    info = f"File: {os.path.basename(file_path)}\n"
                    info += f"Results: {len(results)} plate(s) detected\n\n"
                    for i, r in enumerate(results, 1):
                        info += f"{i}. {r.get('text', 'N/A')} ({r.get('confidence', 0):.2f})\n"
                    self.root.after(0, lambda: self._show_info(info))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Controller not initialized"))
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {str(e)}"))
                self.root.after(0, lambda: self._update_status("❌ Error"))
            finally:
                self.is_processing = False
                
        threading.Thread(target=process, daemon=True).start()
        
    def process_folder(self):
        """Xử lý folder ảnh"""
        if self.is_processing:
            messagebox.showwarning("Warning", "Processing in progress...")
            return
            
        # Select folder
        folder_path = filedialog.askdirectory(title="Select Folder")
        
        if not folder_path:
            return
            
        self._update_status("Processing folder...")
        self._show_info(f"Processing folder: {os.path.basename(folder_path)}\nPlease wait...")
        
        # Process in thread
        def process():
            try:
                self.is_processing = True
                
                if self.controller:
                    # Get image files
                    image_files = []
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                        image_files.extend(Path(folder_path).glob(ext))
                        
                    total = len(image_files)
                    all_results = []
                    
                    for i, img_path in enumerate(image_files, 1):
                        self.root.after(0, lambda i=i, t=total: self._update_status(f"Processing {i}/{t}..."))
                        
                        results = self.controller.process_image(str(img_path))
                        all_results.append({
                            'file': os.path.basename(img_path),
                            'count': len(results),
                            'plates': [r.get('text', 'N/A') for r in results]
                        })
                        
                    # Show summary
                    info = f"Folder: {os.path.basename(folder_path)}\n"
                    info += f"Total images: {total}\n\n"
                    info += "Results:\n"
                    for r in all_results:
                        info += f"- {r['file']}: {r['count']} plate(s)\n"
                        for plate in r['plates']:
                            info += f"  → {plate}\n"
                            
                    self.root.after(0, lambda: self._show_info(info))
                    self.root.after(0, lambda: self._update_status(f"✅ Processed {total} images"))
                    self.root.after(0, lambda: messagebox.showinfo("Complete", f"Processed {total} images"))
                else:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Controller not initialized"))
                    
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Processing failed: {str(e)}"))
                self.root.after(0, lambda: self._update_status("❌ Error"))
            finally:
                self.is_processing = False
                
        threading.Thread(target=process, daemon=True).start()
        
    def process_video(self):
        """Xử lý video"""
        messagebox.showinfo("Video Processing", "Video processing will be implemented with VideoController")
        
    def process_webcam(self):
        """Xử lý webcam"""
        messagebox.showinfo("Webcam", "Webcam processing will open in a new window")
        
    def clear_results(self):
        """Xóa kết quả"""
        self.image_label.config(
            image='',
            text="No image loaded\n\nClick a button to start"
        )
        self.results_text.delete(1.0, tk.END)
        self._show_info("Results cleared.\n\nReady for new processing.")
        self._update_status("Ready")
        self.current_image_path = None
        
    def run(self):
        """Chạy GUI"""
        self.root.mainloop()
        
    def show(self):
        """Alias for run()"""
        self.run()
